import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from termcolor import cprint
from torchvision.models import efficientnet_b0
from torch.autograd import Variable
from einops import reduce
from transformers import BertConfig,BertModel,BertTokenizer

class OpenDoBot_v2(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 视觉骨干网络 (使用预训练的 EfficientNet-B0) ---
        # weights = EfficientNet_B0_Weights.IMAGENET1K_V1 # 较新 torchvision
        # self.VisModel = efficientnet_b0(weights=weights)
        self.VisModel = efficientnet_b0(pretrained=True) # _b0 是一个较小的版本

        # 1. 修改 EfficientNet 的第一个卷积层以接受4个输入通道 (RGBD)
        # EfficientNet-B0 的第一个卷积层通常是 self.VisModel.features[0][0]
        original_conv0 = self.VisModel.features[0][0]
        new_conv0 = nn.Conv2d(
            4, # 输入通道变为4
            original_conv0.out_channels,
            kernel_size=original_conv0.kernel_size,
            stride=original_conv0.stride,
            padding=original_conv0.padding,
            bias=original_conv0.bias is not None
        )
        # 复制RGB通道的权重，并初始化深度通道的权重 (例如，使用RGB权重的平均值)
        with torch.no_grad():
            new_conv0.weight[:, :3, :, :] = original_conv0.weight.clone()
            new_conv0.weight[:, 3, :, :] = original_conv0.weight.mean(dim=1, keepdim=False).clone()
            if original_conv0.bias is not None:
                new_conv0.bias = original_conv0.bias.clone()
        self.VisModel.features[0][0] = new_conv0

        # 2. 修改 EfficientNet 的分类头 (全连接层) 以输出256维特征
        # EfficientNet-B0 的分类器是 self.VisModel.classifier
        num_ftrs = self.VisModel.classifier[1].in_features # 获取原分类头输入特征数
        self.VisModel.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), # EfficientNet 默认带一个Dropout
            nn.Linear(num_ftrs, 256) # 输出我们期望的256维
        )
        
        # 可选：如果数据集非常小，可以冻结 EfficientNet 的部分早期层
        # print("冻结 EfficientNet 部分层...")
        # for name, param in self.VisModel.features.named_parameters():
        #     # 例如，冻结前4个stage (features[0] 到 features[3])
        #     if any(f"features.{i}" in name for i in range(4)):
        #          param.requires_grad = False
        #     else:
        #          param.requires_grad = True

        # --- 融合 Transformer (可以适当减小容量) ---
        config = BertConfig(
            hidden_size=256,
            num_attention_heads=4,      # 从8减少到4
            intermediate_size=256 * 2,  # 从256*4减少
            num_hidden_layers=2         # 从4减少到2
        )
        self.Transformer = BertModel(config)
        self.pos_proj = nn.Linear(4, 256) # 假设 qpos 是4维
        self.act_proj = nn.Linear(256, 4) # 假设 action 是4维

        # --- 语言模型 ---
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_text = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, 256)

        # --- 深度模型 ---
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small") # MiDaS_small 更快，DPT_Large 精度更高

        # --- 正确冻结预训练组件的参数 ---
        for param in self.bert_text.parameters():
            param.requires_grad = False
        for param in self.midas.parameters():
            param.requires_grad = False
        
        cprint("OpenDoBot 模型已初始化 (使用 EfficientNet).", "cyan")
        cprint(f"  - 视觉模型: EfficientNet-B0 (预训练), 已适配4通道输入.", "cyan")
        cprint(f"  - 融合 Transformer: {config.num_hidden_layers} 层, {config.num_attention_heads} 头.", "cyan")
        cprint(f"  - BERT 文本编码器和 MiDaS 深度估计器已冻结.", "cyan")


    def forward(self, qpos, imgtop, id=None, lan=None, actions=None):
        batch_size = imgtop.shape[0]

        # 1. 视觉特征提取
        # 确保 MiDaS 在冻结时不参与梯度计算
        with torch.no_grad() if not next(self.midas.parameters()).requires_grad else torch.enable_grad():
            depth = self.midas(imgtop)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        
        # 归一化深度图 (可选，但有助于稳定)
        min_d = reduce(depth, 'b c h w -> b c 1 1', 'min')
        max_d = reduce(depth, 'b c h w -> b c 1 1', 'max')
        depth = (depth - min_d) / (max_d - min_d + 1e-8)

        imgtop_with_depth = torch.cat([imgtop, depth], dim=1) # (B, 4, H, W)
        
        # 使用 VisModel (EfficientNet) 提取特征
        # EfficientNet 的输出直接就是分类前的特征 (B, num_features)
        # 如果 self.VisModel.classifier 被替换为一个 nn.Linear(num_ftrs, 256)
        # 那么 vis 就是 (B, 256)
        vis = self.VisModel(imgtop_with_depth) # (B, 256)

        # 2. 本体感受特征
        pos = self.pos_proj(qpos) # (B, 256)

        # 3. 语言特征
        if id is not None and lan is None:
            current_id_val = id[0].item() if isinstance(id, torch.Tensor) else id[0]
            if current_id_val == 1:
                lan_list = ["pick the black cube"] * batch_size
            elif current_id_val == 2:
                lan_list = ["pick the green cube"] * batch_size
            elif current_id_val == 3:
                lan_list = ["pick the yellow cube"] * batch_size
            else:
                lan_list = [""] * batch_size
        elif isinstance(lan, str):
            lan_list = [lan] * batch_size
        elif isinstance(lan, list):
            assert len(lan) == batch_size, "语言指令列表长度必须与批次大小一致"
            lan_list = lan
        else:
            cprint("警告: 未提供有效的语言指令 (id 或 lan)。使用空字符串。", "yellow")
            lan_list = [""] * batch_size

        text_tokens = self.tokenizer(lan_list, padding='longest', truncation=True,
                                     return_tensors="pt").to(qpos.device)
        
        with torch.no_grad() if not next(self.bert_text.parameters()).requires_grad else torch.enable_grad():
            text_output = self.bert_text(**text_tokens)
        
        text_feat_pooled = text_output.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat_pooled)

        # 4. Transformer 融合
        cls_token_emb = torch.zeros_like(pos).unsqueeze(1) # (B, 1, 256)
        
        condition = torch.stack([pos, vis, text_feat], dim=1) # (B, 3, 256)
        condition_with_cls = torch.cat([cls_token_emb, condition], dim=1) # (B, 4, 256)

        transformer_output = self.Transformer(inputs_embeds=condition_with_cls).last_hidden_state
        action_embedding = transformer_output[:, 0, :]
        action_pred = self.act_proj(action_embedding)

        if actions is not None:
            loss = F.mse_loss(action_pred, actions, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            return loss
        else:
            return action_pred
class OpenDoBot(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.ResNet = ResNet18()
        
        config = BertConfig(hidden_size=256, num_attention_heads=8, 
                                               intermediate_size=256 * 4, num_hidden_layers=4)
        self.Transformer =  BertModel(config)
        self.pos_proj = nn.Linear(4,256)
        self.act_proj = nn.Linear(256,4)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_text = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, 256)

        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        for param in self.bert_text.parameters() and self.midas.parameters():
            param.requires_grad = False
        

    def forward(self, qpos, imgtop, id=None,lan=None, actions=None):
        
        if actions is not None:
            depth = self.midas(imgtop)
            depth = depth.unsqueeze(1)
            imgtop = torch.cat([imgtop, depth], dim=1)
            vis = self.ResNet(imgtop)
            pos = self.pos_proj(qpos)

            if id is not None:
                if id[0] == 1:
                    lan = ["pick the black cube"]*id.shape[0]
                if id[0] == 2:
                    lan = ["pick the green cube"]*id.shape[0]
                if id[0] == 3:
                    lan = ["pick the yellow cube"]*id.shape[0]

            text_tokens = self.tokenizer(lan, padding=True, truncation=True,
                                     return_tensors="pt").to(qpos.device)
            text_feat = self.bert_text(**text_tokens).last_hidden_state[:, 0, :]  
            text_feat = self.text_proj(text_feat)  

            cls=torch.zeros_like(pos).to(pos.device)
            condition = torch.stack([cls, pos, vis, text_feat],dim=1)
            action_pred = self.Transformer(inputs_embeds = condition).last_hidden_state[:,0,:]
            action_pred = self.act_proj(action_pred)
            loss = F.mse_loss(action_pred, actions, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            return loss
        
        else:
            depth = self.midas(imgtop)
            depth = depth.unsqueeze(1)
            imgtop = torch.cat([imgtop, depth], dim=1)
            vis = self.ResNet(imgtop)
            pos = self.pos_proj(qpos)
            if lan is not None:
                lan = [lan]*pos.shape[0]

            text_tokens = self.tokenizer(lan, padding=True, truncation=True,
                                     return_tensors="pt").to(qpos.device)
            text_feat = self.bert_text(**text_tokens).last_hidden_state[:, 0, :]  
            text_feat = self.text_proj(text_feat)

            cls=torch.zeros_like(pos).to(pos.device)
            condition = torch.stack([cls, pos, vis, text_feat],dim=1)
            action_pred = self.Transformer(inputs_embeds = condition).last_hidden_state[:,0,:]
            action_pred = self.act_proj(action_pred)
            return action_pred


class ResNetDownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride[0],
                               padding=1)
        self.bn1 = nn.GroupNorm(out_channels//16,out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride[1],
                               padding=1)
        self.bn2 = nn.GroupNorm(out_channels//16,out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=stride[0],
                      padding=0), nn.GroupNorm(out_channels//16,out_channels))

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class ResNetBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.GroupNorm(out_channels//16,out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn2 = nn.GroupNorm(out_channels//16,out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1),
                                    ResNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [2, 1]),
                                    ResNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1]),
                                    ResNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [2, 1]),
                                    ResNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out