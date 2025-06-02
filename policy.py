import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from termcolor import cprint
from torch.autograd import Variable
from einops import reduce
from transformers import BertConfig,BertModel,BertTokenizer

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