import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from termcolor import cprint
from torch.autograd import Variable
from einops import reduce
from transformers import BertConfig,BertModel

class OpenDoBot(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.ResNet = ResNet18()
        
        config = BertConfig(hidden_size=256, num_attention_heads=8, 
                                               intermediate_size=256 * 4, num_hidden_layers=3)
        self.Transformer =  BertModel(config)
        self.pos_proj = nn.Linear(4,256)
        self.act_proj = nn.Linear(256,4)

    def forward(self, qpos, imgtop, lan=None, actions=None):
        if actions is not None:
            vis = self.ResNet(imgtop)
            pos = self.pos_proj(qpos)/100.0
            condition = torch.stack([pos,vis],dim=1)
            action_pred = self.Transformer(inputs_embeds = condition).last_hidden_state[:,0,:]
            action_pred = self.act_proj(action_pred)
            loss = F.mse_loss(action_pred, actions/100.0, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            return loss
        else:
            return qpos


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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
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