import numpy as np
import torch
import torch.nn as nn
import math

__all__ = ['ResNet', 'resnet18']

'''for the paper A Modulation Module for Multi-task Learning with Applications in Image Retrieval,
   a channel-wise learned mask is appended after each ResNet block'''

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def construct_mask(n_conditions, embedding_size):
    masks = torch.nn.Embedding(n_conditions, embedding_size)
    # initialize masks
    mask_array = np.zeros([n_conditions, embedding_size])
    mask_array.fill(0.1)
    mask_len = int(embedding_size / n_conditions)
    for i in range(n_conditions):
        mask_array[i, i*mask_len:(i+1)*mask_len] = 1
    masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
    return masks

class ResNet(nn.Module):

    def __init__(self, block, layers, embedding_size=64, n_condition=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_embed = nn.Linear(256 * block.expansion, embedding_size)

        # define channel-wise mask for each block  
        self.mask1 = construct_mask(n_condition, 64)
        self.mask2 = construct_mask(n_condition, 64)
        self.mask3 = construct_mask(n_condition, 128)
        self.mask4 = construct_mask(n_condition, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, c):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.mul(x, self.mask1(c).unsqueeze(-1).unsqueeze(-1))

        x = self.layer1(x)
        x = torch.mul(x, self.mask2(c).unsqueeze(-1).unsqueeze(-1))
        x = self.layer2(x)
        x = torch.mul(x, self.mask3(c).unsqueeze(-1).unsqueeze(-1))
        x = self.layer3(x)
        x = torch.mul(x, self.mask4(c).unsqueeze(-1).unsqueeze(-1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_embed(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    if pretrained:
        state = model.state_dict()
        loaded_state_dict = torch.load('./saves/resnet18-5c106cde.tar')
        for k in loaded_state_dict:
            if k in state:
                state[k] = loaded_state_dict[k]
        model.load_state_dict(state)
    return model