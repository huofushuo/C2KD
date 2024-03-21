from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),

    )


def conv_1x1_bn(num_input_channels, num_mid_channel):
    return nn.Sequential(
        conv1x1(num_input_channels, num_mid_channel),
        nn.BatchNorm2d(num_mid_channel),
        nn.ReLU(inplace=True),
    )


class Shake(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""

    def __init__(self, tea_type):
        super(Shake, self).__init__()
        self.modality = tea_type
        self.fuse1 = conv_bn(64, 64, 2)
        self.fuse2 = conv_1x1_bn(64, 64)
        self.fuse3 = conv_bn(64, 128, 2)
        self.fuse4 = conv_1x1_bn(128, 128)
        self.fuse5 = conv_bn(128, 256, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc = nn.Linear(256, 50)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tea):
        aaa = self.fuse1(tea[0])
        bbb = self.fuse2(tea[1])
        ccc = self.fuse3(aaa + bbb)
        ddd = self.fuse4(tea[2])
        x = self.fuse5(ccc + ddd)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class Fit(nn.Module):
    def __init__(self, tea_type=0):
        super(Fit, self).__init__()
        self.modality = tea_type
        self.fuse1 = ConvReg(64)
        self.fuse2 = ConvReg(64)
        self.fuse3 = ConvReg(128)
        self.fuse4 = ConvReg(256)
        self.fuse5 = ConvReg(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, stu, tea):
        l1 = F.mse_loss(self.fuse1(stu[0]), tea[0])
        l2 = F.mse_loss(self.fuse1(stu[1]), tea[1])
        l3 = F.mse_loss(self.fuse1(stu[2]), tea[2])
        l4 = F.mse_loss(self.fuse1(stu[3]), tea[3])
        l5 = F.mse_loss(self.fuse1(stu[4]), tea[4])
        loss = l1+l2+l3+l4+l5
        return loss




class Fusion(nn.Module):
    def __init__(self, tea_type=0):
        super(Fusion, self).__init__()
        self.modality = tea_type
        # self.fuse = ConvReg(100)

        self.fc = nn.Linear(100, 50)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, stu, tea):
        logits = self.fc(torch.cat([stu, tea], dim=1))
        return logits

class Tea(nn.Module):
    """Teacher proxy for C$^2$KD"""
    def __init__(self, tea_type):
        super(Tea, self).__init__()
        self.modality = tea_type
        if self.modality == 1:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 50)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tea):
        x = tea[4]
        if self.modality == 0:
            (B3, C, H, W) = x.size()
            x = x.view(int(B3/3), -1, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            x = self.avgpool(x)
            x = torch.flatten(x, 3)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 3)

        # x = self.avgpool(tea[4])
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Stu(nn.Module):
    """Student proxy for C$^2$KD"""

    def __init__(self, tea_type):
        super(Stu, self).__init__()
        self.modality = tea_type
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.modality == 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 50)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, stu):
        x = stu[4]
        if self.modality == 1:
            (B3, C, H, W) = x.size()
            x = x.view(int(B3/3), -1, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            x = self.avgpool(x)
            x = torch.flatten(x, 3)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 3)
        # x = self.avgpool(stu[4])
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, ch):
        super(ConvReg, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.bn(x))


class Regress(nn.Module):
    """Simple Linear Regression for hints"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
