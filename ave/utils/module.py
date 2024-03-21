from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        nn.LeakyReLU(0.1, inplace=True),
        # conv1x1(num_mid_channel, num_mid_channel),
        # nn.BatchNorm2d(num_mid_channel),
        # conv3x3(num_mid_channel, num_mid_channel),
        # nn.BatchNorm2d(num_mid_channel),
        # nn.ReLU(inplace=True),
        # conv1x1(num_mid_channel, num_mid_channel),
    )

class Fusion(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=28, x_film=True):
        super(Fusion, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output

# class Fusion(nn.Module):
#     def __init__(self, tea_type=0):
#         super(Fusion, self).__init__()
#         self.modality = tea_type
#         # self.fuse = ConvReg(100)
#
#         self.fc = nn.Linear(512*2, 28)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, stu, tea):
#         aaa = torch.cat([stu, tea], dim=1)
#         logits = self.fc(aaa)
#         return logits

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

        # self.fuse11 = conv_bn(64, 64, 2)
        # self.fuse22 = conv_1x1_bn(64, 64)
        # self.fuse33 = conv_bn(64, 128, 2)
        # self.fuse44 = conv_1x1_bn(128, 128)
        # self.fuse55 = conv_bn(128, 256, 1)
        # if self.modality == 0:
        #     self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        # else:
        #     self.avgpool2 = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(256, 28)


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

        # aaaa = self.fuse11(stu[0])
        # bbbb = self.fuse22(stu[1])
        # cccc = self.fuse33(aaaa + bbbb)
        # dddd = self.fuse44(stu[2])
        # xx = self.fuse55(cccc + dddd)
        # xx = self.avgpool(xx)
        # xx = xx.view(xx.size(0), -1)
        # # x = nn.functional.linear(x, weight, bias)
        x = self.fc(x)
        return x


class Tea(nn.Module):
    """Teacher proxy for C$^2$KD"""

    def __init__(self, ):
        super(Tea, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 28)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tea):
        x = self.avgpool(tea[4])
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Stu(nn.Module):
    """Student proxy for C$^2$KD"""

    def __init__(self, ):
        super(Stu, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 28)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, stu):
        x = self.avgpool(stu[4])
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, ch):
        super(ConvReg, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.bn(x))

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
        B, C, H, W = tea[0].size()
        stu0 = F.interpolate(stu[0], (H, W), mode='bilinear', align_corners=True)
        l1 = F.mse_loss(self.fuse1(stu0), tea[0])
        B, C, H, W = tea[1].size()
        stu1 = F.interpolate(stu[1], (H, W), mode='bilinear', align_corners=True)
        l2 = F.mse_loss(self.fuse2(stu1), tea[1])
        B, C, H, W = tea[2].size()
        stu2 = F.interpolate(stu[2], (H, W), mode='bilinear', align_corners=True)
        l3 = F.mse_loss(self.fuse3(stu2), tea[2])
        B, C, H, W = tea[3].size()
        stu3 = F.interpolate(stu[3], (H, W), mode='bilinear', align_corners=True)
        l4 = F.mse_loss(self.fuse4(stu3), tea[3])
        B, C, H, W = tea[4].size()
        stu4 = F.interpolate(stu[4], (H, W), mode='bilinear', align_corners=True)
        l5 = F.mse_loss(self.fuse5(stu4), tea[4])
        return 0*l1 + 0*l2 + 100*l3 + 0*l4 + 0*l5


class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""
    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""
    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""
    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out




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
