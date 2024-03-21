from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_permute(x, index):  # generate a new tensor
    y = x.reshape(x.shape[0], -1).detach().clone()  # flatten all feature, this function will only be used in the
    # context of with no grad
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = y[perm_index, i]
    y = y.reshape(*x.size())  # reshape to original size
    return y


def my_permute_new(x, index):
    y = deepcopy(x)
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = x[perm_index, i]
    return y


def my_freeze(x, index):  # in place modification
    ori_size = x.size()
    x = x.reshape(x.shape[0], -1)
    x[:, index] = 0
    x = x.reshape(*ori_size)
    return x


def my_freeze_new(x, index):  # in place modification
    # y = deepcopy(x)
    # y = x
    y = x.clone()

    # y[:, index] = 0
    tmp_mean = x[:, index].mean(dim=0)
    y[:, index] = tmp_mean

    return y


class ImageNet(nn.Module):
    def __init__(self, num_class=8):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveMaxPool2d((7, 14))

        self.fc1 = nn.Linear(128 * 7 * 14, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_class)
        self.fc44 = nn.Linear(512*2, 512)
        self.fc55 = nn.Linear(512, num_class)
        self.conv1_fit = ConvReg(32)
        self.conv2_fit = ConvReg(64)
        self.conv3_fit = ConvReg(128)
        self.fc1_fit = FCReg(1024)
        self.fc2_fit = FCReg(128)


    def fusion(self, x):
        x_128 = self.fc44(x)
        x = self.fc55(x_128)
        return x, x_128

    def get_feature_dim(self, place=None):
        feature_dim_list = [3 * 256 * 512, 32 * 128 * 256, 64 * 64 * 128, 128 * 7 * 14, 1024, 128, 8]
        return feature_dim_list[place] if place else feature_dim_list

    def forward(self, x, change_type=None, place=None, index=None):

        x = F.relu(self.conv1(x))
        x_conv1 = self.conv1_fit(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x_conv2 = self.conv2_fit(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x_conv3 = self.conv3_fit(x)
        x = self.pool2(x)
        x = x.view(-1, 128 * 7 * 14)
        x = F.relu(self.fc1(x))
        x_fc1 = self.fc1_fit(x)
        x_128 = F.relu(self.fc2(x))
        x_fc2 = self.fc2_fit(x_128)
        x = self.fc3(x_128)
        return x, x_128, [x_conv1, x_conv2, x_conv3, x_fc1, x_fc2]


class AudioNet(nn.Module):
    def __init__(self, num_class=8):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveMaxPool2d((7, 14))

        self.fc1 = nn.Linear(128 * 7 * 14, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_class)


    def get_feature_dim(self, place=None):
        feature_dim_list = [1 * 40 * 256, 32 * 128 * 126, 64 * 8 * 62, 128 * 7 * 14, 1024, 128, 8]
        return feature_dim_list[place] if place else feature_dim_list

    def forward(self, x, change_type=None, place=None, index=None):
        x = F.relu(self.conv1(x))
        x_conv1 = x
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x_conv2 = x
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x_conv3 = x
        x = self.pool2(x)
        x = x.view(-1, 128 * 7 * 14)
        x = F.relu(self.fc1(x))
        x_fc1 = x
        x_128 = F.relu(self.fc2(x))
        x_fc2 = x_128
        x = self.fc3(x_128)
        return x, x_128, [x_conv1, x_conv2, x_conv3, x_fc1, x_fc2]


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(s_C, s_C, kernel_size=1)
        self.bn = nn.BatchNorm2d(s_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

class FCReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C, use_relu=True):
        super(FCReg, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(s_C, s_C)
        self.bn = nn.BatchNorm2d(s_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        # if self.use_relu:
        #     return self.relu(self.bn(x))
        # else:
        #     return self.bn(x)
        return x