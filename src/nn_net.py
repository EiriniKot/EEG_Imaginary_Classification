import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


def conv2d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int),
    )
    return output_size


class Net(nn.Module):
    def __init__(self, c, d, h, outputs):
        filter_1 = 256
        filter_2 = 128
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(c, filter_1, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(filter_1)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(filter_1, filter_2, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(filter_2)

        conv_out_size = conv2d_output_size(input_size=[c, d, h], out_channels=filter_1, padding=0, kernel_size=3, stride=1)
        conv_out_size = conv2d_output_size(conv_out_size, out_channels=filter_1, padding=0, kernel_size=3, stride=1)
        # conv_out_size = conv2d_output_size(conv_out_size, out_channels=filter_2, padding=0, kernel_size=3, stride=1)
        x = math.prod(conv_out_size)
        self.head = nn.Linear(x, outputs)

    def forward(self, x):
        x = torch.sigmoid(x).float()
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.max2(x))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.softmax(self.head(x), dim=1)
        return x
