import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# Chomp1d 是一个自定义 pytorch 模块，用于裁剪输入张量的最后 chomp_size 个元素。
# 在TCN中，为了确保输出的时间步长与输入保持一致，这个模块非常重要。
class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 输入张量 x 的最后 chomp_size 个时间步被移除
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        # n_inputs: 输入特征的数量。
        # n_outputs: 输出特征的数量。
        # kernel_size: 卷积核的大小。
        # stride: 卷积的步幅（默认为1）。
        # dilation: 膨胀卷积的膨胀系数。使用膨胀卷积能够增加接收场，提高模型捕捉长时依赖的能力。
        # padding: 填充大小，确保输出的时间步与输入一致。
        # dropout: 为每个卷积层添加的 dropout 概率。
        super(TemporalBlock, self).__init__()
        # 使用了连续的卷积层、Chomp1d、ReLU激活和Dropout层。weight_norm 用于规范化卷积层的权重
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 通过下采样 (downsample) 将输入与输出的维度对齐（如果不一致）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        # num_inputs: 输入数据特征的数量。
        # num_channels: 每一层的输出通道数。一组通道数可以指定多个 TemporalBlock。
        # kernel_size: 卷积核大小，默认为2。
        # dropout: dropout概率，默认为0.2。
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)