import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from beartype import beartype
from beartype.typing import Optional, Union, Tuple

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from .revin import RevIN

class Chomp1d(nn.Module):
    # Chomp1d 类用于处理卷积操作后多出来的边缘部分，确保输出的长度与输入长度匹配
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),  # 1x1 卷积
            nn.GELU(),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1),  # 1x1 卷积
            nn.Sigmoid()  # 生成权重
        )

    def forward(self, x):
        # x: (B, M, T)
        attn = self.avg_pool(x)  # (B, M, 1)
        attn = self.fc(attn)  # (B, M, 1)
        return x * attn  # (B, M, T)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 通道注意力机制
        self.channel_attention = ChannelAttention(n_outputs)

        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 第一个卷积层
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 通道注意力机制
        out = self.channel_attention(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class RINDCCNet(nn.Module):
    def __init__(self, 
    num_inputs, 
    outputs, 
    pre_len, 
    num_channels, 
    kernel_size=2, 
    dropout=0.2,        
    use_reversible_instance_norm = False,
    reversible_instance_norm_affine = False
    ):
        super(RINDCCNet, self).__init__()
        layers = []
        self.pre_len = pre_len
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)
        # self.revinlayer = RevIN(num_features=num_inputs, affine=True, subtract_last=False)
        self.reversible_instance_norm = RevIN(num_inputs, affine = reversible_instance_norm_affine) if use_reversible_instance_norm else None
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        # 归一化
        x = rearrange(x, 'b n v -> b v n')  # (B, T, M) -> (B, M, T)
        # if exists(self.reversible_instance_norm):
        #     x, reverse_fn = self.reversible_instance_norm(x)
        x, reverse_fn = self.reversible_instance_norm(x)
        # 通过 TemporalBlock 堆叠
        x = self.network(x)  # (B, M, T)

        # 线性层
        x = x.permute(0, 2, 1)  # (B, T, M)
        x = self.linear(x)  # (B, T, outputs)

        # 反归一化
        x = x.permute(0, 2, 1)  # (B, M, T)
        # if exists(self.reversible_instance_norm):
        #     x = reverse_fn(x)
        x = reverse_fn(x)
        x = x.permute(0, 2, 1)  # (B, M, T)

        return x[:, -self.pre_len:, :]  # 返回预测部分

    def cal_loss(self, pred, label):
        mse = nn.MSELoss(reduction='mean')(pred, label)
        return mse