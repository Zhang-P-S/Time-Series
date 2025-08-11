import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: 特征或通道的数量
        :param eps: 数值稳定性参数，防止除零错误
        :param affine: 如果为True，RevIN有可学习的仿射参数
        :param subtract_last: 如果为True，减去最后一个时间步的值
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # 初始化RevIN参数: (M,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

import torch

# 假设输入数据形状为(batch_size, seq_length, num_features)
x = torch.randn(32, 100, 64)  # batch_size=32, seq_length=100, num_features=64

# 创建RevIN模块
revin = RevIN(num_features=64, affine=True, subtract_last=False)

# 归一化
x_norm = revin(x, mode='norm')

# 反归一化
x_denorm = revin(x_norm, mode='denorm')

# 打印结果
print(f"输入数据形状: {x.shape}")
print(f"归一化后数据形状: {x_norm.shape}")
print(f"反归一化后数据形状: {x_denorm.shape}")
