# ======================
# 导入库
# ======================
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn.utils import weight_norm
import torch.optim.lr_scheduler as lr_scheduler


# 数据集划分
def train_valid_split(data_set,split_rate,seed):
    train_data = data_set[:int(len(data_set) * split_rate[0])]
    valid_data = data_set[int(len(data_set) * split_rate[0]):int(len(data_set) * (split_rate[0] + split_rate[1]))]
    test_data = data_set[int(len(data_set) * (split_rate[0] + split_rate[1])):]

    # print(f'size of test_data is {test_data.shape}')
    # print(f'size of train_data is {train_data.shape}')
    # print(f'size of valid_data is {valid_data.shape}')
    return train_data,valid_data,test_data
# 构造数据集
class TimeSeriesDataset(Dataset):
    # input:数据路径；模式（默认为train）；特征选择
    def __init__(self,
                 data,
                 label,
                 mode='train'):
        self.mode = mode
        # 检查数据类型
        # print(data.dtype)  # 查看数据类型
        if mode == 'test':
            self.data = data
            self.label = label
        else:
            self.data = data
            self.label = label
        self.dim = 1# 选取的是第二维度
        print('* Finished reading the {} set of Orbit Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.label[index]
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def create_inout_sequences(input_data, tw, preLen, config):
    '''
    创建时间序列数据专用的数据分割器，返回输入和输出均为 torch.Tensor。
    
    参数：
    - input_data: 输入的时间序列数据，torch.Tensor。
    - tw: 滑动窗口大小，决定了每个输入序列的长度。
    - preLen: 预测长度，即模型需要预测的未来数据点的数量。
    - config: 配置对象，包含有关特征选择的信息（暂未使用，可扩展）。
    
    返回：
    - inout_seq_data: 输入序列数据，torch.Tensor，形状为 (num_samples, tw, features)。
    - inout_seq_label: 输出序列数据，torch.Tensor，形状为 (num_samples, preLen, features)。
    '''
    if not isinstance(input_data, torch.Tensor):
        raise ValueError("input_data 必须是 torch.Tensor 类型")

    # 初始化列表用于存储序列对
    inout_seq_data = []
    inout_seq_label = []

    # 数据长度
    L = input_data.size(0)

    # 遍历数据，生成序列对
    for i in range(L - tw):
        if (i + tw + preLen) <= L:  # 确保不会超出数据边界
            seq_data = input_data[i:i + tw]          # 输入序列，长度为 tw
            seq_label = input_data[i + tw:i + tw + preLen]  # 输出序列，长度为 preLen
            inout_seq_data.append(seq_data)
            inout_seq_label.append(seq_label)

    # 将列表转为 torch.Tensor
    inout_seq_data = torch.stack(inout_seq_data)  # 形状：(num_samples, tw, features)
    inout_seq_label = torch.stack(inout_seq_label)  # 形状：(num_samples, preLen, features)

    return inout_seq_data, inout_seq_label
