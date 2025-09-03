'''
实现数据导入功能
输入：json文件，包含文件位置
输出：数据集文件输出
'''

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
import pickle

from tskit import *

# ======================
# 实例化
# ======================
parser = argparse.ArgumentParser('数据导入')
parser.add_argument('--json_path', type=str, default=r'D:\AProjects\DrStudy\AI\A1_TimeSeries\Demo\DataInput.json')
args = parser.parse_args(args=[])

# 读取json信息
with open(args.json_path, encoding='utf-8') as f:
    config = json.load(f)

# 直接使用JSON中的配置
input_data_path = config["input_data_path"]
output_data_path = config["output_data_path"]
selected_features = config["selected_features"]
prediction_len = config["prediction_len"]
window_len = config["window_len"]
split_ratio = config["split_ratio"]
batch_size = config["batch_size"]
# ======================
# 数据加载
# ======================
raw_data = pd.read_csv(
    input_data_path,
    header=1,  # 跳过第一行作为标题
)
raw_data.rename(columns={raw_data.columns[0]: 'timestamp'}, inplace=True)

contains_nan = raw_data.isna().any().any()  # 检查整个 DataFrame 是否包含 NaN 值
if contains_nan:
    print("数据存在nan值")  # 如果有 NaN 值，返回 True，否则返回 False
    nan_columns = raw_data.isna().any()
    print(nan_columns)
else:
    print("数据正常！")  # 如果有 NaN 值，返回 True，否则返回 False

# ======================
# 特征选择
# ======================
features_len = len(raw_data.iloc[1, :])-1 # Subtract 'timestamp'
print("* Input data has {} features".format(features_len)) 

# for i in range(1,len(raw_data.iloc[1, :])):
    # plot_image_basic(raw_data.iloc[:, i])
selected_data = raw_data if selected_features == 'all' else raw_data.iloc[:, [0] + selected_features] 

features_len = len(selected_data.iloc[1, :])-1 # Subtract 'timestamp'
data_len = len(selected_data.iloc[:, 1:])
print("* Selected data has {} features".format(features_len)) 
print("* Selected data has {} nums".format(data_len))

selected_timestamp = selected_data.iloc[:,0]
selected_values = selected_data.iloc[:,1:].to_numpy(dtype=float)
# ======================
# 数据载入
# ======================
device = get_device()         
print("* Using device: {}".format(device))
print("* 创建数据加载器")
selected_values_torch = torch.FloatTensor(np.array(selected_values))
print("* selected_values_torch.shape = {}".format(selected_values_torch.shape))

train_data_raw = selected_values_torch[:int(split_ratio[0]*data_len)]
valid_data_raw = selected_values_torch[int(split_ratio[0]*data_len):int((split_ratio[0] + split_ratio[1])*data_len)]
test_data_raw  = selected_values_torch[int((split_ratio[0] + split_ratio[1])*data_len):]
train_timestamp = selected_timestamp.iloc[:int((split_ratio[0] + split_ratio[1])*data_len)]
test_timestamp =  selected_timestamp.iloc[int((split_ratio[0] + split_ratio[1])*data_len):]
print("* train_data_raw.shape = {}".format(train_data_raw.shape))
print("* valid_data_raw.shape = {}".format(valid_data_raw.shape))
print("* test_data_raw.shape  = {}".format(test_data_raw.shape))
print(test_timestamp.shape)
# 定义训练器的的输入
train_data,train_label = create_inout_sequences(train_data_raw, window_len, prediction_len, config)
valid_data,valid_label = create_inout_sequences(valid_data_raw, window_len, prediction_len, config)
test_data,test_label = create_inout_sequences(test_data_raw, window_len, prediction_len, config)

# 创建数据集
train_dataset = TimeSeriesDataset(train_data,train_label,'train')
valid_dataset = TimeSeriesDataset(valid_data,valid_label,'dev')
test_dataset = TimeSeriesDataset(test_data,test_label,'test')

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

print("通过滑动窗口共有训练集数据：", len(train_data), "转化为批次数据:", len(train_loader))
print("通过滑动窗口共有验证集数据：", len(valid_data), "转化为批次数据:", len(valid_loader))
print("通过滑动窗口共有测试集数据：", len(test_data), "转化为批次数据:", len(test_loader))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# ======================
# 数据保存
# ======================

selected_timestamp.to_csv(output_data_path+"\\selected_timestamp.csv", index=False)
train_timestamp.to_csv(output_data_path+"\\train_timestamp.csv", index=False)
test_timestamp.to_csv(output_data_path+"\\test_timestamp.csv", index=False)
pd.DataFrame(test_data_raw.cpu().numpy()).to_csv(output_data_path+"\\test_data.csv", index=False)
# 保存 train_loader 和 valid_loader
with open(output_data_path+'\\train_loader.pkl', 'wb') as f:
    pickle.dump(train_loader, f)
with open(output_data_path+'\\valid_loader.pkl', 'wb') as f:
    pickle.dump(valid_loader, f)
with open(output_data_path+'\\test_loader.pkl', 'wb') as f:
    pickle.dump(test_loader, f)


