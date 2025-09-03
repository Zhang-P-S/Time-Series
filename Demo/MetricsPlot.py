'''
实现指标可视化功能
输入：json文件，包含文件位置
输出：指标可视化图输出
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
import os
from tskit import *

# ======================
# 实例化
# ======================
parser = argparse.ArgumentParser('数据导入')
parser.add_argument('--json_path', type=str, default=r'D:\AProjects\DrStudy\AI\A1_TimeSeries\Demo\MetricsPlot.json')
args = parser.parse_args(args=[])

# 读取json信息
with open(args.json_path, encoding='utf-8') as f:
    config = json.load(f)

# 直接使用JSON中的配置
input_data_path = config["input_data_path"]
output_data_path = config["output_data_path"]
# ======================
# 数据加载
# ======================
raw_data = pd.read_csv(
    input_data_path,
    header=0,  # 跳过第一行作为标题
)
raw_data.rename(columns={raw_data.columns[0]: 'epoch'}, inplace=True)
print(raw_data.head())
# ======================
# 指标绘图
# ======================
# 创建可视化器实例
metric_visualizer = MetricVisualizer()
metric_visualizer.set_style(font_size=14, markersize=8, linewidth=2.5)

# 绘制单个指标
try:
    metric_visualizer.plot_single_metric(
        metrics = raw_data['train_mse'],
        output_file = os.path.join(output_data_path, 'metrics\\Fig5_MSE.pdf'),
        metric_name = 'MSE',
        sample_interval=4,
        title='Training MSE over Epochs',
        isgrid = False,
    )
except Exception as e:
    print(f"绘制单个指标时出错: {e}")

# metric_visualizer.plot_multiple_metrics(
#     metrics=[
#         raw_data['train_mae'],
#         raw_data['train_r2']
#     ],
#     output_file = os.path.join(output_data_path, 'metrics\\Metrics_Comparison.pdf'),
#     metric_names=['Train MAE',  'Train R2'],
#     sample_intervals = [1,1],
#     title='Training Metrics Comparison',
#     ylabel='Metric Value'
# )

