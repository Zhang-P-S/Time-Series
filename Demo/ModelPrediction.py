import argparse
import json
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from tsmodel import *
from tsmodel import iTransformer
from tskit import *
import numpy as np
import pickle
import os
import pandas as pd
# ======================
# 实例化
# ======================
parser = argparse.ArgumentParser('模型预测')
parser.add_argument('--json_path', type=str, default=r'D:\AProjects\DrStudy\AI\A1_TimeSeries\Demo\ModelPrediction.json')
args = parser.parse_args(args=[])

# 读取json信息
with open(args.json_path, encoding='utf-8') as f:
    config = json.load(f)

# 直接使用JSON中的配置
test_data_path = config["test_data_path"]
model_path = config["model_path"]
output_data_path = config["output_data_path"]
num_features = config["num_features"]
prediction_len = config["prediction_len"]
window_len = config["window_len"]
batch_size = config["batch_size"]
model_type = config["model_type"]
input_data_path = config["input_data_path"]
device = get_device()   
# 模型初始化
match model_type:
    case 'RINDCCNET':
        model = RINDCCNet(
            num_features, 
            num_features, 
            prediction_len, 
            num_channels = [32,64,128], 
            kernel_size = 9,
            use_reversible_instance_norm = True,
            reversible_instance_norm_affine = True
        ).float()
    case 'iTransformer':
        model = iTransformer(
            num_variates=num_features,
            lookback_len=window_len,
            dim=128,
            depth=6,
            heads=8,
            dim_head=8,
            pred_length=prediction_len,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True,
            reversible_instance_norm_affine = True
        ).float()
    case 'Crossformer':
        model = iTransformer(
            num_variates=num_features,
            lookback_len=window_len,
            dim=128,
            depth=6,
            heads=8,
            dim_head=8,
            pred_length=prediction_len,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True
        ).float()
    case _:
        raise ValueError(f"Unsupported model type: {model_type}")

model = model.to(device)
# 显卡初始化
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
# 修改这行代码
ckpt = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt)

# 查看参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"* 选取的模型为：{model_type}")
print(f"* 总参数量: {total_params}")
print(f"* 可训练参数量: {trainable_params}")

# ======================
# 导入数据
# ======================
# 加载 test_loader
with open(test_data_path, 'rb') as f:
    test_loader = pickle.load(f)

# ======================
# 开始预测
# ======================
predictor = Predictor(model,device,window_len)
predictor.prediction_epoch(test_loader)

# ======================
# 预测绘图
# ======================
timestamps = pd.read_csv(input_data_path+"\\test_timestamp.csv", parse_dates=[0])['timestamp']
test_data_raw = pd.read_csv(input_data_path+"\\test_data.csv", parse_dates=[0])
print(test_data_raw.head())
# predictor.plot_prediction(timestamps,os.path.join(output_data_path, 'prediction\\prediction.pdf'))
# predictor.plot_error(timestamps,os.path.join(output_data_path, 'prediction\\error.pdf'))
# predictor.pred_vs_gt(timestamps,test_data_raw, 1,os.path.join(output_data_path, 'prediction\\pred_vs_gt.pdf'))
predictor.print_metrics(test_data_raw)
