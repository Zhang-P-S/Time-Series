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

# ======================
# 实例化
# ======================
parser = argparse.ArgumentParser('模型建立')
parser.add_argument('--json_path', type=str, default=r'D:\AProjects\DrStudy\AI\A1_TimeSeries\Demo\ModelSet.json')
args = parser.parse_args(args=[])

# 读取json信息
with open(args.json_path, encoding='utf-8') as f:
    config = json.load(f)

# 直接使用JSON中的配置
train_data_path = config["train_data_path"]
valid_data_path = config["valid_data_path"]
saved_model_path = config["saved_model_path"]
output_data_path = config["output_data_path"]
num_features = config["num_features"]
prediction_len = config["prediction_len"]
window_len = config["window_len"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
model_type = config["model_type"]

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

# 多卡初始化
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

# 查看参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"* 总参数量: {total_params}")
print(f"* 可训练参数量: {trainable_params}")

# ======================
# 导入数据
# ======================
# 加载 train_loader 和 valid_loader
with open(train_data_path, 'rb') as f:
    train_loader = pickle.load(f)

with open(valid_data_path, 'rb') as f:
    valid_loader = pickle.load(f)

loss_record = {'train': [], 'dev': []}  # for recording training loss
r2_loss_record = {'train': [], 'dev': []}
mape_loss_record = {'train': [], 'dev': []}
smape_loss_record = {'train': [], 'dev': []}
lr_history = []
epoch_bestmodel = 0

# 定义配置
config = TrainingConfig(
    num_epochs=num_epochs,
    lr=learning_rate,
    early_stop=5,
    device=device,
)

# 初始化训练器
trainer = Trainer(model, config)

# 开始训练
trainer.train(train_loader, valid_loader, saved_model_path)


save_data(trainer.metrics['train_mse'], os.path.join(output_data_path, 'metrics\\mse_train.csv'))

