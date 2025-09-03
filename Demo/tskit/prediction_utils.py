import torch
import math
import pandas as pd
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from .metrics_utils import *
from .plot_utils import *
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import seaborn as sns
Ctable = {
    'color_gt':         sns.xkcd_rgb['light rose'],
    'color_true':       sns.xkcd_rgb['tea'],
    'color_pred':       sns.xkcd_rgb['violet'],
    'color_error':      sns.xkcd_rgb['bright red'],
    'color_anscore':    sns.xkcd_rgb['burnt orange'],
    'color_anomal':     sns.xkcd_rgb['sunny yellow'],
    
}
@dataclass
class Predictor:
    model: torch.nn.Module
    device: str
    window_len: int
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        'test_mse': [], 'test_r2': [], 'test_mape': [], 
        'test_smape': [], 'test_mae': [], 'test_rmse': []
    })
    preds: List = field(default_factory=list)
    errors: List = field(default_factory=list)

    def prediction_epoch(self, test_loader):
        self.model.eval()
        # 获取第一批数据
        for batch in test_loader:
            first_batch = batch
            break
        self.preds_init = first_batch[0][0].data  # 初始预测张量
        print(self.preds_init.shape)
        self.errors_init = torch.zeros_like(self.preds_init)
        with torch.no_grad():  # 关闭梯度计算
            with tqdm(test_loader, unit='batch', desc="Testing") as pbar:  # 使用 tqdm 包装 test_loader
                for x, y in pbar:
                    x = x.to(self.device)  # 将输入数据移动到设备（如 GPU）
                    y = y.to(self.device)  # 将目标数据移动到设备（如 GPU）
                    pred = self.model(x)  # 模型预测
                    self.preds.append(pred[:, 0, :].cpu())  # 存储预测结果（转回 CPU）

                    # 计算重构误差
                    error = torch.abs(pred[:, 0, :] - y[:, 0, :])  # 只取第一个时间步的误差
                    self.errors.append(error.cpu())  # 存储误差（转回 CPU）

                    self.metrics['test_mse'].append(mse_loss(pred, y).detach().cpu().item())
                    self.metrics['test_r2'].append(r2_loss(pred, y).detach().cpu().item())
                    self.metrics['test_mape'].append(mape_loss(pred, y).detach().cpu().item())
                    self.metrics['test_smape'].append(smape_loss(pred, y).detach().cpu().item())
                    self.metrics['test_mae'].append(mae_loss(pred, y).detach().cpu().item())
                    self.metrics['test_rmse'].append(rmse_loss(pred, y).detach().cpu().item())

                    # 更新进度条描述
                    pbar.set_postfix({
                        'test_mse': mse_loss(pred, y).detach().cpu().item(),  # 显示 MSE 损失
                        'r2_score': r2_loss(pred, y).detach().cpu().item()  # 显示 R2 分数
                    })
        self.test_prediction = torch.cat(self.preds, dim=0) 
        self.test_prediction = torch.cat([self.preds_init, self.test_prediction], dim=0)       

        self.test_error = torch.cat(self.errors, dim=0)  
        self.test_error = torch.cat([self.errors_init, self.test_error], dim=0) 
        print(self.test_prediction.shape)
        print(self.test_error.shape)

    def plot_prediction(self,timestamp,output_file):
        if len(timestamp) == len(self.test_prediction[:,0]):
            drawtool(timestamp,self.test_prediction[:,0],Ctable['color_pred'],'Prediction',output_file)
        else:
            print("* Warning: Length of timestamp does not match length of predictions. Adjusting to fit.")
            drawtool(timestamp[:len(self.test_prediction[:,0])],self.test_prediction[:,0],Ctable['color_pred'],'Prediction',output_file)
    def plot_error(self,timestamp,output_file):
        if len(timestamp) == len(self.test_error[:,0]):
            drawtool(timestamp,self.test_prtest_errorediction[:,0],Ctable['color_error'],'Prediction',output_file)
        else:
            print("* Warning: Length of timestamp does not match length of errors. Adjusting to fit.")
            drawtool(timestamp[:len(self.test_error[:,0])],self.test_error[:,0],Ctable['color_error'],'Error',output_file)

    def pred_vs_gt(self,timestamp,ground_truth, feature_num, output_file):
        # print(len(ground_truth.iloc[:,feature_num]))
        # print(len(self.test_prediction[:,feature_num]))
        # print(len(timestamp))
        if len(timestamp) == len(self.test_prediction[:,feature_num]):
            drawtool(timestamp,ground_truth.iloc[:,feature_num],Ctable['color_gt'],'Prediction',output_file,2,self.test_prediction[:,feature_num],Ctable['color_pred'],"Ground Truth")
            
        else:
            print("* Warning: Length of timestamp does not match length of predictions. Adjusting to fit.")
            drawtool(timestamp[:len(self.test_prediction[:,feature_num])],ground_truth.iloc[:len(self.test_prediction[:,feature_num]),feature_num],Ctable['color_gt'],'Prediction',output_file,2,self.test_prediction[:,feature_num],Ctable['color_pred'],"Ground Truth")

    def print_metrics(self,ground_truth):
        print(type(self.test_prediction))  # 可能是 numpy array
        print(type(ground_truth))          # 可能是 pandas Series/DataFrame 或 numpy array
        # 直接使用values属性，它应该已经是float64类型
        ground_truth_array = ground_truth.values.astype(np.float32)
        ground_truth_torch = torch.FloatTensor(ground_truth_array)
        test_mse = mse_loss(self.test_prediction, ground_truth_torch).item()
        test_r2 = r2_loss(self.test_prediction, ground_truth_torch).item()
        test_mape = mape_loss(self.test_prediction, ground_truth_torch).item()
        test_smape = smape_loss(self.test_prediction, ground_truth_torch).item()
        test_mae = mae_loss(self.test_prediction, ground_truth_torch).item()
        test_rmse = rmse_loss(self.test_prediction, ground_truth_torch).item()
        print("┌────────────────────────────────────┐")
        print("│           MODEL METRICS            │")
        print("├────────────────────────────────────┤")
        print(f"│ • MSE:      {test_mse:12.4f}           │")
        print(f"│ • R²:       {test_r2:12.4f}           │")
        print(f"│ • MAPE:     {test_mape:12.4f}%          │")
        print(f"│ • SMAPE:    {test_smape:12.4f}%          │")
        print(f"│ • MAE:      {test_mae:12.4f}           │")
        print(f"│ • RMSE:     {test_rmse:12.4f}           │")
        print("└────────────────────────────────────┘")