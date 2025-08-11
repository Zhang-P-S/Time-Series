import torch
import matplotlib.pyplot as plt
from iTransformer import iTransformer

# ---------- 模型构建 ----------
model = iTransformer(
    num_variates=137,
    lookback_len=96,
    dim=256,
    depth=6,
    heads=8,
    dim_head=64,
    pred_length=1,
    num_tokens_per_variate=1,
    use_reversible_instance_norm=True
)

# ---------- 输入数据（模拟时间序列） ----------
# shape: (batch, lookback_len, variates)
time_series = torch.randn(2, 96, 137)

# ---------- 运行模型 ----------
model.eval()
with torch.no_grad():
    preds = model(time_series)
print(preds)
print(type(preds))
print(preds.keys())

# ---------- 可视化 ----------
def plot_predictions(time_series, preds, batch_idx=0, var_idx=0):
    history = time_series[batch_idx, :, var_idx].cpu().numpy()
    prediction = preds[batch_idx, :, var_idx].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(history)), history, label='History', color='blue')
    plt.plot(range(len(history), len(history) + len(prediction)), prediction, label='Prediction', color='red')
    plt.axvline(x=len(history)-1, color='gray', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel(f"Variable {var_idx}")
    plt.title("iTransformer Single-step Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------- 调用函数 ----------
# plot_predictions(time_series, preds, batch_idx=0, var_idx=0)
