# %% [markdown]
# # 时间序列预测中的数据分析-＞周期性、相关性、滞后性、趋势性、离群值等特性的分析方法

# %% [markdown]
# ## Import Modure

# %%
import pandas as pd
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %% [markdown]
# ## 1.周期性分析
# - 自相关图(ACF) 
# - 傅里叶变换(FourierTransform)

# %%
def acfDataPlot(data):
    # 计算自相关函数
    acf_result = acf(column_data, nlags=20)
 
    # 设置 seaborn 风格
    sns.set(style='whitegrid')
 
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.stem(acf_result)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
 
    # 保存图像文件
    plt.savefig('acf_plot.png')

# %%
def FourierDataPlot(column_data):
    # 计算傅里叶变换及频谱
 
    # 计算傅里叶变换及频谱
    fft = np.fft.fft(column_data)
    freq = np.fft.fftfreq(len(column_data))
    plt.plot(freq, np.abs(fft))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.savefig('fourier_plot.png')

# %% [markdown]
# ## 2.相关性分析
# - 皮尔逊和斯皮尔曼相关系数
# 
# 用于求解两列数据的相关性系数

# %%
def analyze_correlation(data1, data2):
    """
    分析两列数据的相关性
    参数:
    - data1: 第一列数据，可以是一个一维数组或列表
    - data2: 第二列数据，可以是一个一维数组或列表
    """
    # 将数据转换为NumPy数组
    data1 = np.array(data1)
    data2 = np.array(data2)
 
    # 计算Pearson相关系数
    pearson_corr, _ = stats.pearsonr(data1, data2)
 
    # 计算Spearman相关系数
    spearman_corr, _ = stats.spearmanr(data1, data2)
 
    # 打印相关系数
    print("Pearson相关系数: ", pearson_corr)
    print("Spearman相关系数: ", spearman_corr)
 

# %% [markdown]
# ## 3.滞后性分析
# - 自相关图和偏相关图

# %%
def plot_lag_analysis(data):
    """
    绘制滞后性分析的ACF和PACF图
    参数:
    - data: 时间序列数据，可以是一个一维数组或列表
    """
    # 将数据转换为NumPy数组
    data = np.array(data)
 
    # 绘制ACF图
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(121)
    plot_acf(data, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
 
    # 绘制PACF图
    ax2 = plt.subplot(122)
    plot_pacf(data, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
 
    # 显示图形
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 趋势性分析
# - 线性回归模型检测趋势性

# %%
def calculate_trend(data1,data2):
    # 创建 DataFrame
 
    # 提取自变量和因变量
    x = np.array(data1).reshape(-1, 1)
    y = np.array(data2).reshape(-1, 1)
 
    # 使用线性回归模型拟合数据
    model = LinearRegression()
    model.fit(x, y)
 
    # 提取斜率和截距
    slope = model.coef_[0]
    intercept = model.intercept_
 
 
 
    # 返回趋势性分析结果
    result = {
        'slope': slope,
        'intercept': intercept,
    }
    print(result)

# %% [markdown]
# ## 离群点分析
# - 箱线图检测离群点 
# - Z分数（Z-score）

# %%
import matplotlib.pyplot as plt
 
def detect_outliers(data):
    # 绘制箱线图
    plt.boxplot(data)
 
    # 计算上下须范围
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    lower_threshold = q1 - 1.5 * iqr
 
    # 标记离群点
    outliers = [x for x in data if x > upper_threshold or x < lower_threshold]
    plt.plot(range(1, len(outliers) + 1), outliers, 'ro', label='Outliers')
 
    # 显示图例和标签
    plt.legend()
    plt.xlabel('Data')
    plt.ylabel('Values')
    plt.title('Box Plot with Outliers')
 
    # 显示箱线图
    plt.show()

# %%
def zscore_detect_outlier(data, threshold):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
 
    lower_threshold = q1 - threshold * iqr
    upper_threshold = q3 + threshold * iqr
 
    for i in range(len(data)):
        if data[i] < lower_threshold or data[i] > upper_threshold:
            print(i, data[i])

# %%
## 执行代码

# %%
if __name__ == '__main__':
    # 读取数据框(DataFrame)
    df = pd.read_csv('ETTh1.csv')  # 替换为您的数据文件路径
 
    # 提取某一列数据并转化为时间序列数据
    column_data = df['OT']
    column_data2 = df['MULL']
    # 调用方法
 
    acfDataPlot(column_data)
    FourierDataPlot(column_data)
    plot_boxplot(column_data)
    calculate_trend(column_data,column_data2)
    detect_outliers(column_data)
    zscore_detect_outlier(column_data,0.9)
    analyze_correlation(data1=column_data,data2=column_data2)


