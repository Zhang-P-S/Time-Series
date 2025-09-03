import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from typing import Union, List, Tuple, Optional
from matplotlib import font_manager,rcParams
import matplotlib.font_manager as fm
from matplotlib.pyplot import figure
from matplotlib.dates import DateFormatter, AutoDateLocator
# 设置全局字体
# 添加字体路径
# # Linux
# font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
# fm.fontManager.addfont(font_path)
# font_path = '/usr/share/fonts/truetype/msttcorefonts/SIMSUN.TTC'
# fm.fontManager.addfont(font_path)
# Win
font_path = 'C:/Windows/Fonts/times.ttf'
fm.fontManager.addfont(font_path)
font_path = 'C:/Windows/Fonts/SIMSUN.TTC'
fm.fontManager.addfont(font_path)

rcParams['font.sans-serif'] = ['SimSun']  # 黑体，用于显示中文
rcParams['font.family'] = 'serif'  # 设置字体为衬线体
rcParams['font.serif'] = ['Times New Roman']  # 新罗马字体，用于显示英文

# 解决负号'-'显示为方块的问题
rcParams['axes.unicode_minus'] = False
PLTSIZE = (16 / 2.54, 8 / 2.54)  # size of the plot

def ensure_directory_exists(file_path: str) -> None:
    """确保文件所在目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def plot_image_basic(data):
    if flag == "hold off":
        plt.figure(figsize=PLTSIZE)
    plt.plot(np.arange(len(data)),data, label='Original Data',color=sns.xkcd_rgb['wine red'])
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.show()

def drawtool(x, y, color, label, output_file, num_fig=1, y2=None,color2=None,label2=None):
    # 如果是第一次绘制（hold off），创建新图形

    plt.figure(figsize=PLTSIZE)  # 请确保 PLTSIZE 已定义
    
    x = pd.to_datetime(x)
    
    # 设置日期格式和刻度定位器
    locator = AutoDateLocator(minticks=10, maxticks=15)
    formatter = DateFormatter('%Y-%m')
    
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # 自动旋转 x 轴标签
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 绘制当前数据
    plt.plot(x, y, color=color, linewidth=1, label=label)
    if num_fig == 2 and y2 is not None and color2 is not None and label2 is not None:       
        plt.plot(x, y2, color=color2, linewidth=1, label=label2)
    plt.legend(fontsize=10)
    
    # 如果是最后一次绘制（hold off），保存并显示图形
    ensure_directory_exists(output_file)  # 请确保 ensure_directory_exists 函数已定义
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

class MetricVisualizer:
    """指标可视化类，用于绘制带标记点的训练指标曲线"""
    
    def __init__(self):
        """
        初始化指标可视化器
        
        参数:

        """
        self.plt_size = (16 / 2.54, 8 / 2.54)
        self.font_size = 12
        self.linewidth = 3
        self.markersize = 10
        self.default_colors = [
            (248/255, 204/255, 131/255),  # 橙色
            (144/255, 201/255, 231/255),  # 蓝色
            (171/255, 221/255, 164/255),  # 绿色
            (237/255, 176/255, 211/255),  # 粉色
            (203/255, 153/255, 201/255)   # 紫色
        ]
        self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
    def set_style(self, 
                 font_size: int = 12, 
                 linewidth: int = 3, 
                 markersize: int = 5) -> None:
        """
        设置绘图样式
        
        参数:
        font_size: 字体大小
        linewidth: 线条宽度
        markersize: 标记点大小
        """
        self.font_size = font_size
        self.linewidth = linewidth
        self.markersize = markersize
    def _ensure_directory_exists(self, file_path: str) -> None:
        """确保文件所在目录存在"""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def plot_single_metric(self,
                          metrics: pd,
                          output_file: str,
                          metric_name: str = "MSE",
                          color: Tuple[float, float, float] = None,
                          scale_factor: float = 100,
                          sample_interval: int = 35,
                          title: Optional[str] = None,
                          show_legend: bool = True,
                          isgrid: bool = True) :
        """
        绘制单个指标的曲线图
        
        参数:
        metric_file: 指标数据文件路径
        output_file: 输出图片文件路径
        metric_name: 指标名称
        color: 线条和标记颜色 (RGB元组)
        scale_factor: 数据缩放因子
        sample_interval: 标记点采样间隔
        title: 图表标题
        show_legend: 是否显示图例
        
        返回:
        处理后的数据数组
        """
        # 设置默认颜色
        if color is None:
            color = self.default_colors[0]
        
        # 读取数据
        sampled_data = metrics[::sample_interval]
        
        # 创建图形
        plt.figure(figsize=self.plt_size)
        
        # 绘制完整线条
        epochs = range(len(sampled_data))
        plt.plot(epochs, sampled_data, 
                color=color, 
                linewidth=self.linewidth, 
                label=metric_name,
                alpha=0.8)
        
        # 绘制标记点
        marker_epochs = epochs[::sample_interval]
        marker_values = sampled_data[::sample_interval]
        plt.plot(marker_epochs, marker_values, 
                marker='o', 
                color=color,
                linestyle='',
                markersize=self.markersize,
                alpha=0.8,
                label=f'{metric_name} markers')
        
        # 设置标签和字体
        plt.xlabel('Epoch', fontsize=self.font_size)
        plt.ylabel(metric_name, fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        
        # 设置标题
        if title:
            plt.title(title, fontsize=self.font_size + 2)
        
        # 添加图例
        if show_legend:
            plt.legend(fontsize=self.font_size - 2)
        
        # 添加网格
        if isgrid:
            plt.grid(isgrid, alpha=0.3)
        else:
            plt.grid(False)
        
        # 保存和显示
        plt.tight_layout()
        self._ensure_directory_exists(output_file)
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

    
    def plot_multiple_metrics(self,
                             metrics: List[str],
                             output_file: str,
                             metric_names: Optional[List[str]] = None,
                             colors: Optional[List[Tuple]] = None,
                             scale_factors: Optional[List[float]] = None,
                             sample_intervals: Optional[List[int]] = None,
                             title: Optional[str] = None,
                             ylabel: str = "Metric Value",
                             isgrid: bool = True
                             ) :
        """
        绘制多个指标的对比曲线图
        
        参数:
        metric_files: 指标数据文件路径列表
        output_file: 输出图片文件路径
        metric_names: 指标名称列表
        colors: 颜色列表
        scale_factors: 数据缩放因子列表
        sample_intervals: 标记点采样间隔列表
        title: 图表标题
        ylabel: Y轴标签
        
        返回:
        处理后的数据数组列表
        """
        n_files = len(metrics)
        
        # 设置默认值
        if metric_names is None:
            metric_names = [f'Metric {i+1}' for i in range(n_files)]
        if colors is None:
            colors = self.default_colors[:n_files]
        if scale_factors is None:
            scale_factors = [1] * n_files
        if sample_intervals is None:
            sample_intervals = [1] * n_files
        
        # 验证参数长度
        if len(metric_names) != n_files:
            raise ValueError("metric_names 长度必须与 metric_files 相同")
        if len(colors) != n_files:
            raise ValueError("colors 长度必须与 metric_files 相同")
        if len(scale_factors) != n_files:
            raise ValueError("scale_factors 长度必须与 metric_files 相同")
        if len(sample_intervals) != n_files:
            raise ValueError("sample_intervals 长度必须与 metric_files 相同")
        
        # 创建图形
        plt.figure(figsize=self.plt_size)
        
        all_data = []
        
        # 绘制每个指标
        for i, (data, name, color, scale, interval) in enumerate(zip(
            metrics, metric_names, colors, scale_factors, sample_intervals)):
            
            # 读取和处理数据
            data_result = data
            sampled_data = data_result[::interval]
            all_data.append(sampled_data)
            
            # 绘制完整线条
            epochs = range(len(sampled_data))
            plt.plot(epochs, sampled_data, 
                    color=color, 
                    linewidth=self.linewidth, 
                    label=name,
                    alpha=0.7)
            
            # 绘制标记点
            marker_epochs = epochs[::interval]
            marker_values = sampled_data[::interval]
            plt.plot(marker_epochs, marker_values, 
                    marker=self.marker_styles[i % len(self.marker_styles)],
                    color=color,
                    linestyle='',
                    markersize=self.markersize,
                    alpha=0.8)
        
        # 设置标签和字体
        plt.xlabel('Epoch', fontsize=self.font_size)
        plt.ylabel(ylabel, fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        
        # 设置标题
        if title:
            plt.title(title, fontsize=self.font_size + 2)
        
        # 添加图例
        plt.legend(fontsize=self.font_size - 1)
        
        # 添加网格
        if isgrid:
            plt.grid(isgrid, alpha=0.3)
        else:
            plt.grid(False)
        
        # 保存和显示
        plt.tight_layout()
        self._ensure_directory_exists(output_file)
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

    
    def create_comparison_report(self,
                               metric_files: List[str],
                               metric_names: List[str],
                               output_file: str,
                               scale_factors: Optional[List[float]] = None) -> pd.DataFrame:
        """
        创建指标对比报告
        
        参数:
        metric_files: 指标数据文件路径列表
        metric_names: 指标名称列表
        output_file: 输出报告文件路径
        scale_factors: 数据缩放因子列表
        
        返回:
        包含统计信息的DataFrame
        """
        if scale_factors is None:
            scale_factors = [100] * len(metric_files)
        
        report_data = []
        
        for file, name, scale in zip(metric_files, metric_names, scale_factors):
            try:
                data = self._read_metric_data(file, scale, 1)
                report_data.append({
                    'Metric': name,
                    'Min': np.min(data),
                    'Max': np.max(data),
                    'Mean': np.mean(data),
                    'Std': np.std(data),
                    'Final Value': data[-1],
                    'Best Value': np.min(data) if 'MSE' in name or 'MAE' in name else np.max(data),
                    'Best Epoch': np.argmin(data) if 'MSE' in name or 'MAE' in name else np.argmax(data)
                })
            except Exception as e:
                print(f"处理指标 {name} 时出错: {str(e)}")
        
        report_df = pd.DataFrame(report_data)
        self._ensure_directory_exists(output_file)
        report_df.to_csv(output_file, index=False)
        
        return report_df

# 使用示例
if __name__ == "__main__":

    
    # 创建可视化器实例
    visualizer = MetricVisualizer()
    visualizer.set_style(font_size=14, markersize=12, linewidth=2.5)
    
    # 绘制单个指标
    try:
        mse_data = visualizer.plot_single_metric(
            metric_file='SciSrc2/results/mse_train.csv',
            output_file=os.path.join(config['path']['path_result_saved'], 'Fig5_MSE.pdf'),
            metric_name='MSE',
            sample_interval=35,
            title='Training MSE over Epochs'
        )
        print(f"MSE数据形状: {mse_data.shape}")
    except Exception as e:
        print(f"绘制单个指标时出错: {e}")
    
    # 绘制多个指标对比
    try:
        metrics_data = visualizer.plot_multiple_metrics(
            metric_files=[
                'SciSrc2/results/mse_train.csv',
                'SciSrc2/results/val_mse.csv',
                'SciSrc2/results/train_mae.csv'
            ],
            output_file=os.path.join(config['path']['path_result_saved'], 'Metrics_Comparison.pdf'),
            metric_names=['Train MSE', 'Validation MSE', 'Train MAE'],
            title='Training Metrics Comparison',
            ylabel='Metric Value'
        )
    except Exception as e:
        print(f"绘制多个指标时出错: {e}")
    
    # 创建对比报告
    try:
        report = visualizer.create_comparison_report(
            metric_files=[
                'SciSrc2/results/mse_train.csv',
                'SciSrc2/results/val_mse.csv'
            ],
            metric_names=['Train MSE', 'Validation MSE'],
            output_file=os.path.join(config['path']['path_result_saved'], 'metrics_report.csv')
        )
        print("指标对比报告:")
        print(report)
    except Exception as e:
        print(f"创建报告时出错: {e}")