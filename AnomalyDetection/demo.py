from datetime import datetime
import pandas as pd
# 数据读取教程
# 这里将数据集中 timestamp（UNIX时间戳）转换为标准的 datetime 格式，存入新的列 ds
# AnomalyDetection/Data/taxi/trip_data_1.csv
dataset = pd.read_csv('AnomalyDetection/Data/taxi/trip_data_1.csv')
labels = pd.read_csv('AnomalyDetection/Data/taxi/trip_fare_1.csv')
dataset['ds'] = pd.Series([datetime.fromtimestamp(x) 
    for x in dataset['timestamp']])
dataset = dataset.drop('timestamp', axis=1)
dataset['unique_id'] = 'NYT'
# 将 value 列重命名为 y，可能是为了配合某些时间序列模型的要求，比如 Prophet 模型中的 y 表示目标变量
dataset = dataset.rename(columns={'value': 'y'})
# 标注异常数据
is_anomaly = []
for i, r in labels.iterrows():
    dt_start = datetime.fromtimestamp(r.start)
    dt_end = datetime.fromtimestamp(r.end)
    anomaly_in_period = [dt_start <= x <= dt_end 
    for x in dataset['ds']]
    is_anomaly.append(anomaly_in_period)
dataset['is_anomaly']=pd.DataFrame(is_anomaly).any(axis=0).astype(int)
dataset['ds'] = pd.to_datetime(dataset['ds'])