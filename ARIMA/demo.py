import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from arima_model.arima import ARIMA_Model

n_pred = 10  # Number of steps to predict
n_train = 100  # Number of training samples


df = pd.read_csv('data/ETTh1.csv').iloc[:,0:2]
print(df.shape)


# df.plot(title='Original Time Series', figsize=(10, 4))
# plt.grid(True)
# plt.show()

# 一阶差分（差分后用于模型建模）
df_diff = df['HUFL'].diff().dropna()
# df_diff.plot(title='Differenced Time Series')
# plt.grid(True)
# plt.show()

model = ARIMA_Model(order=(1, 1, 1))
model_fit = model.fit(df['HUFL'])
model = ARIMA(df.iloc[:n_train,1], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
preds = model_fit.forecast(steps=n_pred)

plt.figure(figsize=(10, 4))
plt.plot(df.index[n_train:n_train+n_pred], df['HUFL'][n_train:n_train+n_pred], label='Training Data', color='blue')
plt.plot(df.index[n_train:n_train+n_pred], preds, label='Pred Data', color='red')
plt.grid(True)
plt.show()