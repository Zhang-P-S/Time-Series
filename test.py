import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data\ETTh1.csv')

df['label'] = 'normal'
df.loc[2,'label'] = 'anormaly'
print(df.head())
df= df.set_index('date')
print(df.iloc[:,0].head())
df.reset_index(inplace=True)
timestamp = pd.to_datetime(df['date'])
data = df.loc[:, ['HUFL']]
plt.figure(figsize=(10, 4))
plt.plot(timestamp, data,linewidth=0.5, color='black', label='HUFL')
plt.xlabel('Time')
plt.ylabel('HUFL')
plt.grid(True)


if not os.path.exists('fig'):
    os.makedirs('fig')
plt.savefig('fig/ETTh1.pdf', dpi=300, bbox_inches='tight')
plt.show()
