# For Files
import pandas as pd
import sys
import csv
import os






def save_data(data, file):
    ''' Save predictions to specified file '''
    print('Saving data to {}'.format(file))
    os.makedirs(os.path.dirname(file), exist_ok=True)
    df = pd.DataFrame(data)
    # 保存为CSV文件
    df.to_csv(file, index=False, header=False)