import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
PLTSIZE = (16 / 2.54, 8 / 2.54)  # size of the plot

def plot_image_basic(data):
    plt.figure(figsize=PLTSIZE)
    plt.plot(np.arange(len(data)),data, label='Original Data',color=sns.xkcd_rgb['wine red'])
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.show()