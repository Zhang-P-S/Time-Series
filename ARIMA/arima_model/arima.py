import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class ARIMA_Model:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None

    def fit(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()
        return self.model_fit   
    def forecast(self, steps):
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")
        return self.model_fit.forecast(steps=steps)
