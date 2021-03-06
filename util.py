import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.seasonal import seasonal_decompose


class Util:
    @staticmethod
    def create_sequences(data, seq_length, y_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length - y_length):
            x = data[i : (i + seq_length)]
            y = [i[0] for i in data][(i + seq_length) : (i + seq_length + y_length)]
            xs.append(x)
            ys.append(y)
        return (
            torch.from_numpy(np.array(xs)).float(),
            torch.from_numpy(np.array(ys)).float(),
        )

    @staticmethod
    def fill_na(df):
        alldate = pd.DataFrame(
            pd.date_range(start=df.index[0], end=df.index[-1])
        ).set_index(0)
        df = pd.merge(
            alldate, df, left_index=True, right_index=True, how="left"
        ).fillna(method="ffill")
        return df

    @staticmethod
    def unscale(scaler, y, n_f, col_num=0):  # returns a flat list of unscaled y.
        y_t = [np.repeat(t, n_f) for t in y]
        inv_y = scaler.inverse_transform(y_t)
        true_y = [i[col_num] for i in inv_y]
        return true_y

    @staticmethod
    def MAPE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class TimeDataSet(object):
    def __init__(self, price):
        self.price = price

    def make_dataset(self, period, list_of_roll):

        price_decomp = seasonal_decompose(
            self.price, model="additive", period=period, extrapolate_trend="freq"
        )
        price = price_decomp.resid + price_decomp.trend  # extract seasonality

        l_df = [self.price]
        for i in list_of_roll:
            l_df.append(self.price.rolling(i).mean())
        price = pd.concat(l_df, axis=1)

        return price, price_decomp.seasonal
