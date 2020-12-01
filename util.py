import numpy as np
import pandas as pd
import torch
from model import LSTM, GRU, B_LSTM, BiLSTM
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
        return torch.from_numpy(np.array(xs)).float(), torch.from_numpy(np.array(ys)).float()

    @staticmethod
    def fill_na(df):
        alldate = pd.DataFrame(pd.date_range(start=df.index[0], end=df.index[-1])).set_index(0)
        df = pd.merge(alldate, df, left_index=True, right_index=True, how="left").fillna(
            method="ffill"
        )
        return df

    @staticmethod
    def call_models(p, n_f, seq_length, y_len, model_name="LSTM"):
        path = [p + "/ckpt_" + str(i) + ".pt" for i in range(1, 11)]
        param_dict = dict()
        losses = []
        names = ["ckpt_" + str(i) for i in range(1, 11)]
        for i in range(len(path)):
            checkpoint = torch.load(path[i])
            if model_name == "LSTM":
                param_dict[names[i]] = LSTM(
                    n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len
                )
            elif model_name == "GRU":
                param_dict[names[i]] = GRU(
                    n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len
                )
            elif model_name == "B_LSTM":
                param_dict[names[i]] = B_LSTM(
                    n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len
                )
            elif model_name == "BiLSTM":
                param_dict[names[i]] = BiLSTM(
                    n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len
                )
            param_dict[names[i]].load_state_dict(checkpoint["model_state_dict"])
            losses.append(checkpoint["test_loss"])
            param_dict[names[i]] = param_dict[names[i]].to("cuda")
        return param_dict, losses

    @staticmethod
    def unscale(scaler, y, n_f, col_num=0):  # returns a flat list of unscaled y.
        y_t = [np.repeat(t, n_f) for t in y]
        inv_y = scaler.inverse_transform(y_t)
        true_y = [i[col_num] for i in inv_y]
        return true_y

    @staticmethod
    def MAPE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class AgriDataSet(object):
    def __init__(self, price_path, weather_path):

        self.price_path = price_path
        self.weather_path = weather_path

        price = pd.read_csv(self.price_path, parse_dates=True, header=None)
        temp = price.groupby(2).count()[0]
        k = temp[temp == max(temp)].index[0]

        self.price = (
            price[price[2] == k]
            .groupby([0, 1])
            .mean()
            .pivot_table(index=0, columns=1, values=3)
            .mean(axis=1)
        )
        self.weather = pd.read_csv(self.weather_path, parse_dates=True, index_col=0)

    def _make_pr_dr_var(self, period, list_of_roll):

        price_decomp = seasonal_decompose(
            self.price, model="additive", period=period, extrapolate_trend="freq"
        )
        price = price_decomp.resid + price_decomp.trend  # extract seasonality

        l_df = [self.price]
        for i in list_of_roll:
            l_df.append(self.price.rolling(i).mean())
        price = pd.concat(l_df, axis=1)

        return price, price_decomp.seasonal

    def make_agri_dataset(self, period, list_of_roll):

        price, seasonal = self._make_pr_dr_var(period, list_of_roll)
        df = pd.merge(price, self.weather, left_index=True, right_index=True, how="left")

        return df, seasonal
