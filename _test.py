import torch
import numpy as np
import pandas as pd
from _model import LSTM, GRU, B_LSTM, BiLSTM, B_GRU
import json


class Prediction(object):
    def __init__(self, config_dir, checkpoint_dir):

        with open(config_dir, "r") as conf:
            config = json.load(conf)

        ckpt = torch.load(checkpoint_dir)

        n_f = ckpt["numbers"][0]
        seq_length = ckpt["numbers"][1]
        y_len = ckpt["numbers"][2]

        if ckpt["model_name"] == "LSTM":
            model = LSTM(
                n_features=n_f,
                n_hidden=config["n_hidden"],
                seq_len=seq_length,
                y_length=y_len,
            )
        elif ckpt["model_name"] == "GRU":
            model = GRU(
                n_features=n_f,
                n_hidden=config["n_hidden"],
                seq_len=seq_length,
                y_length=y_len,
            )
        elif ckpt["model_name"] == "B_LSTM":
            model = B_LSTM(
                n_features=n_f,
                n_hidden=config["n_hidden"],
                seq_len=seq_length,
                y_length=y_len,
            )
        elif ckpt["model_name"] == "BiLSTM":
            model = BiLSTM(
                n_features=n_f,
                n_hidden=config["n_hidden"],
                seq_len=seq_length,
                y_length=y_len,
            )
        elif ckpt["model_name"] == "B_GRU":
            model = B_GRU(
                n_features=n_f,
                n_hidden=config["n_hidden"],
                seq_len=seq_length,
                y_length=y_len,
            )

        model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to("cuda")

    def get_forecast(self, X):

        with torch.no_grad():
            test_data = X[-1:]
            forecast = self.model(test_data).cpu().numpy()
        return forecast[0]

    def get_prediction(self, X, days):

        j = days - 1
        with torch.no_grad():
            preds = []
            for i in range(len(X)):
                test_data = X[i : i + 1]
                y_test_pred = self.model(test_data)
                pred = y_test_pred[:, j].cpu().numpy()
                preds.append(pred[0])
        return preds
