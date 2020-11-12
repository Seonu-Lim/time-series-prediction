import numpy as np
import pandas as pd
import torch
from model import (LSTM, B_LSTM)

class Util :

    @staticmethod
    def create_sequences(data,seq_length,y_length) :
        xs = []
        ys = []
        for i in range(len(data)-seq_length-y_length) :
            x= data[i:(i+seq_length)]
            y= [i[0] for i in data][(i+seq_length):(i+seq_length+y_length)]
            xs.append(x)
            ys.append(y)
        return torch.from_numpy(np.array(xs)).float(), torch.from_numpy(np.array(ys)).float()

    @staticmethod
    def fill_na(df) :
        alldate= pd.DataFrame(pd.date_range(start=df.index[0],end=df.index[-1])).set_index(0)
        df = pd.merge(alldate,df,left_index=True,right_index=True,how='left').fillna(method='ffill')
        return df


    @staticmethod
    def call_models(p,n_f,seq_length,y_len,bayesian=False) :
        path = [ p + '/ckpt_'+ str(i) + '.pt' for i in range(1,11)]
        param_dict = dict()
        losses = []
        names = ['ckpt_'+ str(i) for i in range(1,11)]
        for i in range(len(path)) :
            checkpoint = torch.load(path[i])
            if bayesian == False :
                param_dict[names[i]] = LSTM(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
            else :
                param_dict[names[i]] = B_LSTM(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
            param_dict[names[i]].load_state_dict(checkpoint['model_state_dict'])
            losses.append(checkpoint['loss'])
            param_dict[names[i]] = param_dict[names[i]].to('cuda')
        return param_dict, losses
        
    @staticmethod
    def unscale(scaler,y,n_f,col_num=0) : #returns a flat list of unscaled y.
        y_t = [np.repeat(t,n_f) for t in y]
        inv_y = scaler.inverse_transform(y_t)
        true_y = [i[col_num] for i in inv_y]
        return true_y


    @staticmethod
    def MAPE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
