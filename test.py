import torch
import numpy as np
import pandas as pd



def get_forecast(X,scaler,param_dict,start_date) :
    names = ['ckpt_'+ str(i) for i in range(1,11)]
    with torch.no_grad() :
        testdata = X[-1:]
        for j in range(10) :
            y_test_pred = param_dict[names[j]](testdata)
    y_t = [np.repeat([t],n_f) for t in y_test_pred.cpu().numpy()][0].reshape(y_len,n_f)
    inv_t = scaler.inverse_transform(y_t)
    true_t = [i[0] for i in inv_t]
    true_t.insert(0,true_y.iloc[-1].values[0])
    true_t = pd.DataFrame(true_t)
    true_t.index = pd.date_range(start_date,start_date + timedelta(days=14))
    return true_t



def test(param_dict,X,days) :
    names = ['ckpt_'+ str(i) for i in range(1,10)]
    i = days - 1
    with torch.no_grad() :
        allpreds = []
        for j in range(len(names)) :
            preds = []
            for i in range(len(X)):
                testdata = X_test[i:i + 1]
                y_test_pred = param_dict[names[j]](testdata)
                pred = y_test_pred[:, i].cpu().numpy()
                preds.append(pred[0])
            allpreds.append(preds)
    preds = [sum(x)/len(x) for x in zip(*allpreds)]
    return preds
