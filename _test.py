import torch


class Prediction(object):
    def __init__(self, X, param_dict, n_f, seq_length, y_len):

        self.X = X
        self.param_dict = param_dict
        self.n_f = n_f
        self.seq_length = seq_length
        self.y_len = y_len
        self.names = list(param_dict.keys())

    def get_forecast(self):
        with torch.no_grad():
            testdata = self.X[-1:]
            forecast = []
            for j in range(len(self.names)):
                y_t_p = self.param_dict[self.names[j]](testdata).cpu().numpy()
                forecast.append(y_t_p)
        forecast = [sum(x) / len(x) for x in zip(*forecast)]
        forecast = forecast[0].tolist()
        return forecast

    def get_prediction(self, days):
        j = days - 1
        with torch.no_grad():
            allpreds = []
            for j in range(len(self.names)):
                preds = []
                for i in range(len(self.X)):
                    testdata = self.X[i : i + 1]
                    y_test_pred = self.param_dict[self.names[j]](testdata)
                    pred = y_test_pred[:, j].cpu().numpy()
                    preds.append(pred[0])
                allpreds.append(preds)
        preds = [sum(x) / len(x) for x in zip(*allpreds)]
        return preds
