from model import LSTM, GRU, B_LSTM, BiLSTM
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


class Trainer(object) :

    def __init__(self, foldk, num_epochs, X, y) :

        super(Trainer,self).__init__()

        self.foldk = foldk
        self.num_epochs = num_epochs
        self.X = X
        self.y = y

        self.n_f = X.shape[2]
        self.seq_length = X.shape[1]
        self.y_len = y.shape[1]

    def train_model(self, path, loss_function, model_name="LSTM"):

        if not os.path.exists(f"model/{path}") :
            os.makedirs(f"model/{path}")
        X_split = torch.split(self.X, self.X.shape[0] // self.foldk + 1)
        y_split = torch.split(self.y, self.y.shape[0] // self.foldk + 1)
        train_hist = np.zeros((self.num_epochs, (len(X_split))))
        test_hist = np.zeros((self.num_epochs, (len(X_split))))
        for i in range(len(X_split)):
            if model_name == "LSTM":
                model = LSTM(n_features=self.n_f, n_hidden=128, seq_len=self.seq_length, y_length=self.y_len)
            elif model_name == "GRU":
                model = GRU(n_features=self.n_f, n_hidden=128, seq_len=self.seq_length, y_length=self.y_len)
            elif model_name == "B_LSTM":
                model = B_LSTM(n_features=self.n_f, n_hidden=128, seq_len=self.seq_length, y_length=self.y_len)
            elif model_name == "BiLSTM":
                model = BiLSTM(n_features=self.n_f, n_hidden=128, seq_len=self.seq_length, y_length=self.y_len)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = loss_function
            model = model.to("cuda")
            X_test = X_split[i].to("cuda")
            y_test = y_split[i].to("cuda")
            idx = [k for k in range(len(X_split) - 1) if k != i]
            X_train = torch.cat([X_split[x] for x in idx])
            y_train = torch.cat([y_split[x] for x in idx])
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
            print(str(i + 1) + "th fold training start")
            for t in range(self.num_epochs):  # epochs per fold
                for batch_idx, samples in enumerate(dataloader):
                    X_train, y_train = samples
                    X_train = X_train.to("cuda")
                    y_train = y_train.to("cuda")
                    y_pred = model(X_train)
                    loss = torch.sqrt(loss_fn(y_pred.float(), y_train))
                    with torch.no_grad():
                        y_test_pred = model(X_test)
                        test_loss = torch.sqrt(
                            loss_fn(y_test_pred.float(), y_test)
                        )  
                    test_hist[t, i] += test_loss.item()
                    train_hist[t, i] += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                train_hist[t, i] /= len(dataloader)
                test_hist[t, i] /= len(dataloader)
                print(
                    "Epoch {:3d} train loss: {:.4f} test loss: {:.4f}".format(
                        t + 1, train_hist[t, i], test_hist[t, i]
                    )
                )
            print("saving checkpoint at fold", i + 1)

            torch.save(
                {
                    "epoch": num_epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_loss": test_hist[num_epochs - 1, i],
                },
                f"model/{path}/ckpt_{i+1}.pt",
            )
        print("Training Done.")
