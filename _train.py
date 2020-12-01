from model import LSTM, GRU, B_LSTM, BiLSTM
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def train_model(num_epochs, X, y, path, model_name="LSTM"):

    n_f = X.shape[2]
    seq_length = X.shape[1]
    y_len = y.shape[1]

    X_split = torch.split(X, X.shape[0] // 10 + 1)
    y_split = torch.split(y, y.shape[0] // 10 + 1)
    train_hist = np.zeros((num_epochs, (len(X_split))))
    test_hist = np.zeros((num_epochs, (len(X_split))))
    for i in range(len(X_split)):
        if model_name == "LSTM":
            model = LSTM(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
        elif model_name == "GRU":
            model = GRU(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
        elif model_name == "B_LSTM":
            model = B_LSTM(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
        elif model_name == "BiLSTM":
            model = BiLSTM(n_features=n_f, n_hidden=128, seq_len=seq_length, y_length=y_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()  # TODO : apply custom loss #########
        model = model.to("cuda")
        X_test = X_split[i].to("cuda")
        y_test = y_split[i].to("cuda")
        idx = [k for k in range(len(X_split) - 1) if k != i]
        X_train = torch.cat([X_split[x] for x in idx])
        y_train = torch.cat([y_split[x] for x in idx])
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        print(str(i + 1) + "th fold training start")
        for t in range(num_epochs):  # epochs per fold
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
                    )  # TODO : apply custom loss #########
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
