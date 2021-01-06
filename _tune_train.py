from _model import LSTM, GRU, B_LSTM, B_GRU, BiLSTM
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import sys
from ray import tune
from _loss import CustomLoss


def train_model(X, y, num_epochs, loss_func, model_name, config, checkpoint_dir=None):

    # hyperparameters for tuning
    # config : lr(loguniform), n_hidden(samplefrom), batch_size(choice)
    # ASHAScheduler : num_epochs

    if not os.path.exists(f"model/{checkpoint_dir}"):
        os.makedirs(f"model/{checkpoint_dir}")
    train_hist = np.zeros(num_epochs)
    val_hist = np.zeros(num_epochs)

    n_f = X.shape[2]
    seq_length = X.shape[1]
    y_len = y.shape[1]

    ### split to train and val ###
    X_split = torch.split(
        X, [int(X.shape[0] * 0.8), X.shape[0] - int(X.shape[0] * 0.8)]
    )
    y_split = torch.split(
        y, [int(y.shape[0] * 0.8), y.shape[0] - int(y.shape[0] * 0.8)]
    )

    if model_name == "LSTM":
        model = LSTM(
            n_features=n_f,
            n_hidden=config["n_hidden"],
            seq_len=seq_length,
            y_length=y_len,
        )
    elif model_name == "GRU":
        model = GRU(
            n_features=n_f,
            n_hidden=config["n_hidden"],
            seq_len=seq_length,
            y_length=y_len,
        )
    elif model_name == "B_LSTM":
        model = B_LSTM(
            n_features=n_f,
            n_hidden=config["n_hidden"],
            seq_len=seq_length,
            y_length=y_len,
        )
    elif model_name == "BiLSTM":
        model = BiLSTM(
            n_features=n_f,
            n_hidden=config["n_hidden"],
            seq_len=seq_length,
            y_length=y_len,
        )
    elif model_name == "B_GRU":
        model = B_GRU(
            n_features=n_f,
            n_hidden=config["n_hidden"],
            seq_len=seq_length,
            y_length=y_len,
        )
    else:
        print("Input valid model name.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    ### multiGPUsupport ###
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loss_fn = loss_func

    ### train on cuda ###
    model = model.to("cuda")
    X_val = X_split[1].to("cuda")
    y_val = y_split[1].to("cuda")

    dataset = TensorDataset(X_split[0], y_split[0])
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )

    ### train loop ###
    for t in range(num_epochs):
        for batch_idx, samples in enumerate(dataloader):
            X_train, y_train = samples
            X_train = X_train.to("cuda")
            y_train = y_train.to("cuda")
            y_pred = model(X_train)
            loss = torch.sqrt(loss_fn(y_pred.float(), y_train))
            with torch.no_grad():
                y_val_pred = model(X_val)
                val_loss = torch.sqrt(loss_fn(y_val_pred.float(), y_val))
            val_hist[t] += val_loss.item()
            train_hist[t] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_hist[t] /= len(dataloader)
        val_hist[t] /= len(dataloader)
        # print('Epoch {:3d} train loss: {:.4f} val loss: {:.4f}'.format(t + 1, train_hist[t], val_hist[t]))
        tune.report(loss=val_hist[t])

        ### save model state dict, TODO : change to saving full model when training###
        with tune.checkpoint_dir(t) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
    print("Training Done.")
