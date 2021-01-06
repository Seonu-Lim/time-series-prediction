from _model import LSTM, GRU, B_LSTM, B_GRU, BiLSTM
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import sys
import json
from _loss import CustomLoss


def train_model(X, y, num_epochs, loss_func, model_name, config_dir, checkpoint_dir):

    if not os.path.exists(f"model/{checkpoint_dir}"):
        os.makedirs(f"model/{checkpoint_dir}")

    with open(config_dir, "r") as conf:
        config = json.load(conf)

    train_hist = np.zeros(num_epochs)
    n_f = X.shape[2]
    seq_length = X.shape[1]
    y_len = y.shape[1]

    numbers = (n_f, seq_length, y_len)

    ### choose model ###
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

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )

    ### train loop ###
    for t in range(num_epochs):
        for _, samples in enumerate(dataloader):
            X_train, y_train = samples
            X_train = X_train.to("cuda")
            y_train = y_train.to("cuda")
            y_pred = model(X_train)
            loss = torch.sqrt(loss_fn(y_pred.float(), y_train))
            train_hist[t] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_hist[t] /= len(dataloader)
        print("Epoch {:3d} train loss: {:.4f}".format(t + 1, train_hist[t]))

    ### saving checkpoint ###
    ckpt = {
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "numbers": numbers,
        "model_name": model_name,
    }
    torch.save(ckpt, f"model/{checkpoint_dir}/checkpoint.pt")
    print("Training Done.")


def keep_training(X, y, num_epochs, loss_func, config_dir, checkpoint_dir):

    with open(config_dir, "r") as conf:
        config = json.load(conf)

    ckpt = torch.load(checkpoint_dir, map_location="cuda:0")

    model_name = ckpt["model_name"]
    n_f = ckpt["numbers"][0]
    seq_length = ckpt["numbers"][1]
    y_len = ckpt["numbers"][2]

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

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    loss_fn = loss_func
    train_hist = np.zeros(num_epochs)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )

    ### train loop ###
    for t in range(num_epochs):
        for _, samples in enumerate(dataloader):
            X_train, y_train = samples
            X_train = X_train.to("cuda")
            y_train = y_train.to("cuda")
            y_pred = model(X_train)
            loss = torch.sqrt(loss_fn(y_pred.float(), y_train))
            train_hist[t] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_hist[t] /= len(dataloader)
        print("Epoch {:3d} train loss: {:.4f}".format(t + 1, train_hist[t]))

    ### saving checkpoint ###
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "numbers": ckpt["numbers"],
        "model_name": model_name,
    }
    torch.save(ckpt, checkpoint_dir)
