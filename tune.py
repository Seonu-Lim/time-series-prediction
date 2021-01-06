from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import inspect
from _tune_train import train_model
from _loss import CustomLoss
import argparse
import json
import torch
import numpy as np


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--X_dir", type=str)
    parser.add_argument("--y_dir", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--config_dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--n_sample", type=int)

    args = parser.parse_args()

    X_train = torch.load(args.X_dir)
    y_train = torch.load(args.y_dir)

    config = {
        "n_hidden": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
    }

    CL = CustomLoss(1, 2)
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=args.epoch, grace_period=1, reduction_factor=2
    )
    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])

    def train_func(config):
        train_model(
            X=X_train,
            y=y_train,
            num_epochs=args.epoch,
            loss_func=CL.custom_loss_1,
            model_name=args.model,
            config=config,
        )

    result = tune.run(
        train_func,
        resources_per_trial={"cpu": 2, "gpu": 2},
        config=config,
        num_samples=args.n_sample,
        scheduler=scheduler,
        progress_reporter=reporter,
    )
    best_trial = result.get_best_trial("loss", "min", "last")

    with open(args.config_dir, "w") as json_file:
        json.dump(best_trial.last_result["config"], json_file)

    last_loss = best_trial.last_result["loss"]
    print(f"Validation Loss of best model was {last_loss}.")


if __name__ == "__main__":
    main()
