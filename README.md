# Time - Series Prediction using LSTM

This repository contains codes for time series prediction by LSTM, especially when the inputs are multivariate and the outputs are univariate.

## Prerequisites

1. nvidia gpu

2. torch == 1.6.0

3. blitz (for bayesian lstm)

4. Python >= 3.6

5. ray >= 1.0.1

The required Python packages can be simply installed through:

```sh
pip install -r requirements.txt
```

Before running tune.py, run :
```sh
pip install "ray[tune]"
```

For hyperparameter tuning, run :
```sh
python tune.py --X_dir temp/X.pt --y_dir temp/y.pt --epoch 300 --config_dir temp/config.json --model GRU --n_sample 10
```