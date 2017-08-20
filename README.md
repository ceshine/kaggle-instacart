# Instacart Market Basket Analysis Competition

**(Work In Progress)**

This project is based on my solution to the competition and heavily inspired by
[Sean Vasquez's solution](https://github.com/sjvasquez/instacart-basket-prediction).

Currently implementation can get .3988546 private and .4001607 public scores.

TODOs:

1. Port the feature-engineered LightGBM solution into this project (my actual submissions are based on this model)
2. Try more ideas from Sean's solution:
    * Bernoulli mixture models
    * Order size models
    * Skip-Gram with Negative Sampling (SGNS) models
    * Non-Negative Matrix Factorization (NNMF) models
3. Neural network models in second layer

## Requirements

The code is tested on a computer with **16GB RAM** and a **GTX 1070 card (with 8GB VRAM)**.

Python packages:

* joblib==0.11
* lightgbm==2.0.4
* torch==0.2.0.post2
* numpy==1.13.1
* pandas==0.20.2
* scikit-learn==0.18.2

## Usage

### Preprocessing data

```
python basket/preprocessing/prepare_users.py
```

### Train and Predict

```
python train.py
```

This will:

1. Train a neural network with T-4 to T-1 orders as training target.
2. Extract the states of the final layer when predicting T orders.
3. Use the extracted states to train a GBM model using LightGBM.
4. Optimize the predictions of the GBM model against F1 metric and create a submission file.
