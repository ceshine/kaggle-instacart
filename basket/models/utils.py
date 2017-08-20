"""Utility functions
"""

import random

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

IDIR = 'data/raw/'
N_THREAD = 4
memory = joblib.Memory(cachedir="cache/", verbose=1, mmap_mode="r")

tqdm.pandas(desc="BAAAR")


@memory.cache
def get_true_labels():
    train_data = pd.read_csv(
        IDIR + 'order_products__train.csv',
        dtype={
            'order_id': np.int32,
            'product_id': np.int32,
            'reordered': np.int8,
        }
    )
    orders = pd.read_csv(
        IDIR + 'orders.csv',
        dtype={
            'order_id': np.int32,
            'user_id': np.int32
        })
    train_data = train_data.merge(
        orders[["order_id", "user_id"]], on="order_id")
    true_labels = train_data[["user_id", "product_id", "reordered"]].sort_values(
        ["user_id", "product_id"]).groupby(
        "user_id").apply(
        lambda df: df[df.reordered == 1]["product_id"].tolist()
            if df["reordered"].sum() > 0 else ["None"]).reset_index()
    return true_labels


def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0:
        return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)


def f1_score(y_true, y_pred):
    # return np.mean(joblib.Parallel(n_jobs=N_THREAD)(
    #     joblib.delayed(f1_score_single)(x, y) for x, y in zip(y_true, y_pred)))
    return np.mean([
        f1_score_single(x, y) for x, y in zip(y_true, y_pred)
    ])


def construct_product_string(df):
    l = df.loc[df.pass_threshold, "product"].tolist()
    l = [str(x) if x != 0 else "None" for x in l]
    if not l:
        l.append("None")
    return " ".join(l)


def construct_product_list(df, threshold):
    l = df.loc[df.prob > threshold, "product"].tolist()
    l = [x if x != 0 else "None" for x in l]
    if not l:
        l.append("None")
    return l


def evaluate_f1score(val_data, threshold=0.2):
    true_labels = get_true_labels().copy()
    val_data = val_data.copy()
    val_labels = val_data.groupby("user_id").progress_apply(
        construct_product_list, threshold=threshold).reset_index()
    val_labels = val_labels.merge(
        true_labels, on="user_id", suffixes=["_pred", "_true"])
    score = f1_score(val_labels["0_true"].tolist(),
                     val_labels["0_pred"].tolist())
    return score


def load_user_list(*, train_sample_ratio=1.0):
    orders = pd.read_csv(
        IDIR + 'orders.csv',
        dtype={
            'order_id': np.int32,
            'eval_set': 'category',
            'user_id': np.int32
        })
    test_users = orders[orders.eval_set == "test"]["user_id"].unique().tolist()
    train_users = orders[
        orders.eval_set == "train"]["user_id"].unique().tolist()
    if train_sample_ratio < 1:
        train_users = random.sample(train_users, int(
            train_sample_ratio * len(train_users)))
    print("train users: %d" % len(train_users))
    print("test users:  %d" % len(test_users))
    return train_users, test_users
