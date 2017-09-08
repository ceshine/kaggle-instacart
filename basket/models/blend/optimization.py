"""F1 score optimization and submission creation
"""
import numpy as np
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
import pandas as pd

from ..utils import get_true_labels, f1_score_single

N_THREAD = 4
K = 10


class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + \
                    (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + \
                    P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        P = np.sort(P)[::-1]
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]
        if best_k == 0:
            assert predNone
            return 1, predNone, max_f1
        return P[best_k - 1], predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


def construct_product_list(df):
    none_prob = df[df["product"] == 0].prob.values[0]
    df = df[df["product"] != 0]
    best_P, predNone, _ = F1Optimizer.maximize_expectation(
        df.prob.values, none_prob)
    l = df.loc[df.prob >= best_P, "product"].tolist()
    if not l or predNone:
        l.append("None")
    return (df["user_id"].iloc[0], l)


def evaluate(val_data):
    val_label_dict = dict(
        Parallel(n_jobs=N_THREAD)(
            delayed(construct_product_list)(
                val_data.loc[idx])
            for user_id, idx in tqdm(val_data.groupby("user_id").groups.items()))
    )
    true_labels = get_true_labels().copy()
    true_label_dict = dict(zip(true_labels.user_id, true_labels[0]))
    scores = [
        f1_score_single(val_label_dict[u], true_label_dict[u]) for u in val_label_dict.keys()
    ]
    return np.mean(scores)


def construct_product_string(df):
    none_prob = df[df["product"] == 0].prob.values[0]
    df = df[df["product"] != 0]
    best_P, predNone, _ = F1Optimizer.maximize_expectation(
        df.prob.values, none_prob)
    l = df.loc[df.prob >= best_P, "product"].astype(str).tolist()
    if not l or predNone:
        l.append("None")
    return df["user_id"].iloc[0], " ".join(l)


def create_submission(test_pred, *, filename="sub_new.csv"):
    orders = pd.read_csv(
        'data/raw/orders.csv',
        dtype={
            'order_id': np.int32,
            'eval_set': 'category',
            'user_id': np.int32
        })
    orders = orders[orders.eval_set == "test"]
    test_pred = test_pred.sort_values(["user_id", "product"])
    print(test_pred["prob"].describe())
    products = pd.DataFrame(
        Parallel(n_jobs=N_THREAD)(
            delayed(construct_product_string)(
                test_pred.loc[idx])
            for user_id, idx in tqdm(test_pred.groupby("user_id").groups.items())),
        columns=["user_id", "products"]
    )
    print(products.head())
    products = products.merge(
        orders[["order_id", "user_id"]], on="user_id", how="right").fillna("None")
    products = products.sort_values(["order_id"])
    products.rename(columns={0: "products"}, inplace=True)
    products[["order_id", "products"]].to_csv(filename, index=False)


def main(model_name="gbm_preds"):
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    val_data, test_data = joblib.load("cache/{}.pkl".format(model_name))
    val_data = val_data.reset_index(drop=True)
    score = evaluate(val_data)
    print("CV score:", score)
    filename = "sub_{}.csv".format(int(round(score * 1000000)))
    print("Filename:", filename)
    create_submission(test_data, filename=filename)


if __name__ == "__main__":
    main()
