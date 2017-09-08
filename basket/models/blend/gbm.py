import sys
import gc

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..utils import evaluate_f1score

pd.set_option('display.float_format', lambda x: '%.4f' % x)

K = 5
N_THREAD = 4


def train(seed):
    train_data = joblib.load("cache/fc_val.pkl")
    y = train_data["y"].values
    feature_names = [x for x in train_data.columns if x.startswith("f_")]
    gc.collect()
    print("# of features:", len(feature_names))

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        "objective": "binary",
        'metric': "binary_logloss",
        "bagging_fraction": 0.8,
        "bagging_seed": seed,
        "bagging_freq": 1,
        "feature_fraction": .8,
        "feature_fraction_seed": seed,
        "max_depth": 6,
        "learning_rate": 0.03,
        "verbose": 0,
        "num_threads": N_THREAD,
        # 'num_leaves': 32,
        # "min_data_in_leaf": 20,
    }

    group_kfold = GroupKFold(n_splits=K)
    gen = group_kfold.split(train_data, groups=train_data["user_id"])
    del train_data
    gc.collect()

    df_val_global = []
    df_test_global = None
    for train_idx, test_idx in gen:
        train_data = joblib.load("cache/fc_val.pkl")
        dtrain = lgb.Dataset(
            train_data.iloc[train_idx].loc[:, feature_names].values,
            label=y[train_idx], feature_name=feature_names)
        dval = lgb.Dataset(
            train_data.iloc[test_idx].loc[:, feature_names].values,
            label=y[test_idx],  feature_name=feature_names)
        del train_data
        gc.collect()
        bst = lgb.train(params, dtrain, num_boost_round=2000,
                        valid_sets=[dtrain, dval],
                        early_stopping_rounds=50, verbose_eval=50)
        # print(
        #     "".join(["{}: \n\t\t {:.6f}\n".format(name, val)
        #              for name, val in
        #              sorted(zip(bst.feature_name(),
        #                         bst.feature_importance("gain")),
        #                     key=lambda x: x[1], reverse=True)
        #              ])
        # )
        del dtrain, dval
        gc.collect()

        train_data = joblib.load("cache/fc_val.pkl")
        val_data = train_data.iloc[test_idx].copy()
        del train_data
        gc.collect()
        df_val = val_data[["user_id", "product"]].copy()
        df_val["prob"] = bst.predict(
            val_data[feature_names].values, num_iteration=bst.best_iteration
        )
        best_logloss = bst.best_score["valid_1"]["binary_logloss"]
        score = evaluate_f1score(df_val, .19)
        print("best logloss: {:.8f} score: {:.8f}".format(best_logloss, score))
        df_val_global.append(df_val)
        del val_data
        gc.collect()

        test_data = joblib.load("cache/fc_test.pkl")
        df_test = test_data[["user_id", "product"]].copy()
        df_test["prob"] = bst.predict(
            test_data[feature_names].values, num_iteration=bst.best_iteration
        )
        if df_test_global is None:
            df_test_global = df_test
        else:
            df_test_global["prob"] += df_test["prob"]
        del df_test, test_data
        gc.collect()

    df_val_global = pd.concat(df_val_global, axis=0)
    df_test_global["prob"] = df_test_global["prob"] / K
    return df_val_global, df_test_global


def main(seed=888):
    df_val, df_test = train(seed)
    joblib.dump([df_val, df_test], "cache/gbm_preds.pkl")
    print("F1:", evaluate_f1score(df_val, 0.19))
