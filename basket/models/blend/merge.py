import joblib
import pandas as pd


def merge(is_test=False):
    model_list = [
        ("rnn_product_fc_True", "rnn_cnn"),
        ("rnn_product_fc_False", "rnn_bmm")
    ]
    suffix = "_test" if is_test else "_val"
    all_features = []
    for model_name, model_short_name in model_list:
        tmp_data = joblib.load("cache/{}.pkl".format(model_name + suffix))
        feature_names = [x for x in tmp_data.columns if x.startswith("f_")]
        all_features.append(tmp_data[feature_names])
        all_features[-1].columns = [x +
                                    model_short_name for x in feature_names]
    all_features.append(tmp_data[["user_id", "product"]])
    if not is_test:
        all_features.append(tmp_data[["y"]])
    return pd.concat(all_features, axis=1)


def main():
    val_data = merge(False)
    joblib.dump(val_data, "cache/fc_val.pkl")
    del val_data
    test_data = merge(True)
    joblib.dump(test_data, "cache/fc_test.pkl")
