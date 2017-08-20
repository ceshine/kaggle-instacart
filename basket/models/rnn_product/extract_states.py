import joblib
import torch
import pandas as pd
import numpy as np

from .model import get_model
from .data_loader import InstacartDataLoader
from ..utils import load_user_list


DEBUG = 0


def extract(model, user_list, has_labels=False):
    eval_loader = InstacartDataLoader(
        train_users=user_list, shuffle=False,
        batch_size=1024, num_workers=4, pbar=True)
    model.eval()
    preds = []
    labels = []
    for batch, batch_labels, sequence_lengths in eval_loader:
        _, feature_tensor = model(
            batch, sequence_lengths,
            inference=True, extract_fc=True)
        features = feature_tensor.numpy()
        if DEBUG:
            assert features.shape[1] == batch[0].shape[1]
        batch_preds = np.zeros((batch[0].shape[1], 2 + features.shape[2]))
        batch_preds[:, 0] = batch[1][:, 0]  # user_id
        batch_preds[:, 1] = batch[1][:, 1]  # product
        for i in range(batch[0].shape[1]):
            batch_preds[i, 2:] = features[sequence_lengths[i] - 1, i, :]
        preds.append(batch_preds)
        if has_labels:
            labels.append(batch_labels[
                sequence_lengths - 1,
                np.arange(0, len(sequence_lengths))
            ])
    if has_labels:
        return np.concatenate(preds, axis=0), np.concatenate(labels)
    return np.concatenate(preds, axis=0)


def main():
    train_users, test_users = load_user_list(train_sample_ratio=1.)
    model = get_model()
    model.load_state_dict(torch.load("cache/best_model.state"))
    model.eval()
    val_features, val_labels = extract(model, train_users, has_labels=True)
    df_val = pd.DataFrame(
        val_features, columns=(
            ["user_id", "product"] +
            ["f_{}".format(i) for i in range(val_features.shape[1] - 2)]
        )
    )
    df_val["y"] = val_labels
    for col in ("user_id", "product"):
        df_val[col] = df_val[col].astype(np.int32)
    df_val["y"] = df_val["y"].astype(np.int8)
    joblib.dump(df_val, "cache/rnn_product_fc_val.pkl")
    del df_val, val_features, val_labels
    test_features = extract(model, test_users, has_labels=False)
    df_test = pd.DataFrame(
        test_features, columns=(
            ["user_id", "product"] +
            ["f_{}".format(i) for i in range(test_features.shape[1] - 2)]
        )
    )
    for col in ("user_id", "product"):
        df_test[col] = df_test[col].astype(np.int32)
    joblib.dump(df_test, "cache/rnn_product_fc_test.pkl")


if __name__ == "__main__":
    main()
