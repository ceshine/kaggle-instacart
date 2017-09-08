"""RNN + CNN product-level model
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm

from .masked_bce import masked_bce
from .data_loader import InstacartDataLoader
from ..utils import evaluate_f1score, load_user_list

tqdm.pandas(desc="BAAAR")

DEBUG = 0
N_WORKER = 4


class CausalConv1d(nn.Conv1d):
    """ Causal 1D Convolution Layer

    Accepted input layout: (N, C, L)

    Reference: https://github.com/pytorch/pytorch/issues/1333
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        # Transfrom to 2-D layout (N, C, H, W)
        x = F.pad(
            x.unsqueeze(2), (
                self.left_padding, 0, 0, 0
            )
        ).squeeze(2)
        return super(CausalConv1d, self).forward(x)


def batch_to_tensors(input_batch, inference):
    """Convert numpy arrays to tensors
    """
    input_tensor = Variable(torch.from_numpy(
        input_batch[0][:, :, :5]).float().cuda(), volatile=inference)
    hod_tensor = Variable(torch.from_numpy(
        input_batch[0][:, :, 5]).long().cuda(), volatile=inference)
    dow_tensor = Variable(torch.from_numpy(
        input_batch[0][:, :, 6]).long().cuda(), volatile=inference)
    user_tensor = Variable(torch.from_numpy(
        input_batch[1][:, 0]).long().cuda(), volatile=inference).unsqueeze(1)
    product_tensor = Variable(torch.from_numpy(
        input_batch[1][:, 1]).long().cuda(), volatile=inference).unsqueeze(1)
    department_tensor = Variable(torch.from_numpy(
        input_batch[1][:, 2]).long().cuda(), volatile=inference).unsqueeze(1)
    aisle_tensor = Variable(torch.from_numpy(
        input_batch[1][:, 3]).long().cuda(), volatile=inference).unsqueeze(1)
    return (
        input_tensor, hod_tensor, dow_tensor, user_tensor,
        product_tensor, department_tensor, aisle_tensor)


class LSTMModel(nn.Module):
    def __init__(self, rnn_input_size,
                 hidden_size, n_layers, rnn_dropout_p, dense_dropout_p,
                 *, n_department=23, n_aisle=135, n_user=206210,
                 n_product=49689):
        super(LSTMModel, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.department_emb = nn.Embedding(n_department, 3)
        self.aisle_emb = nn.Embedding(n_aisle, 10)
        # self.user_emb = nn.Embedding(n_user, 5)
        self.product_emb = nn.Embedding(n_product, 20)
        self.hod_emb = nn.Embedding(24, 3)
        self.dow_emb = nn.Embedding(7, 2)
        self.dense_dropout_p = dense_dropout_p
        self.rnn = nn.LSTM(
            (self.hod_emb.embedding_dim + self.dow_emb.embedding_dim +
             self.department_emb.embedding_dim +
             self.aisle_emb.embedding_dim +
             # self.user_emb.embedding_dim +
             self.product_emb.embedding_dim +
             rnn_input_size),
            hidden_size,
            n_layers,
            dropout=rnn_dropout_p)
        self.fc2 = nn.Linear(
            (self.department_emb.embedding_dim +
             self.aisle_emb.embedding_dim +
             # self.user_emb.embedding_dim +
             self.product_emb.embedding_dim +
             self.hod_emb.embedding_dim + self.dow_emb.embedding_dim +
             rnn_input_size +
             hidden_size * 2), 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.o2o = nn.Linear(50, 1)
        self.fc1 = nn.Linear(
            (self.department_emb.embedding_dim +
             self.aisle_emb.embedding_dim +
             # self.user_emb.embedding_dim +
             self.product_emb.embedding_dim +
             self.hod_emb.embedding_dim + self.dow_emb.embedding_dim +
             rnn_input_size), hidden_size
        )
        self.convolution_layers = []
        for i in range(3):
            self.convolution_layers.append(
                CausalConv1d(
                    hidden_size, hidden_size,
                    kernel_size=2, dilation=(2**(i))
                )
            )
        for i, layer in enumerate(self.convolution_layers):
            self.add_module('cnn_{}'.format(i), layer)

    def forward(self, input_batch, sequence_lengths, *, inference=False, extract_fc=False):
        (rnn_input, hod_input, dow_input, user, product,
         department, aisle) = batch_to_tensors(input_batch, inference)
        aisle_embedded = self.aisle_emb(aisle).squeeze(1)
        department_embedded = self.department_emb(department).squeeze(1)
        # user_embedded = self.user_emb(user).squeeze(1)
        product_embedded = self.product_emb(product).squeeze(1)
        hod_embedded = self.hod_emb(hod_input)
        dow_embedded = self.dow_emb(dow_input)
        embedding_input = torch.cat(
            (aisle_embedded, department_embedded,  product_embedded,
             # user_embedded
             ), 1).unsqueeze(0)
        embedding_input = embedding_input.repeat(
            rnn_input.size()[0], 1, 1
        )
        # RNN sub-model
        rnn_input = torch.cat(
            (rnn_input, hod_embedded, dow_embedded, embedding_input), 2)
        rnn_input_packed = pack_padded_sequence(
            rnn_input, sequence_lengths, batch_first=False
        )
        output, _ = self.rnn(rnn_input_packed)
        output, _ = pad_packed_sequence(output, batch_first=False)
        # CNN sub-model
        fc1 = F.relu(self.fc1(rnn_input)).permute(1, 2, 0)
        for conv_layer in self.convolution_layers:
            fc1 += F.relu(conv_layer(fc1))
        fc1 = fc1.permute(2, 0, 1)
        # Final FC layer
        fc2_input_flatten = torch.cat(
            (
                output,
                rnn_input[:output.size()[0], :, :],
                fc1[:output.size()[0], :, :]
            ), 2
        ).view(
            output.size()[0] * output.size()[1],
            (output.size()[2] + rnn_input.size()[2] + fc1.size()[2])
        )
        fc2 = F.relu(self.fc2(fc2_input_flatten))
        fc2 = self.bn2(fc2)
        # x = F.dropout(fc2, p=self.dense_dropout_p, training=self.training)
        prob = F.sigmoid(self.o2o(fc2)).view(
            output.size()[0], output.size()[1]
        )
        if extract_fc:
            return prob, fc2.data.cpu().view(
                output.size()[0], output.size()[1], fc2.size()[1]
            )
        return prob


class GRUModel(LSTMModel):

    def __init__(self, rnn_input_size, cont_input_size,
                 hidden_size, n_layers, rnn_dropout_p, dense_dropout_p,
                 *, n_department=23, n_aisle=135, n_user=206210,
                 n_product=49689):
        super(GRUModel, self).__init__(
            rnn_input_size, cont_input_size,
            hidden_size, n_layers, rnn_dropout_p, dense_dropout_p,
            n_department, n_aisle, n_user, n_product
        )
        self.rnn = nn.GRU(
            (self.hod_emb.embedding_dim + self.dow_emb.embedding_dim +
             rnn_input_size),
            hidden_size,
            n_layers,
            dropout=rnn_dropout_p)


def train(input_batch, target_batch, model, optimizer, sequence_lengths, max_lookback=2):
    # Zero gradients
    optimizer.zero_grad()
    preds = model(input_batch, sequence_lengths)
    # Loss calculation and back-propagation
    loss = masked_bce(
        preds,
        Variable(torch.cuda.FloatTensor(target_batch)),
        sequence_lengths,
        max_lookback=max_lookback
    )
    loss.backward()
    # Update parameters with optimizers
    optimizer.step()
    return loss.data


def predict(input_batch, model, sequence_lengths):
    preds = model(input_batch, sequence_lengths, inference=True)
    return preds


def process_eval_dataset(model, eval_users, batch_size):
    eval_loader = InstacartDataLoader(
        train_users=eval_users, shuffle=False,
        batch_size=batch_size, num_workers=N_WORKER,
        pbar=True)
    model.eval()
    losses = []
    val_preds = []
    for batch, labels, sequence_lengths in eval_loader:
        prob = predict(batch, model, sequence_lengths)
        losses.append(masked_bce(
            prob,
            Variable(torch.cuda.FloatTensor(labels)),
            sequence_lengths, max_lookback=1
        ).data.cpu().numpy()[0])
        batch_preds = np.zeros((batch[0].shape[1], 3))
        batch_preds[:, 0] = batch[1][:, 0]  # user_id
        batch_preds[:, 1] = batch[1][:, 1]  # product
        batch_preds[:, 2] = prob.data.cpu().numpy()[
            np.array(sequence_lengths) - 1,
            np.arange(0, len(sequence_lengths))
        ]
        val_preds.append(batch_preds)
    val_data = pd.DataFrame(np.concatenate(val_preds), columns=[
        "user_id", "product", "prob"])
    for col in ("product", "user_id"):
        val_data[col] = val_data[col].astype(np.int32)
    if DEBUG:
        print("Eval size:", val_data.shape[0])
    score = evaluate_f1score(val_data, .2)
    return losses, score, val_data


def process_train_dataset(model, optimizer, train_users, eval_users, test_users, batch_size, max_lookback=2):
    train_loader = InstacartDataLoader(
        train_users=train_users,
        eval_users=eval_users + test_users, shuffle=True,
        batch_size=batch_size, num_workers=N_WORKER, pbar=True)
    losses = []
    for batch, labels, sequence_lengths in train_loader:
        if batch is None:
            continue
        losses.append(train(batch, labels, model, optimizer,
                            sequence_lengths, max_lookback=max_lookback))
    return losses


def process_test_dataset(model, test_users, batch_size):
    model.eval()
    test_preds = []
    test_loader = InstacartDataLoader(
        test_users=test_users, shuffle=False,
        batch_size=batch_size, num_workers=N_WORKER, pbar=True)
    for batch, _, sequence_lengths in test_loader:
        batch_preds = np.zeros((batch[0].shape[1], 3))
        batch_preds[:, 0] = batch[1][:, 0]  # user_id
        batch_preds[:, 1] = batch[1][:, 1]  # product
        batch_preds[:, 2] = predict(
            batch, model, sequence_lengths
        ).data.cpu().numpy()[
            np.array(sequence_lengths) - 1,
            np.arange(0, len(sequence_lengths))
        ]
        test_preds.append(batch_preds)
    test_data = pd.DataFrame(
        np.concatenate(test_preds), columns=[
            "user_id", "product", "prob"])
    for col in ["user_id", "product"]:
        test_data[col] = test_data[col].astype(np.int32)
    return test_data


def get_model():
    return LSTMModel(
        rnn_input_size=5,
        hidden_size=256, n_layers=2,
        rnn_dropout_p=0.1, dense_dropout_p=0.5
    ).cuda()


def base_model(model, train_loops, batch_size=256, predict=True, model_name="rnn_product"):
    train_users, test_users = load_user_list(train_sample_ratio=1.)
    eval_losses = []
    losses_avg = []
    best_val = None
    for n_epoch, lr, max_lookback in train_loops:
        print("MAX_LOOKBACK: {}  LEARNING_RATE: {}".format(max_lookback, lr))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for e in range(n_epoch):
            model.train()
            local_train_losses = process_train_dataset(
                model, optimizer, [], train_users, test_users, batch_size, max_lookback=max_lookback
            )
            local_eval_losses, eval_score, val_data = process_eval_dataset(
                model, train_users, batch_size=2048
            )
            eval_losses.append(np.mean(local_eval_losses))
            print(val_data["prob"].describe())
            if len(eval_losses) == 1 or eval_losses[-1] < np.min(eval_losses[:-1]):
                print("Saving model...")
                # save the best model
                torch.save(model.state_dict(),
                           "cache/best_model_{}.state".format(model_name))
                best_val = val_data
            local_train_losses = np.array(
                [z.cpu().numpy()[0] for z in local_train_losses])
            losses_avg += [np.mean(x)
                           for x in np.array_split(local_train_losses, 10)]
            print("Epoch {} train loss: {:.4f} eval loss: {:.4f} eval score:{:.4f}"
                  "\n======================\n".format(
                      e, np.mean(local_train_losses), eval_losses[-1],
                      eval_score))
    if predict:
        model.load_state_dict(
            torch.load("cache/best_model_{}.state".format(model_name)))
        test_data = process_test_dataset(model, test_users, batch_size=1024)
        return best_val, test_data


def main(seed=888):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    df_val, df_test_new = base_model(
        model=get_model(), train_loops=((3, 0.001, 3), (3, 0.0005, 1)),
        batch_size=256, predict=True
    )
    joblib.dump([df_val, df_test_new], "cache/rnn_product.pkl")
    print("F1 score:", evaluate_f1score(df_val, .2))
