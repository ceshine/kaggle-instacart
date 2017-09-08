"""RNN Bernoulli Mixture Model
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
from .model import base_model, batch_to_tensors

tqdm.pandas(desc="BAAAR")

DEBUG = 0
N_WORKER = 4


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
        self.fc1 = nn.Linear(
            (self.department_emb.embedding_dim +
             self.aisle_emb.embedding_dim +
             # self.user_emb.embedding_dim +
             self.product_emb.embedding_dim +
             self.hod_emb.embedding_dim + self.dow_emb.embedding_dim +
             rnn_input_size + hidden_size), 20
        )
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)

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
        # Final FC layer
        fc1_input_flatten = torch.cat(
            (
                output,
                rnn_input[:output.size()[0], :, :]
            ), 2
        ).view(-1, output.size()[2] + rnn_input.size()[2])
        fc1 = F.relu(self.fc1(fc1_input_flatten))
        fc1 = self.bn1(fc1)
        fc2 = F.relu(self.fc2(fc1))
        fc2 = self.bn2(fc2)
        # x = F.dropout(fc2, p=self.dense_dropout_p, training=self.training)
        mixing_coefs = fc2[:, :(fc2.size()[1] // 2)]
        mixing_coefs = F.softmax(
            mixing_coefs - torch.min(mixing_coefs, 1, keepdim=True)[0]
        )  # B x N
        ps = F.sigmoid(fc2[:, (fc2.size()[1] // 2):])  # B x N
        prob = torch.sum(mixing_coefs * ps, 1).view(
            output.size()[0], output.size()[1])
        if extract_fc:
            return prob, fc1.data.cpu().view(
                output.size()[0], output.size()[1], fc1.size()[1]
            )
        return prob


def get_model():
    return LSTMModel(
        rnn_input_size=5,
        hidden_size=256, n_layers=3,
        rnn_dropout_p=0.2, dense_dropout_p=0.5
    ).cuda()


def main(seed=888):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    df_val, df_test_new = base_model(
        model=get_model(), train_loops=((2, 0.001, 3), (2, 0.0005, 2), (2, 0.0005, 1)),
        batch_size=256, predict=True
    )
    joblib.dump([df_val, df_test_new], "cache/rnn_product_bmm.pkl")
    print("F1 score:", evaluate_f1score(df_val, .2))
