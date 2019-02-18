import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from metal.mmtl.modules import BertEncoder


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    by: xiaodl
    """

    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
            :param x: batch * len * input_size
        """
        if self.training is False or self.dropout_p == 0:
            return x

        if len(x.size()) == 3:
            mask = Variable(
                1.0
                / (1 - self.dropout_p)
                * torch.bernoulli(
                    (1 - self.dropout_p)
                    * (x.data.new(x.size(0), x.size(2)).zero_() + 1)
                ),
                requires_grad=False,
            )
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


class AverageLayer(nn.Module):
    def __init__(self):
        super(AverageLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 2)


class LinearSelfAttn(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = DropoutWrapper(dropout_p=dropout)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        scores = self.linear(x).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        alpha = torch.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class BilinearSelfAttn(nn.Module):
    def __init__(self, x_size, y_size, dropout=0.1):
        super(BilinearSelfAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)
        self.dropout = DropoutWrapper(dropout_p=dropout)

    def forward(self, x, y, x_mask):
        x = self.dropout(x)
        y = self.dropout(y)
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float("inf"))
        beta = torch.softmax(xWy, 1)
        return beta.unsqueeze(1).bmm(x).squeeze(1)


def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training:
        dropout_p = 0.0
    new_data = (1 - dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = Variable(
        1.0 / (1 - dropout_p) * torch.bernoulli(new_data), requires_grad=False
    )
    return mask


class SAN(nn.Module):
    def __init__(
        self, bert_model, emb_size=100, hidden_size=100, num_classes=2, k=5, dropout=0.1
    ):
        super(SAN, self).__init__()
        self.bert_model = bert_model
        self.sent1_attn = LinearSelfAttn(input_size=emb_size)
        self.sent2_attn = LinearSelfAttn(input_size=emb_size)
        self.attn = BilinearSelfAttn(emb_size, emb_size)
        self.final_linear = nn.Linear(emb_size * 4, num_classes)
        self.rnn = nn.GRUCell(emb_size, hidden_size)
        self.k = k
        self.num_classes = num_classes
        self.dropout_num = dropout
        self.dropout = DropoutWrapper(dropout_p=self.dropout_num)
        self.alpha = Parameter(torch.zeros(1, 1, 1))

    def forward(self, X):
        idx_matrix, seg_matrix, mask_matrix = X

        batch_size = idx_matrix.size(0)

        output_layer, _ = self.bert_model.forward(X)
        #        print(output_layer.size())
        res = []

        #        print("idx_matrix", idx_matrix)
        #        print("seg_matrix", seg_matrix)
        #        print("mask_matrix", mask_matrix)
        #        print("sent1_matrix", (1 - mask_matrix + seg_matrix))
        #        print("sent2_matrix", (1 - seg_matrix))

        sk = self.sent1_attn(output_layer, (1 - seg_matrix).byte())
        xk = self.sent2_attn(output_layer, (1 - mask_matrix + seg_matrix).byte())
        #         sk = self.sent1_attn(output_layer, (1 - mask_matrix + seg_matrix).byte())

        for i in range(self.k):
            f = self.final_linear(torch.cat((sk, xk, torch.abs(sk - xk), sk * xk), 1))
            res.append(f)
            xk = self.attn(output_layer, sk, (1 - mask_matrix + seg_matrix).byte())
            #             xk = self.sent2_attn(output_layer, sk, (1 - seg_matrix).byte())
            #            f = self.final_linear(torch.cat((sk, xk, torch.abs(sk - xk), sk * xk), 1))
            #            res.append(f)
            sk = self.dropout(sk)
            sk = self.rnn(xk, sk)

        #        print(res)

        mask = generate_mask(
            self.alpha.data.new(batch_size, self.k), self.dropout_num, self.training
        )
        mask = [m.contiguous() for m in torch.unbind(mask, 1)]
        scores_list = [
            mask[idx].view(batch_size, 1).expand_as(inp) * torch.softmax(inp, 1)
            for idx, inp in enumerate(res)
        ]
        scores = torch.stack(scores_list, 2)
        return scores
