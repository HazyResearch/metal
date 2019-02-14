import torch
import torch.nn as nn

from metal.mmtl.modules import BertEncoder


class AverageLayer(nn.Module):
    def __init__(self, k=5):
        super(AverageLayer, self).__init__()
        self.k = k

    def forward(self, x):
        return x / self.k


class LinearSelfAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        scores = self.linear(x).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        alpha = torch.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class BilinearSelfAttn(nn.Module):
    def __init__(self, x_size, y_size):
        super(BilinearSelfAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y, x_mask):
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float("inf"))
        beta = torch.softmax(xWy, 1)
        return beta.unsqueeze(1).bmm(x).squeeze(1)


class SAN(nn.Module):
    def __init__(self, bert_model, emb_size=100, hidden_size=100, num_classes=2, k=5):
        super(SAN, self).__init__()
        self.bert_model = bert_model
        self.sent1_attn = LinearSelfAttn(input_size=emb_size)
        self.sent2_attn = BilinearSelfAttn(emb_size, emb_size)
        self.final_linear = nn.Linear(emb_size * 4, num_classes)
        self.rnn = nn.GRU(emb_size, hidden_size, 1, batch_first=True)
        self.k = k
        self.num_classes = num_classes

    def forward(self, X):
        idx_matrix, seg_matrix, mask_matrix = X

        batch_size = idx_matrix.size(0)

        output_layer, _ = self.bert_model.forward(X)

        res = output_layer.new_zeros((batch_size, self.num_classes))

        sk = self.sent1_attn(output_layer, (1 - mask_matrix + seg_matrix).byte())

        for i in range(self.k):

            xk = self.sent2_attn(output_layer, sk, (1 - seg_matrix).byte())
            _, sk = self.rnn(xk.unsqueeze(1), sk.unsqueeze(0))
            sk = sk.squeeze(0)

            f = torch.softmax(
                self.final_linear(torch.cat((sk, xk, torch.abs(sk - xk), sk * xk), 1)),
                1,
            )
            res += f

        return res
