import math

import torch
import torch.nn as nn


class SparseLinearModule(nn.Module):
    def __init__(self, embed_size, vocab_size, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.W = nn.Embedding(
            vocab_size, embed_size, sparse=True, padding_idx=padding_idx
        )
        self.b = nn.Parameter(torch.Tensor(embed_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.vocab_size)
        self.W.weight.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.W.weight.data[self.padding_idx].fill_(0)

    def forward(self, X):
        """Execute sparse linear layer

        Args:
            X: an [n, h] torch.LongTensor containing up to h indices of features
                whose weights should be looked up and used in a sparse linear
                multiplication.
        """
        return self.W(X).sum(dim=1) + self.b
