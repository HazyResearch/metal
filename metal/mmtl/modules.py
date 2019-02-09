import os

import torch.nn as nn
from dataset import BERTDataset
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer

BERT_small_outdim = 768
BERT_large_outdim = 1024


class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.1):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        tokens, segments, mask = data
        # TODO: check if we should return all layers or just last hidden representation
        _, hidden_layer = self.bert_model(tokens, segments, mask)
        hidden_layer = self.dropout(hidden_layer)

        return hidden_layer


def BertMulticlassHead(k):
    return nn.Linear(BERT_small_outdim, k, bias=False)


def BertBinaryHead():
    return BertMulticlassHead(2)


def BertRegressionHead():
    return BertMulticlassHead(1)
