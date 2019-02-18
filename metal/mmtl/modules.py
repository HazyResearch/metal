import os

import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.1, cache_dir="./cache/"):
        super(BertEncoder, self).__init__()
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        tokens, segments, mask = data
        output_layer, hidden_layer = self.bert_model(
            tokens, segments, mask, output_all_encoded_layers=False
        )
        output_layer = self.dropout(output_layer)
        hidden_layer = self.dropout(hidden_layer)
        return output_layer, hidden_layer


class BertHiddenLayer(nn.Module):
    def __init__(self, bert_model):
        super(BertHiddenLayer, self).__init__()
        self.bert_model = bert_model

    def forward(self, data):
        _, hidden_layer = self.bert_model.forward(data)
        return hidden_layer


def BertMulticlassHead(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim, bias=True)


def BertBinaryHead(input_dim):
    return BertMulticlassHead(input_dim, 2)


def BertRegressionHead(input_dim):
    return BertMulticlassHead(input_dim, 1)
