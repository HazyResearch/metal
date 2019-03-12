import os

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.1, freeze=False, cache_dir="./cache/"):
        super(BertEncoder, self).__init__()
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir)
        dropout = 0
        self.dropout = nn.Dropout(dropout)
        if freeze:
            for param in self.bert_model.parameters():
                param.requires_grad = False

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


def MulticlassHead(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def BinaryHead(input_dim):
    return MulticlassHead(input_dim, 2)


def RegressionHead(input_dim):
    return MulticlassHead(input_dim, 1)


class SoftAttentionModule(nn.Module):
    def __init__(self, input_dim, nonlinearity=nn.Tanh()):
        super(SoftAttentionModule, self).__init__()
        self.nonlinearity = nonlinearity
        # Initializing as ones to maintain structure
        self.W = torch.nn.Parameter(torch.ones(input_dim))
        self.W.requires_grad = True

    def forward(self, data):
        elementwise_multiply = torch.mul(self.W, data)
        nl = self.nonlinearity(elementwise_multiply)
        scaled_data = torch.mul(nl, data)
        return scaled_data
