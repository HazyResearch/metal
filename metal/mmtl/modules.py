import torch.nn as nn
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer


class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.1, cache_dir="."):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        tokens, segments, mask = data
        _, hidden_layer = self.bert_model(tokens, segments, mask)
        hidden_layer = self.dropout(hidden_layer)
        return hidden_layer


def BertMulticlassHead(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim, bias=False)


def BertBinaryHead(input_dim):
    return BertMulticlassHead(input_dim, 2)


def BertRegressionHead(input_dim):
    return BertMulticlassHead(input_dim, 1)
