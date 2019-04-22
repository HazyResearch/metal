import os

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

from metal.mmtl.modules import MetalModule


class BertRaw(MetalModule):
    """The Huggingface BertModel that passes on attention and drop extra linear layer

    Returns:
        encoded_layers: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len] \in {0, 1}
    """

    def __init__(
        self,
        bert_model_name,
        freeze_bert=False,
        reinit_bert=False,
        pooler=True,
        cache_dir="./cache/",
    ):
        super().__init__()
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        # NOTE: We may not want this extra [768, 768] Linear + tanh, but it's pretrained
        self.pooler = bert_model.pooler if pooler else None
        if freeze_bert:
            for param in self.parameters():
                param.requires_grad = False
        # Don't re-initialize
        if reinit_bert:
            bert_model.apply(bert_model.init_bert_weights)

    def forward(self, X, output_all_encoded_layers=False):
        tokens = X["data"]
        segments = X["segments"]
        attention_mask = X["masks"]
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens)
        if segments is None:
            segments = torch.zeros_like(tokens)

        # For details on extended attention ops, see the details in HuggingFace repo
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(tokens, segments)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return {"data": encoded_layers, "masks": attention_mask}


class BertExtractCls(MetalModule):
    """Extracts the hidden state for just the [CLS] token and may run through pooler"""

    def __init__(self, pooler=None, dropout=0.1):
        super().__init__()
        self.pooler = pooler
        if pooler is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        sequence_output = X["data"]
        if self.pooler is None:
            return {"data": sequence_output[:, 0]}
        else:
            # self.pooler pulls out the [CLS] token, applies square linear, then tanh
            output = self.pooler(sequence_output)
            return {"data": self.dropout(output)}


class BertTokenClassificationHead(nn.Module):
    """Predicts the class for each unmasked token in a sequence"""

    def __init__(self, emb_size, cardinality):
        """
        Args:
            emb_size: the size of the embedding for each token
            cardinality: the number of potential labels for each token
        """
        super().__init__()
        self.linear = nn.Linear(emb_size, cardinality)

    def forward(self, X):
        sequence_output, attention_mask = X
        return self.linear(sequence_output), attention_mask


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
