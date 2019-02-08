import os

import torch.nn as nn
from dataset import BERTDataset
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer

BERT_small_outdim = 768
BERT_large_outdim = 1024


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased"
        )  # also try bert-base-multilingual-cased (recommended)

    def forward(self, data):
        tokens, segments, mask = data
        # TODO: check if we should return all layers or just last hidden representation
        _, hidden_layer = self.bert_model(tokens, segments, mask)
        return hidden_layer


def createBertDataloader(
    task_name,
    batch_sz=8,
    sent1_idx=0,
    sent2_idx=-1,
    label_idx=1,
    header=1,
    label_fn=lambda x: int(x) + 1,
):
    # model = 'bert-base-uncased' # also try bert-base-multilingual-cased (recommended)
    src_path = os.path.join(os.environ["GLUEDATA"], task_name, "{}.tsv")
    dataloaders = {}
    for split in ["train", "dev"]:
        dataset = BERTDataset(
            src_path.format(split),
            sent1_idx=sent1_idx,
            sent2_idx=sent2_idx,
            label_idx=label_idx,
            skip_rows=header,
            label_fn=label_fn,  # labels are scores [1, 2] (multiclass with cardinality k)
        )
        dataloaders[split] = dataset.get_dataloader(batch_size=batch_sz)
    return [dataloaders["train"], dataloaders["dev"], None]


def BertMulticlassHead(k):
    return nn.Linear([BERT_small_outdim, k])


def BertBinaryHead():
    return BertMulticlassHead(2)
