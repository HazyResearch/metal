import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import SST2Dataset
from modules import BertBinaryHead, BertEncoder, BertMulticlassHead
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer
from task import Task
from torch.utils.data import DataLoader, Dataset, TensorDataset

import metal
from metal.end_model import EndModel
from metal.mmtl.scorer import Scorer


"""
dataloaders = createBertDataloader(
    "SST-2",
    batch_sz=8,
    sent1_idx=0,
    sent2_idx=-1,
    label_idx=1,
    header=1,
    label_fn=lambda x: int(x) + 1,
)
"""


def create_task(task_name):
    if task_name == "SST-2":
        bert_model = "bert-base-uncased"
        dataloaders = []
        for split in ["train", "dev"]:
            dataset = SST2Dataset(split=split, bert_model=bert_model, max_len=-1)
            dataloaders.append(dataset.get_dataloader())
        dataloaders.append(None)
        return Task(
            task_name,
            dataloaders,
            BertEncoder(bert_model),
            BertBinaryHead(),
            [Scorer(standard_metrics=["accuracy"])],
        )
    else:
        return None
