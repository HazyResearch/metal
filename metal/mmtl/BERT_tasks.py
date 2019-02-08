import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import SST2Dataset, STSBDataset
from modules import BertBinaryHead, BertEncoder, BertMulticlassHead, BertRegressionHead
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer
from task import Task
from torch.utils.data import DataLoader, Dataset, TensorDataset

import metal
from metal.end_model import EndModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.utils.dataset_utils import get_all_dataloaders
from metal.mmtl.utils.metrics import pearson_corr, spearman_corr


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
    bert_model = "bert-base-uncased"
    bert_encoder = BertEncoder(bert_model)

    if task_name == "CoLA":
        raise NotImplementedError
    if task_name == "SST-2":
        dataloaders = get_all_dataloaders(SST2Dataset, bert_model)
        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            [Scorer(standard_metrics=["accuracy"])],
        )

    elif task_name == "MNLI":
        raise NotImplementedError
    elif task_name == "RTE":
        raise NotImplementedError
    elif task_name == "WNLI":
        raise NotImplementedError
    elif task_name == "QQP":
        raise NotImplementedError
    elif task_name == "MRPC":
        raise NotImplementedError
    elif task_name == "STS-B":
        dataloaders = get_all_dataloaders(STSBDataset, bert_model)
        scorer = Scorer(
            standard_metrics=["train/loss", "valid/loss"],
            custom_metric_fns=[pearson_corr, spearman_corr],
        )

        # x -> sigmoid -> [0,1], and compute mse_loss (y \in [0,1])
        loss_hat_func = lambda x, y: F.mse_loss(torch.sigmoid(x), y)
        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertRegressionHead(),
            scorer,
            loss_hat_func=loss_hat_func,
            output_hat_func=torch.sigmoid,
        )

    elif task_name == "QNLI":
        raise NotImplementedError
    elif task_name == "SNLI":
        raise NotImplementedError
    elif task_name == "SciTail":
        raise NotImplementedError
    else:
        raise ValueError(f"{task_name} does not exist.")
