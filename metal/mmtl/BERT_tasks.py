import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import COLADataset, SST2Dataset, STSBDataset
from modules import BertBinaryHead, BertEncoder, BertMulticlassHead, BertRegressionHead
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer
from sklearn.metrics import matthews_corrcoef
from task import Task
from torch.utils.data import DataLoader, Dataset, TensorDataset

import metal
from metal.end_model import EndModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.utils.dataset_utils import get_all_dataloaders
from metal.mmtl.utils.metrics import matthews_corr, pearson_corr, spearman_corr


def create_task(task_name):
    bert_model = "bert-base-uncased"
    dataloaders = get_all_dataloaders(task_name, bert_model, 0.80)
    bert_encoder = BertEncoder(bert_model)

    if task_name == "COLA":
        scorer = Scorer(
            standard_metrics=[],
            custom_train_funcs=[matthews_corr],
            custom_valid_funcs=[matthews_corr],
        )
        return Task(task_name, dataloaders, bert_encoder, BertBinaryHead(), scorer)

    if task_name == "SST2":
        return Task(task_name, dataloaders, bert_encoder, BertBinaryHead())

    elif task_name == "MNLI":

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertMulticlassHead(3),
            Scorer(standard_metrics=["accuracy"]),
        )

    elif task_name == "RTE":

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            Scorer(standard_metrics=["accuracy"]),
        )

    elif task_name == "WNLI":

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            Scorer(standard_metrics=["accuracy"]),
        )

    elif task_name == "QQP":

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            Scorer(standard_metrics=["accuracy"]),
        )

    elif task_name == "MRPC":

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            Scorer(standard_metrics=["accuracy"]),
        )

    elif task_name == "STSB":
        scorer = Scorer(
            standard_metrics=[],
            custom_train_funcs=[pearson_corr, spearman_corr],
            custom_valid_funcs=[pearson_corr, spearman_corr],
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
