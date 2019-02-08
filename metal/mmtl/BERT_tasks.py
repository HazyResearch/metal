import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import COLADataset, SST2Dataset
from modules import BertBinaryHead, BertEncoder, BertMulticlassHead
from pytorch_pretrained_bert import BertForMaskedLM, BertModel, BertTokenizer
from sklearn.metrics import matthews_corrcoef
from task import Task
from torch.utils.data import DataLoader, Dataset, TensorDataset

import metal
from metal.end_model import EndModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.utils.dataset_utils import get_all_dataloaders


def create_task(task_name):
    bert_model = "bert-base-uncased"
    bert_encoder = BertEncoder(bert_model)

    if task_name == "CoLA":
        dataloaders = get_all_dataloaders(COLADataset, bert_model)

        def matthews_corr(targets, predictions):
            import pdb

            pdb.set_trace()
            predictions = np.argmax(predictions, 1)
            matthews = matthews_corrcoef(targets, predictions)
            return {"matthews_corr": matthews}

        return Task(
            task_name,
            dataloaders,
            bert_encoder,
            BertBinaryHead(),
            Scorer(
                standard_metrics=["accuracy"],
                custom_train_funcs=[matthews_corr],
                custom_valid_funcs=[matthews_corr],
            ),
        )
    if task_name == "SST-2":
        dataloaders = get_all_dataloaders(SST2Dataset, bert_model)
        return Task(task_name, dataloaders, bert_encoder, BertBinaryHead())
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
        raise NotImplementedError
    elif task_name == "QNLI":
        raise NotImplementedError
    elif task_name == "SNLI":
        raise NotImplementedError
    elif task_name == "SciTail":
        raise NotImplementedError
    else:
        raise ValueError(f"{task_name} does not exist.")
