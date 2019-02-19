import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.mmtl.modules import (
    BertBinaryHead,
    BertEncoder,
    BertHiddenLayer,
    BertMulticlassHead,
    BertRegressionHead,
)
from metal.mmtl.san import SAN, AverageLayer
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import ClassificationTask, RegressionTask
from metal.mmtl.utils.dataset_utils import get_all_dataloaders
from metal.mmtl.utils.metrics import (
    acc_f1,
    matthews_corr,
    pearson_spearman,
    ranking_acc_f1,
)


def create_tasks(
    task_names,
    bert_model="bert-base-uncased",
    split_prop=None,
    max_len=512,
    dl_kwargs={},
    bert_kwargs={},
    bert_output_dim=768,
    max_datapoints=-1,
    splits=["train", "valid", "test"],
    seed=None,
):
    assert len(task_names) > 0

    if seed is None:
        seed = np.random.randint(1e6)
        print(f"Using random seed: {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # share bert encoder for all tasks
    bert_encoder = BertEncoder(bert_model, **bert_kwargs)
    bert_hidden_layer = BertHiddenLayer(bert_encoder)

    # creates task and appends to `tasks` list for each `task_name`
    tasks = []
    for task_name in task_names:

        # create data loaders for task
        dataloaders = get_all_dataloaders(
            task_name if not task_name.endswith("_SAN") else task_name[:-4],
            bert_model,
            max_len=max_len,
            dl_kwargs=dl_kwargs,
            split_prop=split_prop,
            max_datapoints=max_datapoints,
            splits=splits,
        )

        if task_name == "COLA":
            scorer = Scorer(
                standard_metrics=["accuracy"],
                custom_metric_funcs={matthews_corr: ["matthews_corr"]},
            )
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                    scorer,
                )
            )

        if task_name == "SST2":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                )
            )

        if task_name == "MNLI":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertMulticlassHead(bert_output_dim, 3),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "MNLI_SAN":
            tasks.append(
                ClassificationTask(
                    "MNLI",
                    dataloaders,
                    SAN(
                        bert_model=bert_encoder,
                        emb_size=bert_output_dim,
                        hidden_size=bert_output_dim,
                        num_classes=3,
                        k=5,
                    ),
                    AverageLayer(),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "RTE":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "RTE_SAN":
            tasks.append(
                ClassificationTask(
                    "RTE",
                    dataloaders,
                    SAN(
                        bert_model=bert_encoder,
                        emb_size=bert_output_dim,
                        hidden_size=bert_output_dim,
                        num_classes=2,
                        k=5,
                    ),
                    AverageLayer(),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "WNLI":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "WNLI_SAN":
            tasks.append(
                ClassificationTask(
                    "WNLI",
                    dataloaders,
                    SAN(
                        bert_model=bert_encoder,
                        emb_size=bert_output_dim,
                        hidden_size=bert_output_dim,
                        num_classes=2,
                        k=5,
                    ),
                    AverageLayer(),
                    Scorer(standard_metrics=["accuracy"]),
                )
            )

        if task_name == "QQP":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                    Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
                )
            )

        if task_name == "QQP_SAN":
            tasks.append(
                ClassificationTask(
                    "QQP",
                    dataloaders,
                    SAN(
                        bert_model=bert_encoder,
                        emb_size=bert_output_dim,
                        hidden_size=bert_output_dim,
                        num_classes=2,
                        k=5,
                    ),
                    AverageLayer(),
                    Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
                )
            )

        if task_name == "MRPC":
            tasks.append(
                ClassificationTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertBinaryHead(bert_output_dim),
                    Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
                )
            )

        if task_name == "MRPC_SAN":
            tasks.append(
                ClassificationTask(
                    "MRPC",
                    dataloaders,
                    SAN(
                        bert_model=bert_encoder,
                        emb_size=bert_output_dim,
                        hidden_size=bert_output_dim,
                        num_classes=2,
                        k=5,
                    ),
                    AverageLayer(),
                    Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
                )
            )

        if task_name == "STSB":
            scorer = Scorer(
                standard_metrics=[],
                custom_metric_funcs={
                    pearson_spearman: [
                        "pearson_corr",
                        "spearman_corr",
                        "pearson_spearman",
                    ]
                },
            )

            tasks.append(
                RegressionTask(
                    task_name,
                    dataloaders,
                    bert_hidden_layer,
                    BertRegressionHead(bert_output_dim),
                    scorer,
                )
            )

        if task_name == "QNLI":
            qnli_head = nn.Linear(bert_output_dim, 2, bias=False)
            tasks.append(
                ClassificationTask(
                    name="QNLI",
                    data_loaders=dataloaders,
                    input_module=bert_hidden_layer,
                    head_module=qnli_head,
                    scorer=Scorer(standard_metrics=["accuracy"]),
                    # TODO: small fix to map 1 to 1 and 2 to 0 but need more consitent way of doing this
                    loss_hat_func=lambda y_hat, y_true: F.cross_entropy(
                        y_hat, y_true - 1
                    ),
                    output_hat_func=partial(F.softmax, dim=1),
                )
            )

        if task_name == "QNLIR":
            # QNLI ranking task
            def ranking_loss(scores, y_true, gamma=1.0):
                scores = torch.sigmoid(scores)
                # TODO: find consistent way to map labels to {0,1}
                y_true = (1 - y_true) + 1
                # TODO: if we're using dev set then these computations won't work
                # make sure we don't compute loss for evaluation
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
                pos_scores = torch.exp(
                    gamma
                    * (max_pool((y_true * scores.view(-1)).view(1, 1, -1)).view(-1))
                )
                neg_scores = torch.exp(
                    gamma
                    * (max_pool(((1 - y_true) * scores.view(-1)).view(1, 1, -1))).view(
                        -1
                    )
                )
                log_likelihood = torch.log(pos_scores / (pos_scores + neg_scores))
                return -torch.mean(log_likelihood)

            scorer = Scorer(
                custom_metric_funcs={ranking_acc_f1: ["accuracy", "f1", "acc_f1"]},
                standard_metrics=[],
            )
            tasks.append(
                ClassificationTask(
                    name="QNLIR",
                    data_loaders=dataloaders,
                    input_module=bert_hidden_layer,
                    head_module=BertRegressionHead(bert_output_dim),
                    scorer=scorer,
                    loss_hat_func=ranking_loss,
                    output_hat_func=torch.sigmoid,
                )
            )

    return tasks
