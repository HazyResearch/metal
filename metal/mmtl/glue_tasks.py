import random

import numpy as np
import torch
import torch.nn as nn

from metal.contrib.modules.lstm_module import EmbeddingsEncoder, LSTMModule
from metal.mmtl.san import SAN, AverageLayer
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import ClassificationTask, RegressionTask
from metal.mmtl.utils.dataset_utils import get_all_dataloaders
from metal.mmtl.utils.metrics import (
    acc_f1,
    matthews_corr,
    mse,
    pearson_spearman,
    ranking_acc_f1,
)
from metal.utils import recursive_merge_dicts, set_seed

from metal.mmtl.modules import (  # BertEncoder,; BertHiddenLayer,
    BertExtractCls,
    BertRaw,
    BinaryHead,
    MulticlassHead,
    RegressionHead,
)

task_defaults = {
    # General
    "split_prop": None,
    "splits": ["train", "valid", "test"],
    "max_len": 512,
    "max_datapoints": -1,
    "seed": None,
    "dl_kwargs": {
        "batch_size": 16,
        "shuffle": True,  # Used only when split_prop is None; otherwise, use Sampler
    },
    "task_dl_kwargs": None,  # Overwrites dl kwargs e.g. {"STSB": {"batch_size": 2}}
    "encoder_type": "bert",
    "bert_model": "bert-base-uncased",  # Required for all encoders for BertTokenizer
    # BERT
    "bert_kwargs": {
        "freeze_bert": False,
        "pooler": True,  # If True, include the [768, 768] linear on top of [CLS] token
    },
    # LSTM
    "lstm_config": {
        "emb_size": 300,
        "hidden_size": 512,
        "vocab_size": 30522,  # bert-base-uncased-vocab.txt
        "bidirectional": True,
        "lstm_num_layers": 1,
    },
}


def create_tasks(task_names, **kwargs):
    assert len(task_names) > 0

    config = recursive_merge_dicts(task_defaults, kwargs)

    if config["seed"] is None:
        config["seed"] = np.random.randint(1e6)
        print(f"Using random seed: {config['seed']}")
    set_seed(config["seed"])

    # share bert encoder for all tasks

    if config["encoder_type"] == "bert":
        bert_kwargs = config["bert_kwargs"]
        bert_model = BertRaw(config["bert_model"], **bert_kwargs)
        # bert_model = BertEncoder(config["bert_model"], **bert_kwargs)
        # bert_encoder = BertEncoder(config["bert_model"], **bert_kwargs)
        # bert_hidden_layer = BertHiddenLayer(bert_encoder)
        if config["bert_model"] == "bert-base-uncased":
            neck_dim = 768
        elif config["bert_model"] == "bert-large-uncased":
            neck_dim = 1024
        input_module = bert_model
        cls_middle_module = BertExtractCls(pooler=bert_model.pooler)
    elif config["encoder_type"] == "lstm":
        # TODO: Allow these constants to be passed in as arguments
        msg = (
            "Non-BERT options are currently broken because of the BertExtractCls "
            "hardcoded into most task heads."
        )
        raise NotImplementedError(msg)
        lstm_config = config["lstm_config"]
        neck_dim = lstm_config["hidden_size"]
        if lstm_config["bidirectional"]:
            neck_dim *= 2
        lstm = LSTMModule(
            lstm_config["emb_size"],
            lstm_config["hidden_size"],
            lstm_reduction="max",
            bidirectional=lstm_config["bidirectional"],
            lstm_num_layers=lstm_config["lstm_num_layers"],
            encoder_class=EmbeddingsEncoder,
            encoder_kwargs={"vocab_size": lstm_config["vocab_size"]},
        )
        input_module = lstm
    else:
        raise NotImplementedError

    # create dict override dl_kwarg for specific task
    # e.g. {"STSB": {"batch_size": 2}}
    task_dl_kwargs = {}
    if config["task_dl_kwargs"]:
        task_configs_str = [
            tuple(config.split(".")) for config in config["task_dl_kwargs"].split(",")
        ]
        for (task_name, kwarg_key, kwarg_val) in task_configs_str:
            if kwarg_key == "batch_size":
                kwarg_val = int(kwarg_val)
            task_dl_kwargs[task_name] = {kwarg_key: kwarg_val}

    # creates task and appends to `tasks` list for each `task_name`
    tasks = []
    # auxiliary_tasks = kwargs.get("auxiliary_tasks", {})

    for task_name in task_names:

        # Override general dl kwargs with task-specific kwargs
        dl_kwargs = config["dl_kwargs"]
        if task_name in task_dl_kwargs:
            dl_kwargs.update(task_dl_kwargs[task_name])

        # create data loaders for task
        dataloaders = get_all_dataloaders(
            task_name if not task_name.endswith("_SAN") else task_name[:-4],
            config["bert_model"],
            max_len=config["max_len"],
            dl_kwargs=dl_kwargs,
            split_prop=config["split_prop"],
            max_datapoints=config["max_datapoints"],
            splits=config["splits"],
            seed=config["seed"],
            generate_uids=kwargs.get("generate_uids", False),
        )

        if task_name == "COLA":
            scorer = Scorer(
                standard_metrics=["accuracy"],
                custom_metric_funcs={matthews_corr: ["matthews_corr"]},
            )
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )

        elif task_name == "SST2":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
            )

        elif task_name == "MNLI":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=MulticlassHead(neck_dim, 3),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "RTE":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "WNLI":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "QQP":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(
                    custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}
                ),
            )

        elif task_name == "MRPC":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(
                    custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}
                ),
            )

        elif task_name == "STSB":
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

            task = RegressionTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=RegressionHead(neck_dim),
                scorer=scorer,
            )

        elif task_name == "QNLI":
            task = ClassificationTask(
                name=task_name,
                data_loaders=dataloaders,
                input_module=input_module,
                middle_module=cls_middle_module,
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        # --------- NON-STANDARD TASK HEADS BELOW THIS POINT ---------

        # NOTE: These tasks
        # elif task_name == "MNLI_SAN":
        #     task = ClassificationTask(
        #         name= "MNLI",
        #         data_loaders=dataloaders,
        #         SAN(
        #             bert_model=bert_encoder,
        #             emb_size=neck_dim,
        #             hidden_size=neck_dim,
        #             num_classes=3,
        #             k=5,
        #         ),
        #         AverageLayer(),
        #         Scorer(standard_metrics=["accuracy"]),
        #     )

        # elif task_name == "RTE_SAN":
        #     task = ClassificationTask(
        #         "RTE",
        #         data_loaders=dataloaders,
        #         SAN(
        #             bert_model=bert_encoder,
        #             emb_size=neck_dim,
        #             hidden_size=neck_dim,
        #             num_classes=2,
        #             k=5,
        #         ),
        #         AverageLayer(),
        #         Scorer(standard_metrics=["accuracy"]),
        #     )

        # elif task_name == "WNLI_SAN":
        #     task = ClassificationTask(
        #         "WNLI",
        #         data_loaders=dataloaders,
        #         SAN(
        #             bert_model=bert_encoder,
        #             emb_size=neck_dim,
        #             hidden_size=neck_dim,
        #             num_classes=2,
        #             k=5,
        #         ),
        #         AverageLayer(),
        #         Scorer(standard_metrics=["accuracy"]),
        #     )

        # elif task_name == "QQP_SAN":
        #     task = ClassificationTask(
        #         "QQP",
        #         data_loaders=dataloaders,
        #         SAN(
        #             bert_model=bert_encoder,
        #             emb_size=neck_dim,
        #             hidden_size=neck_dim,
        #             num_classes=2,
        #             k=5,
        #         ),
        #         AverageLayer(),
        #         Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
        #     )

        # elif task_name == "MRPC_SAN":
        #     task = ClassificationTask(
        #         "MRPC",
        #         data_loaders=dataloaders,
        #         SAN(
        #             bert_model=bert_encoder,
        #             emb_size=neck_dim,
        #             hidden_size=neck_dim,
        #             num_classes=2,
        #             k=5,
        #         ),
        #         AverageLayer(),
        #         Scorer(custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}),
        #     )

        # elif task_name == "QNLIR":
        #     # QNLI ranking task
        #     def ranking_loss(scores, y_true, gamma=1.0):
        #         scores = torch.sigmoid(scores)
        #         # TODO: find consistent way to map labels to {0,1}
        #         y_true = (1 - y_true) + 1
        #         # TODO: if we're using dev set then these computations won't work
        #         # make sure we don't compute loss for evaluation
        #         max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #         pos_scores = torch.exp(
        #             gamma
        #             * (max_pool((y_true * scores.view(-1)).view(1, 1, -1)).view(-1))
        #         )
        #         neg_scores = torch.exp(
        #             gamma
        #             * (max_pool(((1 - y_true) * scores.view(-1)).view(1, 1, -1))).view(
        #                 -1
        #             )
        #         )
        #         log_likelihood = torch.log(pos_scores / (pos_scores + neg_scores))
        #         return -torch.mean(log_likelihood)

        #     scorer = Scorer(
        #         custom_metric_funcs={ranking_acc_f1: ["accuracy", "f1", "acc_f1"]},
        #         standard_metrics=[],
        #     )
        #     task = ClassificationTask(
        #         name="QNLIR",
        #         data_loaders=dataloaders,
        #         input_module=input_module,
        #         head_module=RegressionHead(neck_dim),
        #         scorer=scorer,
        #         loss_hat_func=ranking_loss,
        #         output_hat_func=torch.sigmoid,
        #     )

        # --------- AUXILIARY TASKS BELOW THIS POINT ---------
        # if task_name in auxiliary_tasks.keys():
        #     if "BLEU" in auxiliary_tasks[task_name]:
        #         bleu_dataloaders = {
        #             split: get_bleu_dataloader(dataloaders[split])
        #             for split in dataloaders.keys()
        #         }

        #         # Do we need a loss_hat_func or output_hat_fun?
        #         tasks.append(
        #             RegressionTask(
        #                 name=f"{task_name}_BLEU",
        #                 data_loaders=bleu_dataloaders,
        #                 input_module=bert_hidden_layer,
        #                 head_module=RegressionHead(neck_dim),
        #                 scorer=Scorer(custom_metric_funcs={mse: ["mse"]}),
        #             )
        #         )

        tasks.append(task)
    return tasks
