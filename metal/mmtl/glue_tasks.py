import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.contrib.modules.lstm_module import EmbeddingsEncoder, LSTMModule
from metal.end_model import IdentityModule
from metal.mmtl.auxiliary_tasks import SPACY_TAGS, auxiliary_task_functions
from metal.mmtl.modules import (
    BertExtractCls,
    BertRaw,
    BertTokenClassificationHead,
    BinaryHead,
    MulticlassHead,
    RegressionHead,
    SoftAttentionModule,
)
from metal.mmtl.payload import Payload
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import ClassificationTask, RegressionTask, TokenClassificationTask
from metal.mmtl.utils.dataloaders import get_all_dataloaders
from metal.mmtl.utils.metrics import (
    acc_f1,
    matthews_corr,
    mse,
    pearson_spearman,
    ranking_acc_f1,
)
from metal.utils import recursive_merge_dicts, set_seed

ALL_TASKS = [
    "COLA",
    "SST2",
    "MNLI",
    "SNLI",
    "RTE",
    "WNLI",
    "QQP",
    "MRPC",
    "STSB",
    "QNLI",
]

# List of tasks requiring Spacy tokenization
SPACY_TASKS = ["SPACY_NER", "SPACY_POS"]

# Auxiliary task loss multplier
AUX_LOSS_MULTIPLIER = 1.0

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
    # NOTE: This dropout only applies to the output of the pooler; it will not change
    # the dropout rate of BERT (defaults to 0.1) or add dropout to other modules.
    # The main BERT module ends with a dropout layer already, so token-based tasks
    # that do not use BertExtractCls middle module do not need additional dropout first
    "dropout": 0.1,
    # BERT
    "encoder_type": "bert",
    "bert_model": "bert-base-uncased",  # Required for all encoders for BertTokenizer
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
    "attention_config": {
        "attention_module": None,  # None, soft currently accepted
        "nonlinearity": "tanh",  # tanh, sigmoid currently accepted
    },
    # Auxiliary Tasks
    "auxiliary_task_dict": {  # A map of each aux. task to the payloads it applies to
        "THIRD": ALL_TASKS,
        "BLEU": ["MNLI", "RTE", "WNLI", "QQP", "MRPC", "STSB", "QNLI"],  # sent pairs
        "SPACY_NER": ALL_TASKS,
        "SPACY_POS": ALL_TASKS,
    },
}


def create_tasks_and_payloads(task_names, **kwargs):
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
        if "base" in config["bert_model"]:
            neck_dim = 768
        elif "large" in config["bert_model"]:
            neck_dim = 1024
        input_module = bert_model
        cls_middle_module = BertExtractCls(
            pooler=bert_model.pooler, dropout=config["dropout"]
        )
    else:
        raise NotImplementedError

    # Create dict override dl_kwarg for specific task
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

    tasks = []
    payloads = []
    for task_name in task_names:
        # Pull out names of auxiliary tasks to be dealt with in a second step
        # TODO: fix this logic for cases where auxiliary task for task_name has
        # its own payload
        has_payload = task_name not in config["auxiliary_task_dict"]

        # Note whether this task has auxiliary tasks that apply to it and require spacy
        run_spacy = False
        for aux_task, target_payloads in config["auxiliary_task_dict"].items():
            run_spacy = run_spacy or (
                task_name in target_payloads
                and aux_task in SPACY_TASKS
                and aux_task in task_names
            )

        # Override general dl kwargs with task-specific kwargs
        dl_kwargs = copy.deepcopy(config["dl_kwargs"])
        if task_name in task_dl_kwargs:
            dl_kwargs.update(task_dl_kwargs[task_name])

        # Each primary task has data_loaders to load
        if has_payload:
            data_loaders = get_all_dataloaders(
                task_name if not task_name.endswith("_SAN") else task_name[:-4],
                config["bert_model"],
                max_len=config["max_len"],
                dl_kwargs=dl_kwargs,
                split_prop=config["split_prop"],
                max_datapoints=config["max_datapoints"],
                splits=config["splits"],
                seed=config["seed"],
                generate_uids=kwargs.get("generate_uids", False),
                run_spacy=run_spacy,
            )

        if task_name == "COLA":
            scorer = Scorer(
                standard_metrics=["accuracy"],
                custom_metric_funcs={matthews_corr: ["matthews_corr"]},
            )
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )

        elif task_name == "SST2":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
            )

        elif task_name == "MNLI":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=MulticlassHead(neck_dim, 3),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "SNLI":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=MulticlassHead(neck_dim, 3),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "RTE":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "WNLI":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
            )

        elif task_name == "QQP":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(
                    custom_metric_funcs={acc_f1: ["accuracy", "f1", "acc_f1"]}
                ),
            )

        elif task_name == "MRPC":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
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
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=RegressionHead(neck_dim),
                scorer=scorer,
            )

        elif task_name == "QNLI":
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=Scorer(standard_metrics=["accuracy"]),
                loss_multiplier=AUX_LOSS_MULTIPLIER,
            )

        # AUXILIARY TASKS

        elif task_name == "THIRD":
            # A toy task that predict which third of the sentence each token is in
            OUT_DIM = 3
            task = TokenClassificationTask(
                name="THIRD",
                input_module=input_module,
                head_module=BertTokenClassificationHead(neck_dim, OUT_DIM),
                loss_multiplier=AUX_LOSS_MULTIPLIER,
            )

        elif task_name == "BLEU":
            task = RegressionTask(
                name=task_name,
                input_module=input_module,
                middle_module=cls_middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=RegressionHead(neck_dim),
                output_hat_func=torch.sigmoid,
                loss_hat_func=(
                    lambda out, Y_gold: F.mse_loss(torch.sigmoid(out), Y_gold)
                ),
                scorer=Scorer(custom_metric_funcs={mse: ["mse"]}),
                loss_multiplier=AUX_LOSS_MULTIPLIER,
            )

        elif task_name == "SPACY_NER":
            OUT_DIM = len(SPACY_TAGS["SPACY_NER"])
            task = TokenClassificationTask(
                name=task_name,
                input_module=input_module,
                head_module=BertTokenClassificationHead(neck_dim, OUT_DIM),
                loss_multiplier=AUX_LOSS_MULTIPLIER,
            )

        elif task_name == "SPACY_POS":
            OUT_DIM = len(SPACY_TAGS["SPACY_POS"])
            task = TokenClassificationTask(
                name=task_name,
                input_module=input_module,
                head_module=BertTokenClassificationHead(neck_dim, OUT_DIM),
                loss_multiplier=AUX_LOSS_MULTIPLIER,
            )

        else:
            msg = (
                f"Task name {task_name} was not recognized as a primary or "
                f"auxiliary task."
            )
            raise Exception(msg)

        tasks.append(task)
        if has_payload:
            # Add any requested (and applicable) aux. label sets to existing payload
            for split, data_loader in data_loaders.items():
                payload_name = f"{task_name}_{split}"
                payload = Payload(payload_name, data_loader, [task_name], split)
                # Add auxiliary label sets if applicable

                auxiliary_task_dict = config["auxiliary_task_dict"]
                for aux_task_name, target_payloads in auxiliary_task_dict.items():
                    if aux_task_name in task_names and task_name in target_payloads:
                        aux_task_func = auxiliary_task_functions[aux_task_name]
                        payload = aux_task_func(payload)
                payloads.append(payload)

    return tasks, payloads


def get_attention_module(config, neck_dim):
    # Get attention head
    attention_config = config["attention_config"]
    if attention_config["attention_module"] is None:
        attention_module = IdentityModule()
    elif attention_config["attention_module"] == "soft":
        nonlinearity = attention_config["nonlinearity"]
        if nonlinearity == "tanh":
            nl_fun = nn.Tanh()
        elif nonlinearity == "sigmoid":
            nl_fun = nn.Sigmoid()
        else:
            raise ValueError("Unrecognized attention nonlinearity")
        attention_module = SoftAttentionModule(neck_dim, nonlinearity=nl_fun)
    else:
        raise ValueError("Unrecognized attention layer")

    return attention_module


### Code Graveyard (for code that we're just not ready to delete yet)
#
# elif config["encoder_type"] == "lstm":
#     # TODO: Allow these constants to be passed in as arguments
#     msg = (
#         "Non-BERT options are currently broken because of the BertExtractCls "
#         "hardcoded into most task heads."
#     )
#     raise NotImplementedError(msg)
#     lstm_config = config["lstm_config"]
#     neck_dim = lstm_config["hidden_size"]
#     if lstm_config["bidirectional"]:
#         neck_dim *= 2
#     lstm = LSTMModule(
#         lstm_config["emb_size"],
#         lstm_config["hidden_size"],
#         lstm_reduction="max",
#         bidirectional=lstm_config["bidirectional"],
#         lstm_num_layers=lstm_config["lstm_num_layers"],
#         encoder_class=EmbeddingsEncoder,
#         encoder_kwargs={"vocab_size": lstm_config["vocab_size"]},
#     )
#     input_module = lstm
