import copy
import os
import warnings

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from metal.mmtl.auxiliary_tasks import SPACY_TAGS, auxiliary_task_functions
#from metal.mmtl.chexnet.chexnet_metrics import acc_f1, matthews_corr, mse, pearson_spearman
from metal.mmtl.cxr.cxr_slices import create_slice_labels
from metal.mmtl.modules import (
    BinaryHead,
    RegressionHead,
    SoftAttentionModule,
)
from metal.mmtl.cxr.modules import TorchVisionEncoder
from metal.mmtl.payload import Payload
from metal.mmtl.scorer import Scorer
from metal.mmtl.slicing import create_slice_task
from metal.mmtl.task import ClassificationTask, RegressionTask
from metal.utils import recursive_merge_dicts, set_seed


task_defaults = {
    # General
    "pool_payload_tasks":False, # Pools same task for different payloads if True
    "split_prop": None,
    "splits": ["train", "valid", "test"],
    "subsample": -1,
    "findings":"ALL",
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
    # CNN
    "encoder_type": "DenseNet121",
    "cnn_kwargs": {
        "freeze_cnn": False,
        "pretrained": True,
        "drop_rate": 0.2,
    },
    "attention_config": {
        "attention_module": None,  # None, soft currently accepted
        "nonlinearity": "tanh",  # tanh, sigmoid currently accepted
    },
    # Auxiliary Tasks and Primary Tasks
    "auxiliary_task_dict": {  # A map of each aux. task to the primary task it applies to
        "IGNORE_DRAINS": ["PNEUMOTHORAX"],
    },
    "auxiliary_loss_multiplier": 1.0,
    "tasks": None,  # Comma-sep task list e.g. QNLI,QQP
    # Slicing
    "slice_dict": None,  # A map of the slices that apply to each task
}


def create_tasks_and_payloads(full_task_names, **kwargs):
    assert len(task_names) > 0

    config = recursive_merge_dicts(task_defaults, kwargs)

    if config["seed"] is None:
        config["seed"] = np.random.randint(1e6)
        print(f"Using random seed: {config['seed']}")
    set_seed(config["seed"])

    # share cnn encoder for all tasks
    cnn_kwargs = config["cnn_kwargs"]
    cnn_model = TorchVisionEncoder(config["cnn_model"], **cnn_kwargs)
    neck_dim = cnn_module.encode_dim
    input_module = cnn_model
    middle_module = None # None for now

    # Create dict override dl_kwarg for specific task
    # e.g. {"STSB": {"batch_size": 2}}
    # TODO: CHECK THIS, MAY BE BROKEN FOR CXR
    task_dl_kwargs = {}
    if config["task_dl_kwargs"]:
        task_configs_str = [
            tuple(config.split(".")) for config in config["task_dl_kwargs"].split(",")
        ]
        for (task_name, kwarg_key, kwarg_val) in task_configs_str:
            if kwarg_key == "batch_size":
                kwarg_val = int(kwarg_val)
            task_dl_kwargs[task_name] = {kwarg_key: kwarg_val}

    #payload_names = list(set([t.split(":")[0] for t in full_task_names]))
    #if config["pool_payload_tasks"]:
    #    task_names = list(set([t.split(":")[1] for t in full_task_names]))
    #else:
    #    task_names = full_task_names     
    tasks = []
    payloads = []

    # NEW STRUCTURE
    
    # Get payload:primary task dict
    task_payload_dict = defaultdict(list)
    for full_task_name in full_task_names:
        payload_name = full_task_name.split(":")[0]
        task_name = full_task_name.split(":")[1]
        task_payload_dict[payload_name].append(task_name)

    # If "ALL" supplied as task, only one payload per dataset;
    # Else, create separate payloads for each task-dataset combo 
    for k,v in task_payload_dict.items():
        if "ALL" in v and len(v)>1:
            raise ValueError("Cannot have 'ALL' task with other primary tasks")

    # Getting auxiliary task dict
    auxiliary_task_dict = config["auxiliary_task_dict"]

    # Get primary task:payload dict
    payload_task_dict = invert_dict(task_payload_dict)

    for full_task_name in full_task_names:
        # Pull out names of auxiliary tasks to be dealt with in a second step
        # TODO: fix this logic for cases where auxiliary task for task_name has
        # its own payload
        # Right now, supply tasks as DATASET:TASK, by default if all tasks
        # in a dataset are included, it is the same as training on entire dataset
        if config["pool_payload_tasks"]:
            payload_name = full_task_name.split(":")[0]
            task_name = full_task_name.split(":")[1]
            new_payload = payload_name not in [p.name for p in payloads]
        else:
            task_name = full_task_name
            new_payload = full_task_name not in [p.name for p in payloads]

        # Override general dl kwargs with payload-specific kwargs
        dl_kwargs = copy.deepcopy(config["dl_kwargs"])
        if task_name in task_dl_kwargs:
            dl_kwargs.update(task_dl_kwargs[task_name])

        # Each primary dataset has data_loaders to load
        if new_payload:
            datasets = create_cxr_datasets(
                dataset_name=dataset_name,
                splits=config["splits"],
                subsample=config["subsample"],
                pooled=config["pool_payload_tasks"],
                findings=task_name,
                verbose=True,
                )

            # Wrap datasets with DataLoader objects
            data_loaders = create_cxr_dataloaders(
                datasets,
                dl_kwargs=dl_kwargs,
                split_prop=config["split_prop"],
                splits=config["splits"],
                seed=config["seed"],
            )

        # TODO: PUT IN OPTION TO POOL SAME TASK FOR DIFF SETS HERE?
        if task_name not in NON_STANDARD_TASKS:   
            scorer = Scorer(
                standard_metrics=["accuracy"],
            )
            task = ClassificationTask(
                name=full_task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )    

        elif "PNEUMOTHORAX" in task_name:
            scorer = Scorer(
            standard_metrics=["accuracy"],
            )
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )
        
        # AUXILIARY TASKS

        # NONE YET -- MAYBE AUTOENCODING?

        else:
            msg = (
                f"Task name {task_name} was not recognized as a primary or "
                f"auxiliary task."
            )
            raise Exception(msg)

        tasks.append(task)
        if new_payload:
            # Create payloads (and add slices/auxiliary tasks as applicable)
            for split, data_loader in data_loaders.items():
                payload_name = f"{task_name}_{split}"
                payload = Payload(payload_name, data_loader, [task_name], split)

                # Add auxiliary label sets if applicable
                auxiliary_task_dict = config["auxiliary_task_dict"]
                for aux_task_name, target_payloads in auxiliary_task_dict.items():
                    if aux_task_name in task_names and task_name in target_payloads:
                        aux_task_func = auxiliary_task_functions[aux_task_name]
                        payload = aux_task_func(payload)

                # Add slice task and label sets if applicable
                slice_names = (
                    config["slice_dict"].get(task_name, [])
                    if config["slice_dict"]
                    else []
                )

                if slice_names:
                    dataset = payload.data_loader.dataset
                    for slice_name in slice_names:
                        slice_task_name = f"{task_name}:{slice_name}"
                        slice_task = create_slice_task(task, slice_task_name)
                        tasks.append(slice_task)

                        slice_labels = create_slice_labels(
                            dataset, base_task_name=task_name, slice_name=slice_name
                        )
                        payload.add_label_set(slice_task_name, slice_labels)

                payloads.append(payload)

    return tasks, payloads






    # Initialize payloads with data for any primary tasks they have
    # If pooling option is True, use standard task names across
    # datasets and pool them, otherwise use payload-specific name

    for payload_name in task_payload_dict.keys():
        datasets = create_cxr_datasets(
            dataset_name=payload_name,
            splits=config["splits"],
            subsample=config["subsample"],
            tasks=task_payload_dict[payload_name],
            pooled=config["pool_payload_tasks"],
            verbose=True,
            )

            # Wrap datasets with DataLoader objects
        data_loaders = create_cxr_dataloaders(
            datasets,
            dl_kwargs=dl_kwargs,
            split_prop=config["split_prop"],
            splits=config["splits"],
            seed=config["seed"],
            )

        # Create payloads with primary task names
        for split, data_loader in data_loaders.items():
            payload_name = f"{task_name}_{split}"
            payload = Payload(payload_name, data_loader, 
                task_payload_dict[payload_name], split)
            payloads.append(payload)

    # Getting primary task names
    if config["pool_payload_tasks"]:
        task_names = list(set([t.split(":")[1] for in full_task_names]))
    else:
        task_names = full_task_names
    
    # Adding primary tasks
    for task_name in task_names:
        # Pull out names of auxiliary tasks to be dealt with in a second step
        # TODO: fix this logic for cases where auxiliary task for task_name has
        # its own payload
        # Right now, supply tasks as DATASET:TASK, by default if all tasks
        
        # Override general dl kwargs with payload-specific kwargs
        dl_kwargs = copy.deepcopy(config["dl_kwargs"])
        if task_name in task_dl_kwargs:
            dl_kwargs.update(task_dl_kwargs[task_name])

        if "PNEUMOTHORAX" in task_name:
            scorer = Scorer(
            standard_metrics=["accuracy"],
            )
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )
        else:   
            scorer = Scorer(
                standard_metrics=["accuracy"],
            )
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )    

        # AUXILIARY TASKS

        # NONE YET -- MAYBE AUTOENCODING?

        #else:
        #    msg = (
        #        f"Task name {task_name} was not recognized as a primary or "
        #        f"auxiliary task."
        #    )
        #    raise Exception(msg)

        tasks.append(task)
        
        # Add auxiliary label sets if applicable
        
        if [target_task in auxiliary_task_dict[ 
        for aux_task_name, target_task in auxiliary_task_dict.items():
            if aux_task_name in task_names and task_name in target_payloads:
                aux_task_func = auxiliary_task_functions[aux_task_name]
                payload = aux_task_func(payload)
            
        # Add slice task and label sets if applicable
        slice_names = (
            config["slice_dict"].get(task_name, [])
            if config["slice_dict"]
            else []
        )
            
        if slice_names:
            dataset = payload.data_loader.dataset
            for slice_name in slice_names:
                slice_task_name = f"{task_name}:{slice_name}"
                slice_task = create_slice_task(task, slice_task_name)
                tasks.append(slice_task)
              
                slice_labels = create_slice_labels(
                    dataset, base_task_name=task_name, slice_name=slice_name
                )
                payload.add_label_set(slice_task_name, slice_labels)

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

def create_cxr_datasets(
    dataset_name,
    splits,
    subsample=-1,
    verbose=True,
):
    if verbose:
        print(f"Loading {dataset_name} Dataset")

    datasets = {}
    for split_name in splits:
        # Codebase uses valid but files are saved as dev.tsv
        if split_name == "valid":
            split = "dev"
        else:
            split = split_name
        datasets[split_name] = get_cxr_dataset(
            dataset_name,
            split,
            subsample=subsample,
            findings=findings
        )
    return datasets


def create_cxr_dataloaders(datasets, dl_kwargs, split_prop, splits, seed=123):
    """ Initializes train/dev/test dataloaders given dataset_class"""
    dataloaders = {}

    # When split_prop is not None, we use create an artificial dev set from the train set
    if split_prop and "train" in splits:
        dataloaders["train"], dataloaders["valid"] = datasets["train"].get_dataloader(
            split_prop=split_prop, split_seed=seed, **dl_kwargs
        )

        # Use the dev set as test set if available.
        if "valid" in datasets:
            dataloaders["test"] = datasets["valid"].get_dataloader(**dl_kwargs)

    # When split_prop is None, we use standard train/dev/test splits.
    else:
        for split_name in datasets:
            dataloaders[split_name] = datasets[split_name].get_dataloader(**dl_kwargs)
    return dataloaders

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key] 
            else: 
                inverse[item].append(key) 
    return inverse
