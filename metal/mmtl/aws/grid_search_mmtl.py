import argparse
import copy
import datetime
import json
import os
import random

import numpy as np

from metal.mmtl.BERT_tasks import create_task
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.trainer import MultitaskTrainer

trainer_config_space = {
    "verbose": True,
    "progress_bar": True,
    # "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True}, ## TODO?
    "n_epochs": 10,
    # 'grad_clip': 1.0,  ## TODO?
    "l2": 0.1,
    "optimizer_config": {
        "optimizer": "adam",
        "optimizer_common": {
            "lr": {"is_hyperparam": True, "range": [1e-5, 1], "scale": "log"}
        },
        "adam_config": {
            "betas": (
                # 0.9,
                {"is_hyperparam": True, "range": [0.5, 0.9], "scale": "linear"},
                0.999,
            )
        },
    },
    "lr_scheduler": "exponential",  # reduce_on_plateau  ## TODO? Warmup
    "lr_scheduler_config": {
        "lr_freeze": 1,
        # Scheduler - exponential
        "exponential_config": {"gamma": 0.9},  # decay rate
        # Scheduler - reduce_on_plateau
        "plateau_config": {
            "factor": 0.5,
            "patience": 10,
            "threshold": 0.0001,
            "min_lr": 1e-4,
        },
    },
    # Logger (see metal/logging/logger.py for descriptions)
    "logger": True,
    "logger_config": {
        "log_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
        "log_every": 1,
        "score_every": 1,
    },  # Checkpointer (see metal/logging/checkpointer.py for descriptions)
    "checkpoint": True,  # If True, checkpoint models when certain conditions are met
    "checkpoint_config": {
        "checkpoint_every": 0,  # Save a model checkpoint every this many log_units
        "checkpoint_best": True,
        # "checkpoint_final": False,  # Save a model checkpoint at the end of training
        "checkpoint_metric": "QNLI/valid/accuracy",
        "checkpoint_metric_mode": "max",
        "checkpoint_dir": "test_checkpoint",
        "checkpoint_runway": 0,
    },
}

rng = random.Random(0)


def sample_hyperparam(d):
    def range_param_func(v):
        scale = v.get("scale", "linear")
        mini = min(v["range"])
        maxi = max(v["range"])
        if scale == "linear":
            func = lambda rand: mini + (maxi - mini) * rand
        elif scale == "log":
            mini = np.log(mini)
            maxi = np.log(maxi)
            func = lambda rand: np.exp(mini + (maxi - mini) * rand)
        else:
            raise ValueError(f"Unrecognized scale '{scale}' for " "parameter {k}")
        return func

    param_range_func = range_param_func(d)
    return float(param_range_func(rng.random()))


def is_dict_hyperparam(d):
    if "is_hyperparam" not in d.keys():
        return False
    return d["is_hyperparam"]


def sample_random_config(config_space):
    copied_config = copy.deepcopy(config_space)
    for k, v in config_space.items():
        if type(v) == dict:
            if is_dict_hyperparam(v):
                copied_config[k] = sample_hyperparam(v)
            else:
                copied_config[k] = sample_random_config(v)
        if type(v) == list or type(v) == tuple:
            to_replace = []
            for element in v:
                if type(element) == dict and is_dict_hyperparam(element):
                    to_replace.append(sample_hyperparam(element))
                else:
                    to_replace.append(element)

            copied_config[k] = to_replace
    return copied_config


def create_command_dict(config_path):
    return {
        "cmd": "cat test",
        "files_to_put": [(config_path, "test")],
        "files_to_get": [],
    }


def generate_configs_and_commands(args, n=10):
    configspace_path = "%s/configspace" % args.outputpath
    if not os.path.exists(configspace_path):
        os.makedirs(configspace_path)

    command_dicts = []
    for i in range(n):

        # Gen random config
        random_config = sample_random_config(trainer_config_space)

        # Write to directory
        config_path = "%s/config_%d.json" % (configspace_path, i)
        with open(config_path, "w") as f:
            json.dump(random_config, f)

        # Create command dict
        command_dicts.append(create_command_dict(config_path))

    return command_dicts
