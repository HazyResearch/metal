"""
Sample call: python metal/mmtl/aws/mmtl_aws.py --mode run --aws_access_key_id xxx --aws_secret_access_key xxx --keypath ~/personalkeyncalifornia.pem
Sample output:
...
Putting file output/configspace/config_1.json -> config
Putting file output/configspace/config_0.json -> config
Getting file config -> output/1/config
Getting dir metal/checkpoint/ -> output/1/checkpointdir
Putting file output/configspace/config_2.json -> config
Getting file config -> output/0/config
Getting dir metal/checkpoint/ -> output/0/checkpointdir
Getting file config -> output/2/config
Getting dir metal/checkpoint/ -> output/2/checkpointdir
Results

(venv-mmtl) maxlam@dawn6:/lfs/1/maxlam/metal$ ls output/
0  0.out  1  1.out  2  2.out  configspace
(venv-mmtl) maxlam@dawn6:/lfs/1/maxlam/metal$ ls output/0
checkpointdir  config  stderr  stdout
(venv-mmtl) maxlam@dawn6:/lfs/1/maxlam/metal$ tail output/0/stdout
Requirement already satisfied: pycparser in /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from cffi>=1.1->bcrypt>=3.1.3->paramiko->-r metal/mmtl/requirements-mmtl.txt (line 15)) (2.18)
Requirement already satisfied: webencodings in /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages (from html5lib!=1.0b1,!=1.0b2,!=1.0b3,!=1.0b4,!=1.0b5,!=1.0b6,!=1.0b7,!=1.0b8,>=0.99999999pre->bleach->nbconvert->jupyter->-r metal/mmtl/requirements-mmtl.txt (line 10)) (0.5.1)
/home/ubuntu/metal
Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.
Loading QNLI Dataset
Could not find kwarg "device" in destination dict.
Could not find kwarg "lr_freeze" in destination dict.
Beginning train loop.
Expecting a total of _approximately_ 3 examples and 3 batches per epoch from 1 tasks.
[1.0 epo]: TRAIN:[loss=74.170]
(venv-mmtl) maxlam@dawn6:/lfs/1/maxlam/metal$
"""
import argparse
import copy
import datetime
import json
import os
import random

import numpy as np

trainer_config_space = {
    "verbose": True,
    "progress_bar": True,
    # "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True}, ## TODO?
    "n_epochs": 1,
    # 'grad_clip': 1.0,  ## TODO?
    "l2": {"is_hyperparam": True, "range": [0, 1000], "scale": "linear"},
    "optimizer_config": {
        "optimizer": "adam",
        "optimizer_common": {
            "lr": {"is_hyperparam": True, "range": [1e-5, 1], "scale": "log"}
        },
        "adam_config": {
            "betas": (
                # 0.9,
                {"is_hyperparam": True, "range": [0, 0.9], "scale": "linear"},
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
    COMMAND_PREFIX = (
        "pkill -9 python;"  # Kill all python processes
        "source activate pytorch_p36;"
        "export GLUEDATA=/home/ubuntu/glue/;"  # Assumes ami has this here
        "rm -rf metal;"
        "git clone -b mmtl https://github.com/HazyResearch/metal.git;"
        "cd metal; source add_to_path.sh; pip install -r metal/mmtl/requirements-mmtl.txt;"
        "pwd;"
    )
    # COMMAND = "python metal/mmtl/launch.py --tasks QNLI --n_epochs 2 --log_every 0.25 --score_every 0.25 --max_len 256 --batch_size 8 --checkpoint_dir ./checkpoint --checkpoint_metric QNLI/valid/accuracy --checkpoint_metric_mode max --max_datapoints 32 --override_train_config ../config"
    COMMAND = "python metal/mmtl/launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint_dir ./checkpoint --batch_size 16 --n_epochs 3 --max_datapoints 256"
    return {
        "cmd": COMMAND_PREFIX + COMMAND,
        "files_to_put": [(config_path, "config")],
        "files_to_get": [("config", "config")],
        "dirs_to_get": [("metal/checkpoint/", "checkpointdir")],
    }


def generate_configs_and_commands(args, n=2):
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
