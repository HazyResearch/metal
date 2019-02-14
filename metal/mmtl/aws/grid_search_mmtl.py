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
import time

import numpy as np

from metal.tuners.random_tuner import RandomSearchTuner
from metal.tuners.tuner import ModelTuner

train_search_space = {
    "verbose": True,
    "progress_bar": False,
    # hyperparams
    "l2": {"range": [1e-5, 1], "scale": "log"},
    "lr": {"range": [1e-5, 1], "scale": "log"},
    # Tensorboard
    "writer": "tensorboard",
    "log_dir": f"tensorboard_logs",
}


def create_command_dict(config_path):
    COMMAND_PREFIX = (
        "pkill -9 tensorboard;"  # Kill pre-existing tensorboard
        "pkill -9 python;"  # Kill all python processes
        "source activate pytorch_p36;"
        "export GLUEDATA=/home/ubuntu/glue/;"  # Assumes ami has this here
        "rm -rf metal;"
        "git clone -b mmtl https://github.com/HazyResearch/metal.git;"
        "cd metal; source add_to_path.sh; pip install -r metal/mmtl/requirements-mmtl.txt;"
        "mkdir logs;"
        " ( tensorboard --logdir logs & ) ;"
    )
    # COMMAND = "python metal/mmtl/launch.py --tasks QNLI --n_epochs 2 --log_every 0.25 --score_every 0.25 --max_len 256 --batch_size 8 --checkpoint_dir ./checkpoint --checkpoint_metric QNLI/valid/accuracy --checkpoint_metric_mode max --max_datapoints 32 --override_train_config ../config"
    COMMAND = " ( python metal/mmtl/launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint_dir ./checkpoint --batch_size 4 --n_epochs 3 --max_datapoints 32 --override_train_config ../config  2>&1 | tee output ) "
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

    tuner = RandomSearchTuner(None, seed=time.time())
    configs = tuner.config_generator(train_search_space, n, tuner.rng, True)

    command_dicts = []
    for i, random_config in enumerate(configs):

        # Write to directory
        config_path = "%s/config_%d.json" % (configspace_path, i)
        with open(config_path, "w") as f:
            json.dump(random_config, f)

        # Create command dict
        command_dicts.append(create_command_dict(config_path))

    return command_dicts
