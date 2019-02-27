"""
Example command to run all 9 tasks: python launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint_dir ckpt --batch_size 16
"""

import argparse
import datetime
import json
import logging
import os

import numpy as np

from metal.mmtl.glue_tasks import create_tasks, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

logging.basicConfig(level=logging.INFO)

# Auxiliary task dict -- global for now!
AUXILIARY_TASKS = {
    "STSB": ["BLEU"],
    "MRPC": ["BLEU"],
    "MRPC_SAN": ["BLEU"],
    "QQP": ["BLEU"],
    "QQP_SAN": ["BLEU"],
}


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    existing_dirs = np.array(
        [
            d
            for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]
    ).astype(np.int)
    if len(existing_dirs) > 0:
        return str(existing_dirs.max() + 1)
    else:
        return "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MetalModel on single or multiple tasks.", add_help=False
    )

    # Model arguments
    # TODO: parse these automatically from model dict
    parser.add_argument(
        "--tasks", required=True, type=str, help="Comma-sep task list e.g. QNLI,QQP"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default=None,
        help="Pretrained model for weight initialization",
    )

    # Training arguments
    parser.add_argument(
        "--override_train_config",
        type=str,
        default=None,
        help=(
            "Whether to override train_config dict with json loaded from path. "
            "This is used, e.g., for tuning."
        ),
    )

    # Override dl_kwargs for specific task
    # NOTE: (VC) we currently apply the same data loader kwargs to all tasks
    # This allows us to override that general config for a specific task.
    # e.g. --override_task_dl STSB.batch_size.2,SST2.batch_size.16
    parser.add_argument(
        "--override_task_dl",
        type=str,
        default=None,
        help="Task-specific config for overriding general dl_kwargs",
    )

    # Use auxiliary tasks
    parser.add_argument(
        "--use_auxiliary", type=int, default=False, help="Use auxiliary tasks or not"
    )

    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    args = parser.parse_args()

    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")

    # Override config with config stored in json if specified
    if args.override_train_config is not None:
        with open(args.override_train_config, "r") as f:
            override_config = json.loads(f.read())
        config = recursive_merge_dicts(trainer_config, override_config, misses="report")

    # Set intelligent writer_config settings
    trainer_config["writer"] = "tensorboard"  # Always store tensorboard logs

    # Set splits based on split_prop
    if task_config["split_prop"]:
        # Create a valid set from train set
        task_config["split_prop"] = float(task_config["split_prop"])
        # Sampler will be used and handles shuffle automatically
        task_config["dl_kwargs"]["shuffle"] = False
        task_config["splits"] = ["train", "test"]
    else:
        task_config["splits"] = ["train", "valid", "test"]

    # Creating auxiliarty task dict
    if args.use_auxiliary:
        auxiliary_tasks = AUXILIARY_TASKS
    else:
        auxiliary_tasks = {}

    # Overrwrite dl_kwargs for specific tasks
    if args.override_task_dl:
        task_configs_str = [
            tuple(config.split(".")) for config in args.override_task_dl.split(",")
        ]
        task_dl_kwargs = {}
        for config in task_configs_str:
            task_name, kwarg_key, kwarg_val = config
            if kwarg_key == "batch_size":
                kwarg_val = int(kwarg_val)
            task_dl_kwargs[task_name] = {kwarg_key: kwarg_val}

        task_config["task_dl_kwargs"] = task_dl_kwargs

    # Getting primary task names
    task_names = [task_name for task_name in args.tasks.split(",")]

    # Getting tasks, primary and auxiliary
    tasks = create_tasks(task_names, auxiliary_tasks=auxiliary_tasks, **task_config)

    # Updating with auxiliary tasks!
    task_names_with_aux = [task.name for task in tasks]
    print("Training on tasks:")
    print(task_names_with_aux)

    # Updating run_name here to include auxiliary tasks
    if not trainer_config["writer_config"]["run_name"]:
        trainer_config["writer_config"]["run_name"] = ".".join(task_names_with_aux)

    model_config["verbose"] = False
    model = MetalModel(tasks, **model_config)

    if args.model_weights:
        model.load_weights(args.model_weights)

    # add metadata to trainer_config that will be logged to disk
    trainer_config["n_paramaters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    trainer = MultitaskTrainer(**trainer_config)
    # Force early instantiation of writer to write all three configs to dict
    trainer._set_writer()
    # trainer_config will get written automatically right before training
    trainer.writer.write_config(model_config, "model_config")
    trainer.writer.write_config(task_config, "task_config")

    trainer.train_model(model, tasks)
