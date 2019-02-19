"""
Example command to run all 9 tasks: python launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint_dir ckpt --batch_size 16
"""

import argparse
import datetime
import json
import logging
import os

import numpy as np

from metal.mmtl.bert_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.trainer import MultitaskTrainer, trainer_config
from metal.utils import add_flags_from_config

logging.basicConfig(level=logging.INFO)


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


def merge_dicts(d1, d2):
    """merges d2 into a copy of d1."""
    d = d1.copy()
    for param in d:
        default = d[param]
        if isinstance(default, dict):
            d[param] = merge_dicts(default, d2)
        else:
            if param in d2.keys():
                d[param] = d2[param]
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MetalModel on single or multiple tasks.", add_help=False
    )
    parser.add_argument("--device", type=int, help="0 for gpu, -1 for cpu", default=0)
    parser.add_argument(
        "--tasks", required=True, type=str, help="Comma-sep task list e.g. QNLI,QQP"
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Which bert model to use.",
    )
    parser.add_argument(
        "--bert_output_dim", type=int, default=768, help="Bert model output dimension."
    )
    parser.add_argument(
        "--max_len", type=int, default=512, help="Maximum sequence length."
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=-1,
        help="Maximum number of examples per datasets. For debugging purposes.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--split_prop",
        type=float,
        default=None,
        help="Proportion of training data to use for validation.",
    )

    parser.add_argument(
        "--override_train_config",
        type=str,
        default=None,
        help="Whether to override train_config dict with json loaded from path. For tuning",
    )

    parser = add_flags_from_config(parser, trainer_config)
    args = parser.parse_args()

    config = merge_dicts(trainer_config, vars(args))

    # set default run_dir
    d = datetime.datetime.today()

    # Override json
    if args.override_train_config is not None:
        with open(args.override_train_config, "r") as f:
            override_config = json.loads(f.read())
        config = merge_dicts(config, override_config)

    # Update logging config
    if not args.run_name:
        run_name = args.tasks
    else:
        run_name = args.run_name

    writer_config = {
        "log_dir": f"{os.environ['METALHOME']}/logs",
        "run_dir": args.run_dir,
        "run_name": run_name,
        "include_config": True,
        "writer_metrics": [],
    }

    config["writer_config"] = writer_config
    config["writer"] = "tensorboard"

    task_names = [task_name for task_name in args.tasks.split(",")]
    tasks = create_tasks(
        task_names=task_names,
        bert_model=args.bert_model,
        split_prop=args.split_prop,
        max_len=args.max_len,
        dl_kwargs={"batch_size": args.batch_size},
        bert_output_dim=args.bert_output_dim,
        max_datapoints=args.max_datapoints,
    )

    model = MetalModel(tasks, verbose=False, device=args.device)

    # add metadata to config that will be logged to disk
    config.update(
        {
            "n_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "batch_size": args.batch_size,
            "max_seq_len": args.max_len,
        }
    )

    trainer = MultitaskTrainer()
    trainer.train_model(model, tasks, **config)

    # compute and save final scores
    test_scores = {}
    for task in tasks:
        scores = task.scorer.score(model, task)
        test_scores[task.name] = scores

    print(test_scores)
    if trainer.writer:
        save_dir = trainer.writer.log_subdir
        save_path = os.path.join(save_dir, "metrics.json")
        with open(save_path, "w") as f:
            json.dump(test_scores, f)
        print(f"Saved metrics to {save_path}")
