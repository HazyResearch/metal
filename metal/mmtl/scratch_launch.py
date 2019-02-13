"""
Example command to run all 9 tasks: python launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint-dir ckpt --batch-size 16
"""

import argparse
import datetime
import json
import os

import numpy as np

from metal.mmtl.BERT_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.scorer import Scorer
from metal.mmtl.trainer import MultitaskTrainer

parser = argparse.ArgumentParser(
    description="Train MetalModel on single or multiple tasks."
)

parser.add_argument("--device", type=int, help="0 for gpu, -1 for cpu", default=0)
parser.add_argument(
    "--tasks", required=True, type=str, help="Comma-sep task list e.g. QNLI,QQP"
)
parser.add_argument(
    "--bert-model",
    type=str,
    default="bert-base-uncased",
    help="Which bert model to use.",
)
parser.add_argument(
    "--bert-output-dim", type=int, default=768, help="Bert model output dimension."
)
parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length.")
parser.add_argument(
    "--max-datapoints",
    type=int,
    default=-1,
    help="Maximum number of examples per datasets. For debugging purposes.",
)
parser.add_argument(
    "--batch-size", type=int, default=16, help="Batch size for training."
)
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
parser.add_argument(
    "--lr-freeze", type=int, default=1, help="Number of epochs to freeze lr for."
)
parser.add_argument("--l2", type=float, default=0.01, help="Weight decay.")
parser.add_argument(
    "--n-epochs", type=int, default=5, help="Number of epochs to train for."
)
parser.add_argument(
    "--lr-scheduler", type=str, default="exponential", help="Learning rate scheduler."
)
parser.add_argument("--log-every", type=float, default=0.25, help="Log frequency.")
parser.add_argument("--score-every", type=float, default=0.5, help="Scoring frequency.")
parser.add_argument(
    "--checkpoint-dir",
    required=True,
    type=str,
    help="Where to save the best model and logs.",
)
parser.add_argument(
    "--checkpoint-metric",
    type=str,
    help="Which metric to use to checkpoint best model.",
    default="train/loss",
)
parser.add_argument(
    "--checkpoint-metric-mode",
    type=str,
    default="min",  # assuming default metric is loss
    help="Whether to save max or min.",
)
parser.add_argument(
    "--override-train-config",
    type=str,
    default=None,
    help="Whether to override train_config dict with json loaded from path. For tuning",
)


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
    args = parser.parse_args()
    d = datetime.datetime.today()

    run_dir = os.path.join(
        os.path.join(args.checkpoint_dir, f"{d.day}-{d.month}-{d.year}/{args.tasks}/")
    )
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    run_name = get_dir_name(run_dir)
    trainer_config = {
        "verbose": True,
        "progress_bar": True,
        # "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True}, ## TODO?
        "n_epochs": args.n_epochs,
        # 'grad_clip': 1.0,  ## TODO?
        "l2": args.l2,
        "optimizer_config": {
            "optimizer": "adam",
            "optimizer_common": {"lr": args.lr},
            "adam_config": {"betas": (0.9, 0.999)},
        },
        "lr_scheduler": "exponential",  # reduce_on_plateau  ## TODO? Warmup
        "lr_scheduler_config": {
            "lr_freeze": args.lr_freeze,
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
            "log_every": args.log_every,
            "score_every": args.score_every,
        },  # Checkpointer (see metal/logging/checkpointer.py for descriptions)
        "checkpoint": True,  # If True, checkpoint models when certain conditions are met
        "checkpoint_config": {
            "checkpoint_every": 0,  # Save a model checkpoint every this many log_units
            "checkpoint_best": True,
            # "checkpoint_final": False,  # Save a model checkpoint at the end of training
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_mode": args.checkpoint_metric_mode,
            "checkpoint_dir": os.path.join(run_dir, run_name),
            "checkpoint_runway": 0,
        },
    }

    # Override json
    if args.override_train_config is not None:
        with open(args.override_train_config, "r") as f:
            trainer_config = json.loads(f.read())

    tasks = []
    task_names = [task_name for task_name in args.tasks.split(",")]
    tasks = create_tasks(
        task_names=task_names,
        bert_model=args.bert_model,
        split_prop=0.8,
        max_len=args.max_len,
        dl_kwargs={"batch_size": args.batch_size},
        bert_output_dim=args.bert_output_dim,
        max_datapoints=args.max_datapoints,
    )

    model = MetalModel(tasks, verbose=False, device=args.device)
    trainer = MultitaskTrainer()
    trainer.train_model(model, tasks, **trainer_config)
    for task in tasks:
        # TODO: replace with split="test" when we support this
        scores = task.scorer.score(
            model, task, target_metrics=[f"{task.name}/test/accuracy"]
        )
        print(scores)
    print(os.path.join(run_dir, run_name))
