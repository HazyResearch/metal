import os

import numpy as np
import torch
import torch.optim as optim

from metal.logging import Checkpointer, Logger, LogWriter, TensorBoardWriter
from metal.utils import place_on_gpu, recursive_merge_dicts

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    # Only use tqdm notebook if not in travis testing
    if "CI" not in os.environ:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

TRAIN = 0
VALID = 1
TEST = 2


trainer_config = {
    "verbose": True,
    # Device
    "device": "cpu",
    # Loss function config
    "loss_fn_reduction": "mean",
    # Display
    "progress_bar": False,
    # Dataloader
    "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True},
    # Loss weights
    "loss_weights": None,
    # Train Loop
    "n_epochs": 10,
    # 'grad_clip': 0.0,
    "l2": 0.0,
    "validation_metric": "accuracy",
    "validation_freq": 1,
    "validation_scoring_kwargs": {},
    # Evaluate dev for during training every this many epochs
    # Optimizer
    "optimizer_config": {
        "optimizer": "adam",
        "optimizer_common": {"lr": 0.01},
        # Optimizer - SGD
        "sgd_config": {"momentum": 0.9},
        # Optimizer - Adam
        "adam_config": {"betas": (0.9, 0.999)},
        # Optimizer - RMSProp
        "rmsprop_config": {},  # Use defaults
    },
    # LR Scheduler (for learning rate)
    "lr_scheduler": "reduce_on_plateau",
    # [None, 'exponential', 'reduce_on_plateau']
    # 'reduce_on_plateau' uses checkpoint_metric to assess plateaus
    "lr_scheduler_config": {
        # Freeze learning rate initially this many epochs
        "lr_freeze": 0,
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
        "log_train_every": 1,  # How often train metrics are calculated (optionally logged to TB)
        "log_train_metrics": [
            "loss"
        ],  # Metrics to calculate and report every `log_train_every` units. This can include built-in and user-defined metrics.
        "log_train_metrics_func": None,  # A function or list of functions that map a model + train_loader to a dictionary of custom metrics
        "log_valid_every": 1,  # How frequently to evaluate on valid set (must be multiple of log_freq)
        "log_valid_metrics": [
            "accuracy"
        ],  # Metrics to calculate and report every `log_valid_every` units; this can include built-in and user-defined metrics
        "log_valid_metrics_func": None,  # A function or list of functions that maps a model + valid_loader to a dictionary of custom metrics
    },
    # LogWriter/Tensorboard (see metal/logging/writer.py for descriptions)
    "writer": None,  # [None, "json", "tensorboard"]
    "writer_config": {  # Log (or event) file stored at log_dir/run_dir/run_name
        "log_dir": None,
        "run_dir": None,
        "run_name": None,
        "writer_metrics": None,  # May specify a subset of metrics in metrics_dict to be written
        "include_config": True,  # If True, include model config in log
    },
    # Checkpointer (see metal/logging/checkpointer.py for descriptions)
    "checkpoint": True,  # If True, checkpoint models when certain conditions are met
    "checkpoint_config": {
        "checkpoint_best": True,
        "checkpoint_every": None,  # uses log_valid_unit for units; if not None, checkpoint this often regardless of performance
        "checkpoint_metric": "accuracy",  # Must be in metrics dict; assumes valid split unless appended with "train/"
        "checkpoint_metric_mode": "max",  # ['max', 'min']
        "checkpoint_dir": "checkpoints",
        "checkpoint_runway": 0,
    },
}


class MultitaskTrainer(object):
    """Driver for the MTL training process"""

    def __init__(self, config={}):
        self.config = recursive_merge_dicts(trainer_config, config)

    def train_model(self, model, tasks, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)

        # Move model to GPU
        if self.config["verbose"] and model.config["device"] != "cpu":
            print("Using GPU...")
        model.to(model.config["device"])

        # Set training components
        self._set_writer()
        self._set_logger()
        self._set_checkpointer()
        self._set_optimizer(model)
        # TODO: Accept training matrix to give more fine-tuned training commands
        self._set_scheduler()

        # TODO: Restore the ability to resume training from a given epoch

        # Train the model
        # TODO: Allow other ways to train besides 1 epoch of all datasets
        model.train()
        # metrics_dict = {}
        for epoch in range(self.config["n_epochs"]):

            # progress_bar = (
            #     self.config["progress_bar"]
            #     and self.config["verbose"]
            #     and self.logger.log_unit == "epochs"
            # )

            # t = tqdm(
            #     enumerate(self._get_train_batches(tasks, approx_batch_counts)),
            #     total=total_batches,
            #     disable=(not progress_bar),
            # )

            self.running_loss = 0.0
            self.running_examples = 0
            for batch_num, (task_name, batch) in enumerate(
                self._get_train_batches(tasks)
            ):
                # NOTE: actual batch_size may not equal config's target batch_size
                # batch_size = len(batch[0])

                # Moving data to device
                if self.config["device"] != "cpu":
                    batch = place_on_gpu(batch)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to calculate the average loss per example
                loss = model(batch, task_name)
                if torch.isnan(loss):
                    msg = "Loss is NaN. Consider reducing learning rate."
                    raise Exception(msg)

                # Backward pass to calculate gradients
                # Loss is an average loss per example
                loss.backward()

                # Perform optimizer step
                self.optimizer.step()

                # Calculate metrics, log, and checkpoint as necessary
                # metrics_dict = self._execute_logging(
                #     train_loader, valid_loader, loss, batch_size
                # )
                # metrics_hist.update(metrics_dict)

                # tqdm output
                # t.set_postfix(loss=metrics_dict["train/loss"])

            # Apply learning rate scheduler
            # self._update_scheduler(epoch, metrics_hist)

        # self.eval()

        # # Restore best model if applicable
        # if self.checkpointer:
        #     self.checkpointer.load_best_model(model=self)

        # # Write log if applicable
        # if self.writer:
        #     if self.writer.include_config:
        #         self.writer.add_config(self.config)
        #     self.writer.close()

        # # Print confusion matrix if applicable
        # if self.config["verbose"]:
        #     print("Finished Training")
        #     if valid_loader is not None:
        #         self.score(
        #             valid_loader,
        #             metric=self.config["validation_metric"],
        #             verbose=True,
        #             print_confusion_matrix=True,
        #         )

    def _get_train_batches(self, tasks):
        """Yields batches one at a time sampled from tasks with some strategy"""
        # TODO: Allow more involved strategies for sampling from tasks
        # For now, just use proportional sampling
        # Length of a dataloader is the number of batches it contains
        approx_batch_counts = [len(t.data_loaders[TRAIN]) for t in tasks]
        total_batches = sum(approx_batch_counts)
        batch_distribution = np.array(approx_batch_counts) / total_batches
        train_loaders = [iter(t.data_loaders[TRAIN]) for t in tasks]
        # NOTE: actual number of batches per task may not be exact, since we sample
        # instead of enumerating, but it should be close.
        for task_idx in np.random.choice(
            len(tasks), total_batches, replace=True, p=batch_distribution
        ):
            # HACK: this is still not ideal (since near the end we may need to sample
            # multiple times before finding the task that hasn't run out yet)
            try:
                yield (tasks[task_idx].name, next(train_loaders[task_idx]))
            except StopIteration:
                continue

    def _set_writer(self):
        if self.config["writer"] is None:
            self.writer = None
        elif self.config["writer"] == "json":
            self.writer = LogWriter(**(self.config["writer_config"]))
        elif self.config["writer"] == "tensorboard":
            self.writer = TensorBoardWriter(**(self.config["writer_config"]))
        else:
            raise Exception(f"Unrecognized writer: {self.config['writer']}")

    def _set_logger(self):
        self.logger = Logger(
            self.config["logger_config"], self.writer, verbose=self.config["verbose"]
        )

    def _set_checkpointer(self):
        if self.config["checkpoint"]:
            self.checkpointer = Checkpointer(
                self.config["checkpoint_config"], verbose=self.config["verbose"]
            )
        else:
            self.checkpointer = None

    def _set_optimizer(self, model):
        optimizer_config = self.config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["sgd_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "rmsprop":
            optimizer = optim.RMSprop(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["rmsprop_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "sparseadam":
            optimizer = optim.SparseAdam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
            )
            if self.config["l2"]:
                raise Exception(
                    "SparseAdam optimizer does not support weight_decay (l2 penalty)."
                )
        else:
            raise ValueError(f"Did not recognize optimizer option '{opt}'")
        self.optimizer = optimizer

    def _set_scheduler(self):
        lr_scheduler = self.config["lr_scheduler"]
        if lr_scheduler is None:
            lr_scheduler = None
        else:
            lr_scheduler_config = self.config["lr_scheduler_config"]
            if lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, **lr_scheduler_config["exponential_config"]
                )
            elif lr_scheduler == "reduce_on_plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **lr_scheduler_config["plateau_config"]
                )
            else:
                raise ValueError(
                    f"Did not recognize lr_scheduler option '{lr_scheduler}'"
                )
        self.lr_scheduler = lr_scheduler
