em_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # Network
    # The first value is the output dim of the input module (or the sum of
    # the output dims of all the input modules if multitask=True and
    # multiple input modules are provided). The last value is the
    # output dim of the head layer (i.e., the cardinality of the
    # classification task). The remaining values are the output dims of
    # middle layers (if any). The number of middle layers will be inferred
    # from this list.
    "layer_out_dims": [10, 2],
    # Input layer configs
    "input_layer_config": {
        "input_relu": True,
        "input_batchnorm": False,
        "input_dropout": 0.0,
    },
    # Middle layer configs
    "middle_layer_config": {
        "middle_relu": True,
        "middle_batchnorm": False,
        "middle_dropout": 0.0,
    },
    # Can optionally skip the head layer completely, for e.g. running baseline
    # models...
    "skip_head": False,
    # GPU
    "use_cuda": False,
    # TRAINING
    "train_config": {
        # Loss function config
        "loss_fn_reduction": "sum",
        # Display
        "disable_prog_bar": False,  # Disable progress bar each epoch
        # Dataloader
        "data_loader_config": {"batch_size": 32, "num_workers": 1},
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
        # Scheduler (for learning rate)
        "lr_scheduler_config": {
            "lr_scheduler": "reduce_on_plateau",
            # ['constant', 'exponential', 'reduce_on_plateau']
            # 'reduce_on_plateau' uses checkpoint_metric to assess plateaus
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
        # Loggers
        "logger": True,
        "logger_config": {
            "log_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
            "log_train_every": 1,  # How often train metrics are calculated (optionally logged to TB)
            "log_train_metrics": [
                "train/loss"
            ],  # Can include built-in and user-defined metrics
            "log_train_metrics_func": None,  # A function that maps a model + dataloader to a dictionary of metrics
            "log_valid_every": 1,  # How frequently to evaluate on valid set (must be multiple of log_freq)
            "log_valid_metrics": ["valid/accuracy"],
            "log_valid_metrics_func": None,
        },
        # Tensorboard
        "tensorboard": False,  # If True, write certain metrics to Tensorboard
        "tensorboard_config": {
            "tensorboard_config": {  # Event file stored at log_dir/run_dir/run_name (see slicing)
                "tb_metrics": None,  # Must be a subset of log_metrics; defaults to all of them
                "log_dir": "tensorboard",
                "run_dir": None,
                "run_name": None,
            }
        },
        # Checkpointer
        "checkpoint": True,  # If True, checkpoint models when certain conditions are met
        "checkpoint_config": {
            "checkpoint_best": True,
            "checkpoint_freq": None,  # uses log_valid_unit for units; if not None, checkpoint this often regardless of performance
            "checkpoint_metric": "valid/accuracy",  # Must be in metrics dict
            "checkpoint_metric_mode": "max",  # ['max', 'min']
            "checkpoint_dir": "checkpoints",
            "checkpoint_runway": 0,
        },
    },
}
