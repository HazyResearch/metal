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
        "print_every": 1,  # Print after this many epochs
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
        },
        # Scheduler
        "scheduler_config": {
            "scheduler": "reduce_on_plateau",
            # ['constant', 'exponential', 'reduce_on_plateu']
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
        # Checkpointer
        "checkpoint": True,
        "checkpoint_config": {
            "checkpoint_min": -1,
            # The initial best score to beat to merit checkpointing
            "checkpoint_runway": 0,
            # Don't start taking checkpoints until after this many epochs
            "checkpoint_destination": "checkpoints",
        },
    },
}
