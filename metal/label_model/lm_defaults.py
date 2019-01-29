lm_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # GPU
    "use_cuda": False,
    # TRAIN
    "train_config": {
        # Dataloader
        "data_loader_config": {"batch_size": 1000, "num_workers": 1},
        # Classifier
        # Class balance (if learn_class_balance=False, fix to class_balance)
        "learn_class_balance": False,
        # LF precision initializations / priors (float or np.array)
        "prec_init": 0.7,
        # Centered L2 regularization strength (int, float, or np.array)
        "l2": 0.0,
        # Optimizer
        "optimizer_config": {
            "optimizer": "sgd",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
        },
        # Scheduler
        "lr_scheduler_config": {"lr_scheduler": None},
        # Train loop
        "n_epochs": 100,
        "print_every": 10,
        "disable_prog_bar": True,  # Disable progress bar each epoch
        # Loggers
        "train_logger_config": {
            "log_train_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
            "log_train_every": 1,  # How often train metrics are calculated (optionally logged to TB)
            "log_train_print_freq": None,  # None: use log_freq, 0: never print, X: print every X units
            "log_train_metrics": [
                "loss"
            ],  # Can include built-in and user-defined metrics
            "log_train_metrics_func": None,  # A function that maps a model + dataloader to a dictionary of metrics
        },
        "valid_logger_config": {
            "log_valid_unit": "epochs",
            "log_valid_every": 1,
            "log_valid_print_freq": None,
            "log_valid_metrics": ["accuracy"],
            "log_valid_metrics_func": None,
        },
        # Tensorboard
        "tensorboard": False,
        # Checkpointer
        "checkpoint": False,
    },
}
