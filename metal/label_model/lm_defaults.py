lm_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # TRAIN
    "train_config": {
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
        "scheduler_config": {"scheduler": None},
        # Checkpointer
        "checkpoint": False,
        # Train loop
        "n_epochs": 100,
        "print_every": 10,
    },
}
