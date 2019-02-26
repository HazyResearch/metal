# Arguments for launch script

search_space = {
    # hyperparams
    "l2": {"range": [1e-4, 1e-1], "scale": "log"},
    "lr": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
    # Glue
    "trainer_metrics": ["glue"],
    "lr_scheduler": "linear",
    "warmup_steps": [0, 0.1, 0.3, 0.5, 0.7],
    "warmup_unit": "epochs",
    "batch_size": [2, 4, 8, 16, 32],
    "split_prop": [0.8, 0.9, 0.99],
}

launch_args = {
    # Device args
    "device": 0,
    # Model specification
    "bert_model": "bert-base-uncased",
    # Dataloader specification
    "max_len": 200,
    "lr_scheduler": "linear",
    # Checkpointing and logging
    "log_every": 0.25,
    "score_every": 0.25,
    "checkpoint_metric": "COLA/valid/matthews_corr",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    "lr_scheduler": "linear",
    "lr": 1e-5,
    "n_epochs": 5,
    "l2": 0.0,
    "batch_size": 8,
    "split_prop": 0.8,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "COLA",
}
