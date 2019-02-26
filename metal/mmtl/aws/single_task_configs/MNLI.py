# Arguments for launch script

search_space = {
    # hyperparams
    "l2": [1e-1, 1e-2, 1e-3],  # 1e-4 looked same as 1e-3
    "batch_size": [32],
    "lr": [5e-5],  # 1e-4 and 1e-5 did worse
    "min_lr": [1e-5, 1e-6, 0],
    "lr_scheduler": ["linear"],  # exponential did worse
}

launch_args = {
    # Device args
    "device": 0,
    "fp16": 1,
    # Model specification
    "bert_model": "bert-base-uncased",
    # Dataloader specification
    "max_len": 200,
    # Checkpointing and logging
    "log_every": 0.1,
    "score_every": 0.1,
    "checkpoint_metric": "MNLI/valid/accuracy",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    # "lr_scheduler": "linear",
    "lr": 1e-5,
    "n_epochs": 5,
    "l2": 0.01,
    "batch_size": 32,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "MNLI",
}
