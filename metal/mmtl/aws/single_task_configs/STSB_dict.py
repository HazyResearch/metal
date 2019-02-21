# Arguments for launch script

search_space = {
    # hyperparams
    "l2": [0, 0.001, 0.01],
    "lr": [1e-8, 1e-7, 1e-6, 5e-5, 1e-5],
    "warmup_steps": 0.5,
    "warmup_unit": "epochs",
}

launch_args = {
    # Device args
    "device": 0,
    # Model specification
    "bert_model": "bert-base-uncased",
    "bert_output_dim": 768,
    # Dataloader specification
    "max_len": 256,
    # Checkpointing and logging
    "log_every": 0.2,
    "score_every": 0.2,
    "checkpoint_metric": "STSB/valid/pearson_corr",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    "lr_scheduler": "linear",
    "lr": 1e-5,
    "n_epochs": 10,
    "l2": 0.0,
    "batch_size": 32,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "STSB",
}
