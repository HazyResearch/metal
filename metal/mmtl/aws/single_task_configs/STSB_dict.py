# Arguments for launch script

search_space = {
    # hyperparams
    "l2": [0, 0.001, 0.01],
    "lr": [1e-8, 1e-7, 1e-6],
    # Glue
    "trainer_metrics": ["glue"],
    "warmup_steps": [0, 0.1, 0.3, 0.5, 0.7],
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
    "log_every": 0.25,
    "score_every": 0.25,
    "checkpoint_metric": "STSB/valid/pearson_corr",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    "lr_scheduler": "linear",
    "lr": 1e-5,
    "n_epochs": 5,
    "l2": 0.0,
    "batch_size": 32,
    "split_prop": 0.9,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "STSB",
}
