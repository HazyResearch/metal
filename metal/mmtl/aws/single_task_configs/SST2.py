# Arguments for launch script

search_space = {
    # hyperparams
    "l2": [0, 0, 0, 0, 0],
    "lr": [5e-5],
    "tasks": ["SST2"],
    "n_epochs": [3],
}

launch_args = {
    # Device args
    "device": 0,
    # "fp16": 1,
    # Model specification
    "bert_model": "bert-base-uncased",
    # Dataloader specification
    "max_len": 200,
    # Checkpointing and logging
    "log_every": 0.25,
    "score_every": 0.25,
    "checkpoint_metric": "model/valid/gold/glue_score",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    "lr_scheduler": "linear",
    "n_epochs": 10,
    "batch_size": 32,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
}
