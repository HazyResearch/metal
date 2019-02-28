# Arguments for launch script

search_space = {
    # hyperparams
    "l2": [0],
    "batch_size": [16],
    "lr": [1e-4, 5e-5, 5e-5, 2e-5],
    "lr_scheduler": ["linear"],
}

launch_args = {
    # Device args
    "device": 0,
    "fp16": 1,
    # Model specification
    "bert_model": "bert-large-uncased",
    # "bert_output_dim": 1024,
    # Dataloader specification
    "max_len": 200,
    # Checkpointing and logging
    "log_every": 0.1,
    "score_every": 0.1,
    "checkpoint_metric": "model/valid/glue",
    "checkpoint_metric_mode": "max",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    # "lr_scheduler": "linear",
    "lr": 1e-5,
    "n_epochs": 10,
    "l2": 0.01,
    "batch_size": 32,
    "max_datapoints": -1,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "QQP",
}
