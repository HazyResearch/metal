# Arguments for launch script

search_space = {
    # hyperparams
    "l2": {"range": [1e-5, 1], "scale": "log"},
    "lr": {"range": [1e-5, 1], "scale": "log"},
    # Glue
    "trainer_metrics": ["glue"],
    "lr_scheduler": "linear",
    "warmup_steps": 0.5,
    "warmup_unit": "epochs",
}

launch_args = {
    # Device args
    "device": 0,
    # Model specification
    "bert_model": "bert-base-uncased",
    # Dataloader specification
    "max_len": 200,
    "lr_scheduler": "exponential",
    # Checkpointing and logging
    "log_every": 0.25,
    "score_every": 0.25,
    "checkpoint_metric": "model/train/loss",
    "checkpoint_clean": 1,
    "checkpoint_best": 1,
    "progress_bar": 1,
    # Training settings
    "lr_scheduler": "exponential",
    "lr": 0.01,
    "n_epochs": 5,
    "l2": 0.0,
    "batch_size": 16,
    "split_prop": 0.8,
    "max_datapoints": 1000,
    # Writer arguments
    "writer": "tensorboard",
    # "run_dir": "test_run",
    # "run_name": "test_name"
    # Task arguments
    "tasks": "COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI",
}
