# Arguments for launch script

search_space = {
    "verbose": True,
    # hyperparams
    "l2": {"range": [1e-5, 1], "scale": "log"},
    "lr": {"range": [1e-5, 1], "scale": "log"},
    # Tensorboard
    "writer": "tensorboard",
    "log_dir": f"tensorboard_logs",
    # Glue
    "trainer_metrics": ["glue"],
    "lr_scheduler": "linear",
    "warmup_steps": 0.5,
    "warmup_unit": "epochs",
}

launch_args = {
    "device": 0,
    "bert_model": "bert-base-uncased",
    "bert_output_dim": 768,
    "max_len": 200,
    "lr_scheduler": "exponential",
    "log_every": 0.25,
    "score_every": 0.25,
    "checkpoint_dir": "checkpoint",
    "checkpoint_metric": "train/loss",
    "checkpoint_best": 1,
    "progress_bar": 1,
    "lr": 0.01,
    "n_epochs": 5,
    "l2": 0.0,
    "batch_size": 16,
    "split_prop": 0.8,
    "max_datapoints": 1000,
    "log_dir": "tensorboard_logs",
    "writer": "tensorboard",
    "tasks": "COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI",
}
