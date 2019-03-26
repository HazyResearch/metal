import os

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.trainer import MultitaskTrainer

task_name = "MRPC"

writer_dict = {
    "writer": "tensorboard",
    "writer_config": {  # Log (or event) file stored at log_dir/run_dir/run_name
        "log_dir": f"{os.environ['METALHOME']}/logs",
        "run_dir": "BERT_base",
        "run_name": task_name,
        "writer_metrics": [],  # May specify a subset of metrics in metrics_dict to be written
        "include_config": True,  # If True, include model config in log
    },
}

tasks, payloads = create_glue_tasks_payloads(
    [
        task_name,
        # "COLA",
        # "SST2",
        # "MNLI",
        # "RTE",
        # "WNLI",
        # "QQP",
        # "MRPC",
        # "STSB",
        # "QNLI"
    ],
    split_prop=0.99,
    #  max_datapoints = 100,
    dl_kwargs={"batch_size": 2},
)

model = MetalModel(tasks)
trainer = MultitaskTrainer(**writer_dict)

trainer.train_model(
    model,
    payloads,
    lr=0.00001,
    l2=0,
    log_every=1,
    checkpoint_metric="model/train/loss",
    checkpoint_metric_mode="min",
    checkpoint_best=True,
    n_epochs=3,
    progress_bar=True,
)

print("WE WIN!")
