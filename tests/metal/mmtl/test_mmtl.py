import unittest

from metal.mmtl.BERT_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.trainer import MultitaskTrainer


class MMTLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task_names = [
            "COLA",
            "SST2",
            "MNLI",
            "RTE",
            "WNLI",
            "QQP",
            "MRPC",
            "STSB",
            "QNLI",
        ]
        cls.tasks = create_tasks(
            task_names, max_datapoints=100, dl_kwargs={"batch_size": 8}
        )

    def test_mmtl_training(self):
        model = MetalModel(self.tasks)
        trainer = MultitaskTrainer()
        trainer.train_model(
            model,
            self.tasks,
            checkpoint_metric="train/loss",
            checkpoint_metric_mode="min",
            n_epochs=1,
            verbose=False,
        )
