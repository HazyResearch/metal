import unittest

from nose.tools import nottest

from metal.mmtl.glue.glue_metrics import glue_score
from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.trainer import MultitaskTrainer


@nottest
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
        tasks, payloads = create_glue_tasks_payloads(
            task_names, max_datapoints=100, max_len=200, dl_kwargs={"batch_size": 8}
        )
        cls.tasks = tasks
        cls.payloads = payloads

    def test_mmtl_training(self):
        model = MetalModel(self.tasks, verbose=False)
        trainer = MultitaskTrainer(verbose=True)
        trainer.train_model(
            model, self.payloads, n_epochs=1, aggregate_metric_fns=[glue_score]
        )


if __name__ == "__main__":
    unittest.main()
