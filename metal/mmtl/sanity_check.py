import unittest

from nose.tools import nottest

from metal.mmtl.glue_tasks import create_tasks
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
        cls.tasks = create_tasks(
            task_names, max_datapoints=100, max_len=200, dl_kwargs={"batch_size": 8}
        )

    def test_mmtl_training(self):
        model = MetalModel(self.tasks, verbose=False)
        trainer = MultitaskTrainer(verbose=False)
        trainer.train_model(model, self.tasks, n_epochs=1)


if __name__ == "__main__":
    unittest.main()
