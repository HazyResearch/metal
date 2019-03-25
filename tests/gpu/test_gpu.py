import os
import pickle
import unittest

import GPUtil

from metal.end_model import EndModel
from metal.label_model import LabelModel
from metal.utils import split_data

# Making sure we're using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GPUTest(unittest.TestCase):
    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.",
    )
    def test_gpustorage(self):
        # Running basics tutorial problem
        with open("tutorials/data/basics_tutorial.pkl", "rb") as f:
            X, Y, L, D = pickle.load(f)

        Xs, Ys, Ls, Ds = split_data(
            X, Y, L, D, splits=[0.8, 0.1, 0.1], stratify_by=Y, seed=123
        )

        label_model = LabelModel(k=2, seed=123)
        label_model.train_model(Ls[0], Y_dev=Ys[1], n_epochs=500, log_train_every=25)
        Y_train_ps = label_model.predict_proba(Ls[0])

        # Creating a really large end model to use lots of memory
        end_model = EndModel([1000, 100000, 2], seed=123, device="cuda")

        # Getting initial GPU storage use
        initial_gpu_mem = GPUtil.getGPUs()[0].memoryUsed

        # Training model
        end_model.train_model(
            (Xs[0], Y_train_ps),
            valid_data=(Xs[1], Ys[1]),
            l2=0.1,
            batch_size=256,
            n_epochs=3,
            log_train_every=1,
            validation_metric="f1",
        )

        # Final GPU storage use
        final_gpu_mem = GPUtil.getGPUs()[0].memoryUsed

        # On a Titan X, this model uses ~ 3 GB of memory
        gpu_mem_difference = final_gpu_mem - initial_gpu_mem

        self.assertGreater(gpu_mem_difference, 1000)


if __name__ == "__main__":
    unittest.main()
