import unittest
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.payload import Payload
from metal.mmtl.task import ClassificationTask
from metal.mmtl.trainer import MultitaskTrainer
from metal.utils import split_data

SPLITS = ["train", "valid", "test"]


def create_tasks(T):
    tasks = []
    for t in range(T):
        task_name = f"task{t}"
        input_module = nn.Linear(2, 8)
        head_module = nn.Linear(8, 2)
        task = ClassificationTask(
            name=task_name, input_module=input_module, head_module=head_module
        )
        tasks.append(task)
    return tasks


def create_payloads(N, T, batch_size=1):
    # Create two instance sets from the same (uniform) distribution, each of which
    # have labels for the same tasks (classification with respect to parallel
    # linear boundaries).

    labels_to_tasks = {f"labelset{t}": f"task{t}" for t in range(T)}
    payloads = []

    for t in range(T):
        X = np.random.random((N, 2)) * 2 - 1
        Y = np.zeros((N, 2))
        Y[:, 0] = (X[:, 0] > X[:, 1] + 0.5).astype(int) + 1
        Y[:, 1] = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

        uids = list(range(t * N, (t + 1) * N))
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)

        uid_lists, Xs, Ys = split_data(uids, X, Y, splits=[0.8, 0.1, 0.1], shuffle=True)

        for i, split in enumerate(SPLITS):
            payload_name = f"payload{t}_{split}"
            X_dict = {"data": Xs[i], "uids": uid_lists[i]}
            Y_dict = {f"labelset{t}": Ys[i][:, t] for t in range(T)}
            dataset = MmtlDataset(X_dict, Y_dict)
            data_loader = MmtlDataLoader(dataset, batch_size=batch_size)
            payload = Payload(payload_name, data_loader, labels_to_tasks, split)
            payloads.append(payload)
    return payloads


class MmtlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.trainer = MultitaskTrainer(verbose=False, lr=0.005)

    def test_mmtl_singletask(self):
        """One task with one train payload and one labelset"""
        N = 600
        T = 1

        tasks = create_tasks(T)
        model = MetalModel(tasks, verbose=False)
        payloads = create_payloads(N, T, batch_size=2)
        metrics_dict = self.trainer.train_model(model, payloads)
        self.assertEqual(len(metrics_dict), len(SPLITS) * T)
        for metric, score in metrics_dict.items():
            self.assertGreater(score, 0.9)

    def test_mmtl_multitask(self):
        """Two tasks with two train payloads and two labelsets each"""
        N = 600
        T = 2

        tasks = create_tasks(T)
        model = MetalModel(tasks, verbose=False)
        payloads = create_payloads(N, T, batch_size=2)
        metrics_dict = self.trainer.train_model(model, payloads, verbose=False)
        # For 3 payloads, each of 2 tasks each has 2 label sets
        self.assertEqual(len(metrics_dict), len(SPLITS) * T ** 2)
        for metric, score in metrics_dict.items():
            self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main()
