import unittest

import numpy as np
import torch
import torch.nn as nn

from metal.modules import IdentityModule
from metal.multitask import MTEndModel, TaskGraph, TaskHierarchy


class MTEndModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        n = 1200

        X = np.random.random((n, 2)) * 2 - 1
        Y = np.zeros((n, 2))
        Y[:, 0] = (X[:, 0] > X[:, 1] + 0.5).astype(int) + 1
        Y[:, 1] = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

        X = torch.tensor(X, dtype=torch.float)

        Xs = [X[:1000], X[1000:1100], X[1100:]]
        Ys = [
            [Y[:1000, 0], Y[:1000, 1]],
            [Y[1000:1100, 0], Y[1000:1100, 1]],
            [Y[1100:, 0], Y[1100:, 1]],
        ]
        cls.Xs = Xs
        cls.Ys = Ys

    def test_multitask_top(self):
        """Attach all task heads to the top layer"""
        edges = []
        cards = [2, 2]
        tg = TaskGraph(cards, edges)
        em = MTEndModel(
            layer_out_dims=[2, 8, 4],
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            task_head_layers="top",
        )
        top_layer = len(em.config["layer_out_dims"]) - 1
        self.assertEqual(len(em.task_map[top_layer]), em.t)
        em.train(
            (self.Xs[0], self.Ys[0]),
            dev_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
        )
        score = em.score((self.Xs[2], self.Ys[2]), reduce="mean", verbose=False)
        self.assertGreater(score, 0.95)

    def test_multitask_custom_attachments(self):
        """Attach the task heads at user-specified layers"""
        edges = [(0, 1)]
        cards = [2, 2]
        tg = TaskHierarchy(cards, edges)
        em = MTEndModel(
            layer_out_dims=[2, 8, 4],
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            task_head_layers=[1, 2],
        )
        self.assertEqual(em.task_map[1][0], 0)
        self.assertEqual(em.task_map[2][0], 1)
        em.train(
            (self.Xs[0], self.Ys[0]),
            dev_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
        )
        score = em.score((self.Xs[2], self.Ys[2]), reduce="mean", verbose=False)
        self.assertGreater(score, 0.95)

    def test_multitask_two_modules(self):
        """Accept a different representation for each task"""
        edges = []
        cards = [2, 2]
        tg = TaskGraph(cards, edges)
        em = MTEndModel(
            layer_out_dims=[2, 8, 4],
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            input_modules=[IdentityModule(), IdentityModule()],
            task_head_layers="top",
        )
        Xs = []
        for i, X in enumerate(self.Xs):
            Xs.append([X[:, 0], X[:, 1]])
        em.train(
            (Xs[0], self.Ys[0]),
            dev_data=(Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
        )
        score = em.score((Xs[2], self.Ys[2]), reduce="mean", verbose=False)
        self.assertGreater(score, 0.95)

    def test_multitask_custom_heads(self):
        """Accept a different representation for each task"""
        edges = []
        cards = [2, 2]
        tg = TaskGraph(cards, edges)
        em = MTEndModel(
            layer_out_dims=[2, 8, 4],
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            head_modules=[nn.Linear(8, 2), nn.Linear(4, 2)],
            task_head_layers=[1, 2],
        )
        em.train(
            (self.Xs[0], self.Ys[0]),
            dev_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
        )
        score = em.score((self.Xs[2], self.Ys[2]), reduce="mean", verbose=False)
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
