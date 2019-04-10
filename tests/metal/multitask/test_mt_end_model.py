import unittest

import numpy as np
import torch
import torch.nn as nn

from metal.end_model.identity_module import IdentityModule
from metal.metrics import METRICS
from metal.multitask import MTEndModel
from metal.multitask.task_graph import TaskGraph, TaskHierarchy


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
            task_head_layers="top",
        )
        top_layer = len(em.config["layer_out_dims"]) - 1
        self.assertEqual(len(em.task_map[top_layer]), em.t)
        em.train_model(
            (self.Xs[0], self.Ys[0]),
            valid_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
            checkpoint=False,
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
            task_head_layers=[1, 2],
        )
        self.assertEqual(em.task_map[1][0], 0)
        self.assertEqual(em.task_map[2][0], 1)
        em.train_model(
            (self.Xs[0], self.Ys[0]),
            valid_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
            checkpoint=False,
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
            input_modules=[IdentityModule(), IdentityModule()],
            task_head_layers="top",
        )
        Xs = []
        for i, X in enumerate(self.Xs):
            Xs.append([X[:, 0], X[:, 1]])
        em.train_model(
            (Xs[0], self.Ys[0]),
            valid_data=(Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
            checkpoint=False,
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
            head_modules=[nn.Linear(8, 2), nn.Linear(4, 2)],
            task_head_layers=[1, 2],
        )
        em.train_model(
            (self.Xs[0], self.Ys[0]),
            valid_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=10,
            checkpoint=False,
        )
        score = em.score((self.Xs[2], self.Ys[2]), reduce="mean", verbose=False)
        self.assertGreater(score, 0.95)

    def test_scoring(self):
        edges = [(0, 1)]
        cards = [2, 2]
        tg = TaskHierarchy(cards, edges)
        em = MTEndModel(layer_out_dims=[2, 8, 4], task_graph=tg, seed=1, verbose=False)
        em.train_model(
            (self.Xs[0], self.Ys[0]),
            valid_data=(self.Xs[1], self.Ys[1]),
            verbose=False,
            n_epochs=3,
            checkpoint=False,
            validation_task=0,
        )
        tasks = [0, 1]
        for metric in METRICS:
            all_scores = em.score(
                (self.Xs[2], self.Ys[2]), metric=metric, reduce=None, verbose=False
            )
            task_specific_scores_score_method = [
                em.score(
                    (self.Xs[2], self.Ys[2]),
                    metric=metric,
                    validation_task=task,
                    verbose=False,
                )
                for task in tasks
            ]
            task_specific_scores_score_task_method = [
                em.score_task(
                    self.Xs[2], self.Ys[2], t=task, metric=metric, verbose=False
                )
                for task in tasks
            ]

            for i in range(len(tasks)):
                self.assertEqual(
                    all_scores[i],
                    task_specific_scores_score_method[i],
                    task_specific_scores_score_task_method[i],
                )


if __name__ == "__main__":
    unittest.main()
