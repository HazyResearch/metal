import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.multitask import MTEndModel, TaskGraph

class EndModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seed
        np.random.seed(1)

        N = 1200

        X = np.random.random((N,2)) * 2 - 1
        Y = np.zeros((N,2))
        Y[:,0] = (X[:,0] > X[:,1] + 0.5).astype(int) + 1
        Y[:,1] = (X[:,0] > X[:,1] + 0.25).astype(int) + 1

        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)

        Xs = [X[:1000], X[1000:1100], X[1100:]]
        Ys = [[Y[:1000,     0], Y[:1000,     1]],
              [Y[1000:1100, 0], Y[1000:1100, 1]], 
              [Y[1100:,     0], Y[1100:,     1]]] 
        cls.Xs = Xs
        cls.Ys = Ys

    def test_multitask_top(self):
        edges = []
        cards = [2,2]
        tg = TaskGraph(edges, cards)
        em = MTEndModel(
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            layer_output_dims=[2,4,2],
            task_head_layers='top',
        )
        em.train(self.Xs[0], self.Ys[0], self.Xs[1], self.Ys[1],
            verbose=False,
            n_epochs=10,
        )
        score = em.score(self.Xs[2], self.Ys[2], reduce='mean', verbose=False)
        self.assertEqual(score, 0.940)

    # TODO: Uncomment this test once 'auto' assignment of task heads to layers
    # is implemented in MTEndModel
    # def test_multitask_auto(self):
    #     edges = [(0,1)]
    #     cards = [2,2]
    #     tg = TaskGraph(edges, cards)
    #     em = MTEndModel(
    #         task_graph=tg,
    #         seed=1,
    #         dropout=0.0,
    #         layer_output_dims=[2,4,2],
    #         task_head_layers='auto')
    #     em.train(self.Xs[0], self.Ys[0], self.Xs[1], self.Ys[1],
    #         verbose=False,
    #         n_epochs=10,
    #     )
    #     score = em.score(self.Xs[2], self.Ys[2], reduce='mean')
    #     self.assertEqual(score, 42)

    # Having the output of the first task (X > 0.5) should be helpful for the
    # second task (X > 0.25).
    def test_multitask_custom(self):
        edges = [(0,1)]
        cards = [2,2]
        tg = TaskGraph(edges, cards)
        em = MTEndModel(
            task_graph=tg,
            seed=1,
            verbose=False,
            dropout=0.0,
            layer_output_dims=[2,4,2],
            task_head_layers=[1,2],
        )
        em.train(self.Xs[0], self.Ys[0], self.Xs[1], self.Ys[1],
            verbose=False,
            n_epochs=10,
        )
        score = em.score(self.Xs[2], self.Ys[2], reduce='mean', verbose=False)
        self.assertEqual(score, 0.965)
        
if __name__ == '__main__':
    unittest.main()        