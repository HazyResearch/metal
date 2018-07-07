import sys
import unittest

import numpy as np
import torch

sys.path.append("../metal")
from metal.end_model import EndModel, LogisticRegression
from metal.structs import TaskGraph

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
        Ys = [Y[:1000, 0], Y[1000:1100, 0], Y[1100:, 0]]
        cls.single_problem = (Xs, Ys)

        Ys = [Y[:1000, :], Y[1000:1100, :], Y[1100:, :]]
        cls.multi_problem = (Xs, Ys)

    def test_logreg(self):
        em = LogisticRegression(seed=1, verbose=False,
            input_dim=2)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 0.99)

    def test_singletask(self):
        em = EndModel(seed=1, verbose=False, 
            layer_output_dims=[2,4,2], batchnorm=False, dropout=0.0)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 1.0)

    def test_singletask_extras(self):
        em = EndModel(seed=1, verbose=False, 
            layer_output_dims=[2,4,2], batchnorm=True, dropout=0.05)
        Xs, Ys = self.single_problem
        em.train(Xs[0], Ys[0], Xs[1], Ys[1], verbose=False, n_epochs=10)
        score = em.score(Xs[2], Ys[2], verbose=False)
        self.assertEqual(score, 0.93)

    # def test_multitask_top(self):
    #     label_map=[[1,2],[1,2]]
    #     em = EndModel(
    #         label_map=label_map, 
    #         seed=1,
    #         verbose=False,
    #         dropout=0.0,
    #         layer_output_dims=[2,4,2],
    #         head_layers='top',
    #     )
    #     em.train(self.X_train, self.Y_train, self.X_dev, self.Y_dev,
    #         verbose=False,
    #         n_epochs=10,
    #     )
    #     score = em.score(self.X_test, self.Y_test, reduce='mean', verbose=False)
    #     self.assertEqual(score, 0.975)

    # def test_multitask_auto(self):
        # label_map=[[1,2],[1,2]]
        # edges = [(0,1)]
        # cards = [2,2]
        # tg = TaskGraph(edges, cards)
        # em = EndModel(
        #     label_map=label_map,
        #     task_graph=tg,
        #     seed=1,
        #     dropout=0.0,
        #     layer_output_dims=[2,4,2],
        #     head_layers='auto')
        # em.train(self.X_train, self.Y_train, self.X_dev, self.Y_dev,
        #     verbose=False,
        #     n_epochs=10,)
        # score = em.score(self.X_test, self.Y_test, reduce='mean')
        # self.assertEqual(score, ?)

    # def test_multitask_custom(self):
    #     label_map=[[1,2],[1,2]]
    #     em = EndModel(
    #         label_map=label_map, 
    #         seed=1,
    #         verbose=False,
    #         dropout=0.0,
    #         layer_output_dims=[2,4,2],
    #         head_layers=[1,2],
    #     )
    #     em.train(self.X_train, self.Y_train, self.X_dev, self.Y_dev,
    #         verbose=False,
    #         n_epochs=10,
    #     )
    #     score = em.score(self.X_test, self.Y_test, reduce='mean', verbose=False)
    #     self.assertEqual(score, 0.96)
        
if __name__ == '__main__':
    unittest.main()        