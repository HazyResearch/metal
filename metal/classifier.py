import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from metal.metrics import metric_score


class Classifier(nn.Module):
    """Simple abstract base class for a probabilistic classifier.
    
    The main contribution of children classes will be an implementation of the
    predict_proba() method. The relationships between the predict/score
    functions are as follows:

    score
	    |
	predict
	    |
	*predict_proba
    
    The method predict_proba() method calculates the probabilistic labels,
    the predict() method handles tie-breaking, and the score() method 
    calculates metrics based on predictions.
    """

    def __init__(self, cardinality=2, seed=None):
        super().__init__()
        self.multitask = False
        self.k = cardinality

        if seed is None:
            seed = np.random.randint(1e6)
        self._set_seed(seed)

    def _set_seed(self, seed):
        self.seed = seed
        if torch.cuda.is_available():
            # TODO: confirm this works for gpus without knowing gpu_id
            # torch.cuda.set_device(self.config['gpu_id'])
            torch.backends.cudnn.enabled = True
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _reset_module(m):
        """An initialization method to be applied recursively to all modules"""
        raise NotImplementedError

    def save(self, filepath):
        """Serialize and save a model.
        
        Example:
            end_model = EndModel(...)
            end_model.train(...)
            end_model.save('my_end_model.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Deserialize and load a model.

        Example:
            end_model = EndModel.load('my_end_model.pkl')
            end_model.score(...)
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def reset(self):
        """Initializes all modules in a network"""
        # The apply(f) method recursively calls f on itself and all children
        self.apply(self._reset_module)

    def train(self, X, Y, X_dev=None, Y_dev=None, **kwargs):
        """Trains a classifier

        Take care to initialize weights outside the training loop and zero out 
        gradients at the beginning of each iteration inside the loop.
        """
        raise NotImplementedError

    def score(self, X, Y, metric=['accuracy'], break_ties='random', 
        verbose=True, **kwargs):
        """Scores the predictive performance of the Classifier on all tasks

        Args:
            X: The input for the predict method
            Y: An [N] or [N, 1] torch.Tensor or np.ndarray of gold labels in 
                {1,...,K_t}
            metric: A metric (string) with which to score performance or a 
                list of such metrics
            break_ties: How to break ties when making predictions

        Returns:
            scores: A (float) score
        """
        Y = self._to_numpy(Y)
        Y_p = self.predict(X, break_ties=break_ties, **kwargs)

        metric_list = metric if isinstance(metric, list) else [metric]
        for metric in metric_list:
            score = metric_score(Y, Y_p, metric, ignore_in_gold=[0])
            if verbose:
                print(f"{metric.capitalize()}: {score:.3f}")

        return score

    def predict(self, X, break_ties='random', **kwargs):
        """Predicts hard (int) labels for an input X on all tasks
        
        Args:
            X: The input for the predict_proba method
            break_ties: A tie-breaking policy

        Returns:
            An N-dim np.ndarray of predictions
        """
        Y_p = self._to_numpy(self.predict_proba(X, **kwargs))
        Y_ph = self._break_ties(Y_p, break_ties)
        return Y_ph

    def predict_proba(self, X, **kwargs):
        """Predicts soft probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            An [N, K_t] np.ndarray of soft predictions
        """
        raise NotImplementedError

    def _break_ties(self, Y_s, break_ties='random'):
        """Break ties in each row of a tensor according to the specified policy

        Args:
            Y_s: An [N, K_t] np.ndarray of probabilities
            break_ties: A tie-breaking policy:
                'abstain': return an abstain vote (0)
                'random': randomly choose among the tied options
                    NOTE: if break_ties='random', repeated runs may have 
                    slightly different results due to difference in broken ties
        """
        N, k = Y_s.shape
        Y_th = np.zeros(N)
        diffs = np.abs(Y_s - Y_s.max(axis=1).reshape(-1, 1))

        TOL = 1e-5
        for i in range(N):
            max_idxs = np.where(diffs[i, :] < TOL)[0]
            if len(max_idxs) == 1:
                Y_th[i] = max_idxs[0] + 1
            # Deal with 'tie votes' according to the specified policy
            elif break_ties == 'random':
                Y_th[i] = np.random.choice(max_idxs) + 1
            elif break_ties == 'abstain':
                Y_th[i] = 0
            else:
                ValueError(f'break_ties={break_ties} policy not recognized.')     
        return Y_th 

    @staticmethod
    def _to_numpy(Z):
        """Converts a None, list, np.ndarray, or torch.Tensor to np.ndarray"""
        if Z is None:
            return Z
        elif isinstance(Z, np.ndarray):
            return Z
        elif isinstance(Z, list):
            return np.array(Z)
        elif isinstance(Z, torch.Tensor):
            return Z.numpy()
        else:
            msg = (f"Expected None, list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead.")
            raise Exception(msg)

    @staticmethod
    def _to_torch(Z):
        """Converts a None, list, np.ndarray, or torch.Tensor to torch.Tensor"""
        if Z is None:
            return None
        elif isinstance(Z, torch.Tensor):
            return Z
        elif isinstance(Z, list):
            return torch.from_numpy(np.array(Z))
        elif isinstance(Z, np.ndarray):
            return torch.from_numpy(Z)
        else:
            msg = (f"Expected list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead.")
            raise Exception(msg)

    def _check(self, var, val=None, typ=None, shape=None):
        if val is not None and not var != val:
            msg = f"Expected value {val} but got value {var}."
            raise ValueError(msg)
        if typ is not None and not isinstance(var, typ):
            msg = f"Expected type {typ} but got type {type(var)}."
            raise ValueError(msg)
        if shape is not None and not var.shape != shape:
            msg = f"Expected shape {shape} but got shape {var.shape}."
            raise ValueError(msg)
            
    def _check_or_set_attr(self, name, val, set_val=False):
        if set_val:
            setattr(self, name, val)
        else:
            true_val = getattr(self, name)
            if val != true_val:
                raise Exception(f"{name} = {val}, but should be {true_val}.")