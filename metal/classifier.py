import random

import numpy as np
import torch
import torch.nn as nn

from metal.analysis import confusion_matrix
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

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

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


class MTClassifier(Classifier):
    """Simple abstract base class for a *multi-class* probabilistic classifier.
    
    The main contribution of children classes will be an implementation of the
    predict_proba() method. The relationships between the six predict/score
    functions are as follows:
    score 		    	score_task
	    |			         |
	predict 	     	predict_task
	    |	    (default)    |
	*predict_proba  <- 	predict_task_proba
    
    Methods on the left return a list of results for all tasks (including
    a singleton list if there is only one task). 
    Methods on the right return what would be a single element in the list 
    returned by their counterpart on the left.

    The method predict_proba() method calculates the probabilistic labels,
    the predict() method handles tie-breaking, and the score() method 
    calculates metrics based on predictions.

    Children classes must implement predict_proba so that interactions between 
    tasks are handled correctly in applicable multi-task settings. 
    If it is possible to calculate task probabilities independently, they may 
    also override predict_task_proba for efficiency. 
    Otherwise, predict_task_proba() will default to calling predict_proba and 
    accessing element t.
    """

    def __init__(self, cardinalities, seed=None):
        super().__init__()
        self.multitask = True
        self.K = cardinalities

        if seed is None:
            seed = np.random.randint(1e6)
        self._set_seed(seed)

    def score(self, X, Y, metric='accuracy', reduce='mean', break_ties='random',
        verbose=True, **kwargs):
        """Scores the predictive performance of the Classifier on all tasks

        Args:
            X: The input for the predict method
            Y: A T-length list of [N] or [N, 1] np.ndarrays or torch.Tensors of 
                gold labels in {1,...,K_t}
            metric: The metric with which to score performance on each task
            reduce: How to reduce the scores of multiple tasks:
                 None : return a T-length list of scores
                'mean': return the mean score across tasks
            break_ties: How to break ties when making predictions
        Returns:
            scores: A (float) score or a T-length list of such scores if 
                reduce=None
        """
        self._check(Y, typ=list)
        Y = [self._to_numpy(Y_t) for Y_t in Y]

        Y_p = self.predict(X, break_ties=break_ties, **kwargs)
        self._check(Y_p, typ=list)

        task_scores = []
        for t, Y_tp in enumerate(Y_p):
            score = metric_score(Y[t], Y_tp, metric, ignore_in_gold=[0])
            task_scores.append(score)

        # TODO: Other options for reduce, including scoring only certain
        # primary tasks, and converting to end labels using TaskGraph...
        if reduce is None:
            score = task_scores
        elif reduce == 'mean':
            score = np.mean(task_scores)
        else:
            raise Exception(f"Keyword reduce='{reduce}' not recognized.")

        if verbose:
            if reduce is None:
                for t, score_t in enumerate(score):
                    print(f"{metric.capitalize()} (t={t}): {score_t:0.3f}")
            else:
                print(f"{metric.capitalize()}: {score:.3f}")

        return score

    def predict(self, X, break_ties='random', **kwargs):
        """Predicts hard (int) labels for an input X on all tasks
        
        Args:
            X: The input for the predict_proba method
            break_ties: A tie-breaking policy
        Returns:
            A T-length list of N-dim np.ndarrays of predictions
        """
        Y_p = self.predict_proba(X, **kwargs)
        self._check(Y_p, typ=list)
        self._check(Y_p[0], typ=np.ndarray)

        Y_ph = []
        for Y_tp in Y_p:
            Y_tph = self._break_ties(Y_tp, break_ties)
            Y_ph.append(Y_tph)

        return Y_ph

    def predict_proba(self, X, **kwargs):
        """Predicts soft probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            A T-length list of [N, K_t] np.ndarrays of soft predictions
        """
        raise NotImplementedError

    def score_task(self, X, Y, t=0, metric='accuracy', verbose=True, **kwargs):
        """Scores the predictive performance of the Classifier on task t
        
        Args:
            X: The input for the predict_task method
            Y: A [N] or [N, 1] np.ndarray or torch.Tensor of gold labels in 
                {1,...,K_t}
            t: The task index to score
            metric: The metric with which to score performance on this task
        Returns:
            The (float) score of the Classifier for the specified task and 
            metric
        """
        Y = self._to_numpy(Y)
        Y_tp = self.predict_task(X, t=t, **kwargs)
        score = metric_score(Y, Y_tp, metric, ignore_in_gold=[0], **kwargs)
        if verbose:
            print(f"[t={t}] {metric.capitalize()}: {score:.3f}")
        return score

    def predict_task(self, X, t=0, break_ties='random', **kwargs):
        """Predicts hard (int) labels for an input X on task t
        
        Args:
            X: The input for the predict_task_proba method
            t: The task index to predict
        Returns:
            An N-dim tensor of hard (int) predictions for the specified task
        """
        Y_tp = self.predict_task_proba(X, **kwargs)
        Y_tph = self._break_ties(Y_tp, break_ties)
        return Y_tph

    def predict_task_proba(self, X, t=0, **kwargs):
        """Predicts soft probabilistic labels for an input X on task t
        
        Args:
            X: The input for the predict_proba method
            t: The task index to predict for which to predict probabilities
        Returns:
            An [N, K_t] tensor of predictions for task t

        NOTE: By default, this method calls predict_proba and extracts element
        t. If it is possible to predict individual tasks in isolation, however,
        this method may be overriden for efficiency's sake.
        """
        return self.predict_proba(X, **kwargs)[t]        