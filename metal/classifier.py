import numpy as np
import torch
import torch.nn as nn

from metal.metrics import metric_score, confusion_matrix
from metal.utils import multitask_decorator

class Classifier(nn.Module):
    """Simple abstract base class for a probabilistic classifier.
    
    The main contribution of children classes will be an implementation of the
    predict_proba() method. The relationships between the six predict/score
    functions are as follows:

    score 		    	score_task
	    |			         |
	predict 	     	predict_task
	    |	    (default)    |
	*predict_proba  <- 	predict_task_proba
    
    Methods on the left return a list of results for all tasks (or a single 
    result not in a list in the case of a single-task problem). 
    Methods on the right return what would be a single element in the list 
    returned by their counterpart on the left.
    The score method handles reducing the scores from multiple tasks, and the
    predict methods handles tie-breaking.
    
    Children classes must implement predict_proba so that interactions between 
    tasks are handled correctly in applicable multi-task settings. 
    If it is possible to calculate task probabilities independently, they may 
    also override predict_task_proba for efficiency. 
    Otherwise, predict_task_proba() will default to calling predict_proba and 
    accessing element t.
    """

    def __init__(self, multitask=False):
        super().__init__()
        self.multitask = multitask

    @staticmethod
    def _reset_module(m):
        """An initialization method to be applied recursively to all modules"""
        raise NotImplementedError

    def reset(self):
        """Initializes all modules in a network"""
        # The apply(f) method recursively calls f on itself and all children
        self.apply(self._reset_module)

    def train(self, X, Y, **kwargs):
        """Trains a classifier

        Take care to initialize weights outside the training loop and zero out 
        gradients at teh beginning of each iteration inside the loop.
        """
        raise NotImplementedError

    def score(self, X, Y, metric='accuracy', reduce='mean', verbose=True, 
        **kwargs):
        """Scores the predictive performance of the Classifier on all tasks

        Args:
            X: The input for the predict method
            Y: An [N] or [N, 1] tensor of gold labels in {1,...,K_t} or a
                T-length list of such tensors if multitask=True.
            metric: The metric with which to score performance on each task
            reduce: How to reduce the scores of multiple tasks if multitask=True
                None: return a T-length list of scores
                'mean': return the mean score across tasks

        Returns:
            scores: A (float) score or a T-length list of such scores if 
            multitask=True and reduce=None
        """
        Y_p = self.predict(X, **kwargs)
        if self.multitask:
            self._check(Y, typ=list)
            self._check(Y_p, typ=list)
            self._check(Y[0], typ=torch.Tensor)
            self._check(Y_p[0], typ=torch.Tensor)
        else:
            self._check(Y, typ=torch.Tensor)
            self._check(Y_p, typ=torch.Tensor)
            Y = [Y]
            Y_p = [Y_p]

        task_scores = []
        for t, Y_tp in enumerate(Y_p):
            score = metric_score(Y[t], Y_tp, metric, ignore_in_gold=[0], 
                **kwargs)
            task_scores.append(score)

        # TODO: Other options for reduce, including scoring only certain
        # primary tasks, and converting to end labels using TaskGraph...
        if not self.multitask:
            score = task_scores[0]
        else:
            if reduce is None:
                score = task_scores
            if reduce == 'mean':
                score = np.mean(task_scores)
            else:
                raise Exception(f"Keyword reduce='{reduce}' not recognized.")

        if verbose:
            if self.multitask and reduce is None:
                for t, score_t in enumerate(score):
                    print(f"{metric.capitalize()} (t={t}): {score:.3f}")
            else:
                print(f"{metric.capitalize()}: {score:.3f}")

        return score

    @multitask_decorator    
    def predict(self, X, break_ties='random', **kwargs):
        """Predicts hard (int) labels for an input X on all tasks
        
        Args:
            X: The input for the predict_proba method
            Y: A T-length list of [N] or [N, 1] tensors of gold labels in
                {1,...,K_t}
            break_ties: A tie-breaking policy
            as_list: If True, return the result as a list regardless of whether
                multitask=True

        Returns:
            An N-dim tensor of predictions or a T-length list of such
            tensors if multitask=True
        """
        Y_p = self.predict_proba(X, **kwargs)
        if self.multitask:
            self._check(Y_p, typ=list)
            self._check(Y_p[0], typ=torch.Tensor)
        else:
            self._check(Y_p, typ=torch.Tensor)
            Y_p = [Y_p]

        Y_ph = []
        for Y_tp in Y_p:
            Y_tph = self._break_ties(Y_tp.numpy(), break_ties)
            Y_ph.append(torch.tensor(Y_tph, dtype=torch.short))

        return Y_ph

    @multitask_decorator
    def predict_proba(self, X, **kwargs):
        """Predicts soft probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            An [N, K_t] tensor of soft predictions or a T-length list of such
            tensors if multitask=True
        """
        raise NotImplementedError

    def score_task(self, X, Y, t=0, metric='accuracy', verbose=True, **kwargs):
        """Score the predictive performance of the Classifier on task t
        
        Args:
            X: The input for the predict_task method
            Y: A [N] or [N, 1] tensor of gold labels in {1,...,K_t}
            t: The task index to score
            metric: The metric with which to score performance on this task
        Returns:
            The (float) score of the Classifier for the specified task and 
            metric
        """
        self._check(Y, typ=torch.Tensor)

        Y_tp = self.predict_task(X, t=t, **kwargs)
        score = metric_score(Y[t], Y_tp, metric, ignore_in_gold=[0], **kwargs)
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
        Y_tph = self._break_ties(Y_tp.numpy(), break_ties)
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

    def confusion(self, X, Y, **kwargs):
        # TODO: implement this here
        raise NotImplementedError
    
    def error_analysis(self, session, X, Y):
        # TODO: implement this here
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    
    def _break_ties(self, Y_ts, break_ties='random'):
        """Break ties in each row of a tensor according to the specified policy

        Args:
            Y_ts: An [N, K_t] numpy array of probabilities
            break_ties: A tie-breaking policy:
                'random': randomly choose among the tied options
                'abstain': return an abstain vote (0)
        """
        N, k = Y_ts.shape
        Y_th = np.zeros(N)
        diffs = np.abs(Y_ts - Y_ts.max(axis=1).reshape(-1, 1))

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