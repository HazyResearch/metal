import numpy as np
import torch.nn as nn

from metal.metrics import metric_score, confusion_matrix

class Classifier(nn.Module):
    """Simple abstract base class for a probabilistic classifier."""

    def __init__(self, cardinality=2, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.cardinality = cardinality

    def train(self, X, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def score(self, X, Y, metric='accuracy', reduce_tasks='mean', verbose=True, 
        **kwargs):
        """Scores the predictive performance of the Classifier

        Args:
            X: A T-length list of N x ? data matrices for each task
            Y: A T x N matrix of gold labels in {1,...,K_t}
            metric: The metric to with which to score performance on each task
            reduce_tasks: How to reduce the scores of multiple tasks; the
                default is to take the mean score across tasks
        """
        task_scores = [
            self.score_task(X[t], Y[t], t=t, metric=metric,
                verbose=(verbose and self.T > 1), **kwargs)
            for t in range(self.T)
        ]

        # TODO: Other options for reduce_tasks, including scoring only certain
        # primary tasks, and converting to end labels using TaskGraph...
        if reduce_tasks == 'mean':
            score = np.mean(task_scores)
        else:
            raise Exception(f"Reduce_tasks='{reduce_tasks}' not recognized.")

        if verbose:
            print(f"{metric.capitalize()}: {score:.3f}")
        return score

    def score_task(self, X, Y, t=0, metric='accuracy', verbose=True, **kwargs):
        Y_p = self.predict(X, **kwargs)
        score = metric_score(Y, Y_p, metric, ignore_in_gold=[0], **kwargs)
        if verbose:
            print(f"[t={t}] {metric.capitalize()}: {score:.3f}")
        return score
 
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
    
    def _check_or_set_attr(self, name, val, set_val=False):
        if set_val:
            setattr(self, name, val)
        else:
            true_val = getattr(self, name)
            if val != true_val:
                raise Exception(f"{name} = {val}, but should be {true_val}.")