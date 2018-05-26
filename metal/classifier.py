import numpy as np
import torch
import torch.nn as nn

from metal.metrics import metric_score, confusion_matrix

class Classifier(nn.Module):
    """Simple abstract base class for a probabilistic classifier."""

    def __init__(self):
        super().__init__()

    def train(self, X, Y, **kwargs):
        raise NotImplementedError

    def predict_proba(self, X, t=0, **kwargs):
        """Returns an [N, K_t] tensor of soft (float) predictions for task t."""
        raise NotImplementedError

    def predict(self, X, t=0, break_ties='random'):
        """Returns an N-dim tensor of hard (int) predictions for task t."""
        Y_ts = self.predict_proba(X, t=t).numpy()

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
        
        return torch.tensor(Y_th, dtype=torch.short)

    def predict_tasks_proba(self, X):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [self.predict_proba(X, t=t) for t in range(self.T)]

    def predict_tasks(self, X, break_ties='random'):
        """Returns a list of T [N, K_t] tensors of hard (int) predictions."""
        return [self.predict(X, t=t, break_ties=break_ties) for t in range(self.T)]

    def score(self, X, Y, metric='accuracy', verbose=True, **kwargs):
        Y_p = self.predict(X, **kwargs)
        score = metric_score(Y, Y_p, metric, ignore_in_gold=[0], **kwargs)
        if verbose:
            print(f"{metric.capitalize()}: {score:.3f}")
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