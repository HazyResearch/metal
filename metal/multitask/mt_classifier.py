import numpy as np

from metal.classifier import Classifier
from metal.metrics import metric_score
from metal.multitask.utils import MultiXYDataset, MultiYDataset


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

    Args:
        K: (list) A t-length list of cardinalities (ints) for each task
    """

    def __init__(self, K, config):
        Classifier.__init__(self, None, config)
        self.multitask = True
        self.K = K

    def predict_proba(self, X, **kwargs):
        """Predicts probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            A t-length list of [n, K_t] np.ndarrays of probabilistic labels
        """
        raise NotImplementedError

    def predict(self, X, break_ties="random", return_probs=False, **kwargs):
        """Predicts int labels for an input X on all tasks

        Args:
            X: The input for the predict_proba method
            break_ties: A tie-breaking policy
            return_probs: Return the predicted probabilities as well

        Returns:
            Y_p: A t-length list of n-dim np.ndarrays of predictions in [1, K_t]
            [Optionally: Y_s: A t-length list of [n, K_t] np.ndarrays of
                predicted probabilities]
        """
        Y_s = self.predict_proba(X, **kwargs)
        self._check(Y_s, typ=list)
        self._check(Y_s[0], typ=np.ndarray)

        Y_p = []
        for Y_ts in Y_s:
            Y_tp = self._break_ties(Y_ts, break_ties)
            Y_p.append(Y_tp.astype(np.int))

        if return_probs:
            return Y_p, Y_s
        else:
            return Y_p

    def score(
        self,
        data,
        metric="accuracy",
        validation_task=None,
        reduce="mean",
        break_ties="random",
        verbose=True,
        print_confusion_matrix=False,
        **kwargs,
    ):
        """Scores the predictive performance of the Classifier on all tasks
        Args:
            data: either a Pytorch Dataset, DataLoader or tuple supplying (X,Y):
                X: The input for the predict method
                Y: A t-length list of [n] or [n, 1] np.ndarrays or
                   torch.Tensors of gold labels in {1,...,K_t}
            metric: The metric with which to score performance on each task
            validation_task:
                int: returns score for specific task number.
            reduce: How to reduce the scores of multiple tasks:
                 None : return a t-length list of scores
                'mean': return the mean score across tasks
            break_ties: How to break ties when making predictions
        Returns:
            scores: A (float) score or a t-length list of such scores if
                reduce=None
        """
        Y_p, Y, Y_s = self._get_predictions(
            data, break_ties=break_ties, return_probs=True, **kwargs
        )

        # TODO: Handle multiple metrics...
        metric_list = metric if isinstance(metric, list) else [metric]
        if len(metric_list) > 1:
            raise NotImplementedError(
                "Multiple metrics for multi-task score() not yet supported."
            )
        metric = metric_list[0]

        # Return score for task t only.
        if validation_task is not None:
            score = metric_score(
                Y[validation_task],
                Y_p[validation_task],
                metric,
                probs=Y_s[validation_task],
                ignore_in_gold=[0],
            )
            if verbose:
                print(f"{metric.capitalize()}: {score:.3f}")
            return score

        task_scores = []
        for t, Y_tp in enumerate(Y_p):
            score = metric_score(Y[t], Y_tp, metric, probs=Y_s[t], ignore_in_gold=[0])
            task_scores.append(score)

        # TODO: Other options for reduce, including scoring only certain
        # primary tasks, and converting to end labels using TaskGraph...
        if reduce is None:
            score = task_scores
        elif reduce == "mean":
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

    def score_task(self, X, Y, t=0, metric="accuracy", verbose=True, **kwargs):
        """Scores the predictive performance of the Classifier on task t

        Args:
            X: The input for the predict_task method
            Y: A [n] or [n, 1] np.ndarray or torch.Tensor of gold labels in
                {1,...,K_t}
            t: The task index to score
            metric: The metric with which to score performance on this task
        Returns:
            The (float) score of the Classifier for the specified task and
            metric
        """
        Y = self._to_numpy(Y)
        Y_tp = self.predict_task(X, t=t, **kwargs)
        probs = self.predict_proba(X)[t]
        score = metric_score(
            Y[t], Y_tp, metric, ignore_in_gold=[0], probs=probs, **kwargs
        )
        if verbose:
            print(f"[t={t}] {metric.capitalize()}: {score:.3f}")
        return score

    def predict_task(self, X, t=0, break_ties="random", **kwargs):
        """Predicts int labels for an input X on task t

        Args:
            X: The input for the predict_task_proba method
            t: The task index to predict
        Returns:
            An n-dim tensor of int predictions for the specified task
        """
        Y_tp = self.predict_task_proba(X, t=t, **kwargs)
        Y_tph = self._break_ties(Y_tp, break_ties)
        return Y_tph

    def predict_task_proba(self, X, t=0, **kwargs):
        """Predicts probabilistic labels for an input X on task t

        Args:
            X: The input for the predict_proba method
            t: The task index to predict for which to predict probabilities
        Returns:
            An [n, K_t] tensor of predictions for task t
        NOTE: By default, this method calls predict_proba and extracts element
        t. If it is possible to predict individual tasks in isolation, however,
        this method may be overriden for efficiency's sake.
        """
        return self.predict_proba(X, **kwargs)[t]

    def _create_dataset(self, *data):
        X, Y = data
        if isinstance(X, list):
            return MultiXYDataset(X, Y)
        else:
            return MultiYDataset(X, Y)

    @staticmethod
    def _to_torch(Z, dtype=None):
        """Converts a None, list, np.ndarray, or torch.Tensor to torch.Tensor"""
        if isinstance(Z, list):
            return [Classifier._to_torch(z, dtype=dtype) for z in Z]
        else:
            return Classifier._to_torch(Z)

    @staticmethod
    def _to_numpy(Z):
        """Converts a None, list, np.ndarray, or torch.Tensor to np.ndarray"""
        if isinstance(Z, list):
            return [Classifier._to_numpy(z) for z in Z]
        else:
            return Classifier._to_numpy(Z)

    @staticmethod
    def _stack_batches(X):
        """Given a list of batches, each consisting of a T-len list of
        np.ndarrays, stack along the first (batch) axis, returning a T-len list
        of np.ndarrays."""
        return [Classifier._stack_batches(Xt) for Xt in zip(*X)]
