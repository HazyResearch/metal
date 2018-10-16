import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from metal.analysis import confusion_matrix
from metal.metrics import metric_score
from metal.utils import Checkpointer, place_on_gpu, recursive_merge_dicts


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

    Args:
        k: (int) The cardinality of the classifier
        seed: (int) A random seed to set
    """

    # A class variable indicating whether the class implements its own custom L2
    # regularization (True) or not (False); in the latter case, generic L2 in
    # the optimizer is used
    implements_l2 = False

    def __init__(self, k, config):
        super().__init__()
        self.config = config
        self.multitask = False
        self.k = k

        # Set random seed
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        self._set_seed(self.config["seed"])

        # Confirm that cuda is available if config is using CUDA
        if self.config["use_cuda"] and not torch.cuda.is_available():
            raise ValueError("use_cuda=True but CUDA not available.")

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

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def save(self, destination=None):
        """Serialize and save a model.

        If destination is a filepath, write to file.
        If destination is None, return a bytes object.

        Example:
            end_model = EndModel(...)
            end_model.train(...)
            end_model.save('my_end_model.pkl')
        """
        if destination is None:
            return pickle.dumps(self)
        else:
            with open(destination, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(source=None):
        """Deserialize and load a model.

        If source is a filepath, load from file.
        If source is a bytes object, load from bytes.

        Example:
            end_model = EndModel.load('my_end_model.pkl')
            end_model.score(...)
        """
        if isinstance(source, bytes):
            return pickle.loads(source)
        else:
            with open(source, "rb") as f:
                return pickle.load(f)

    def reset(self):
        """Initializes all modules in a network"""
        # The apply(f) method recursively calls f on itself and all children
        self.apply(self._reset_module)

    def train(self, *args, **kwargs):
        """Trains a classifier

        Take care to initialize weights outside the training loop and zero out
        gradients at the beginning of each iteration inside the loop.
        """
        raise NotImplementedError

    def _create_checkpointer(self, checkpoint_config):
        model_class = type(self).__name__
        return Checkpointer(
            model_class, **checkpoint_config, verbose=self.config["verbose"]
        )

    def _train(self, train_data, loss_fn, dev_data=None):
        """The internal training routine called by train() after initial setup

        Args:
            train_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the train split
            loss_fn: the loss function to minimize (maps *data -> loss)
            dev_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the dev split

        If dev_data is not provided, then no checkpointing or
        evaluation on the dev set will occur.
        """
        train_config = self.config["train_config"]
        evaluate_dev = dev_data is not None

        # Convert data to DataLoaders
        train_loader = self._create_data_loader(train_data)
        dev_loader = self._create_data_loader(dev_data)

        # Set the optimizer
        optimizer = self._set_optimizer(train_config)

        # Set the lr scheduler
        scheduler_config = train_config["scheduler_config"]
        lr_scheduler = self._set_scheduler(scheduler_config, optimizer)

        # Create the checkpointer if applicable
        if evaluate_dev and train_config["checkpoint"]:
            checkpointer = self._create_checkpointer(
                train_config["checkpoint_config"]
            )

        # Moving model to GPU
        if self.config["use_cuda"]:
            if self.config["verbose"]:
                print("Using GPU...")
            self.cuda()

        # Train the model
        for epoch in range(train_config["n_epochs"]):
            epoch_loss = 0.0
            for batch_num, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                disable=(
                    train_config["disable_prog_bar"]
                    or not self.config["verbose"]
                ),
            ):

                # Moving data to GPU
                if self.config["use_cuda"]:
                    data = place_on_gpu(data)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # import pdb; pdb.set_trace()
                # Forward pass to calculate outputs
                loss = loss_fn(*data)
                if torch.isnan(loss):
                    msg = "Loss is NaN. Consider reducing learning rate."
                    raise Exception(msg)

                # Backward pass to calculate gradients
                loss.backward()

                # TODO: restore this once it has unit tests
                # Clip gradients
                # if grad_clip:
                #     torch.nn.utils.clip_grad_norm(
                #        self.net.parameters(), grad_clip)

                # Perform optimizer step
                optimizer.step()

                # Keep running sum of losses
                epoch_loss += loss.detach()

            # Calculate average loss per training example
            # Saving division until this stage protects against the potential
            # mistake of averaging batch losses when the last batch is an orphan
            train_loss = epoch_loss / len(train_loader.dataset)

            # Checkpoint performance on dev
            if evaluate_dev and (epoch % train_config["validation_freq"] == 0):
                val_metric = train_config["validation_metric"]
                dev_score = self.score(
                    dev_loader,
                    metric=val_metric,
                    verbose=False,
                    print_confusion_matrix=False,
                )

                if train_config["checkpoint"]:
                    checkpointer.checkpoint(self, epoch, dev_score)

            # Apply learning rate scheduler
            if (
                lr_scheduler is not None
                and epoch + 1 >= scheduler_config["lr_freeze"]
            ):
                if scheduler_config["scheduler"] == "reduce_on_plateau":
                    if evaluate_dev:
                        lr_scheduler.step(dev_score)
                else:
                    lr_scheduler.step()

            # Report progress
            if self.config["verbose"] and (
                epoch % train_config["print_every"] == 0
                or epoch == train_config["n_epochs"] - 1
            ):
                msg = f"[E:{epoch}]\tTrain Loss: {train_loss:.3f}"
                if evaluate_dev:
                    msg += f"\tDev score: {dev_score:.3f}"
                print(msg)

        # Restore best model if applicable
        if evaluate_dev and train_config["checkpoint"]:
            checkpointer.restore(model=self)

        # Print confusion matrix if applicable
        if self.config["verbose"]:
            print("Finished Training")
            if evaluate_dev:
                self.score(
                    dev_loader,
                    metric=["accuracy"],
                    verbose=True,
                    print_confusion_matrix=True,
                )

    def _create_dataset(self, *data):
        """Converts input data to the appropriate Dataset"""
        # Make sure data is a tuple of dense tensors
        data = [self._to_torch(x, dtype=torch.FloatTensor) for x in data]
        return TensorDataset(*data)

    def _create_data_loader(self, data, **kwargs):
        """Converts input data into a DataLoader"""
        if data is None:
            return None

        # Set DataLoader config
        # NOTE: Not applicable if data is already a DataLoader
        config = {
            **self.config["train_config"]["data_loader_config"],
            **kwargs,
            "pin_memory": self.config["use_cuda"],
        }

        # Return data as DataLoader
        if isinstance(data, (tuple, list)):
            return DataLoader(self._create_dataset(*data), **config)
        elif isinstance(data, Dataset):
            return DataLoader(data, **config)
        elif isinstance(data, DataLoader):
            return data
        else:
            raise ValueError("Input data type not recognized.")

    def _set_optimizer(self, train_config):
        optimizer_config = train_config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        # We set L2 here if the class does not implement its own L2 reg
        l2 = 0 if self.implements_l2 else train_config.get("l2", 0)

        if opt == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                **optimizer_config["optimizer_common"],
                **optimizer_config["sgd_config"],
                weight_decay=l2,
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
                weight_decay=l2,
            )
        else:
            raise ValueError(f"Did not recognize optimizer option '{opt}''")
        return optimizer

    def _set_scheduler(self, scheduler_config, optimizer):
        scheduler = scheduler_config["scheduler"]
        if scheduler is None:
            lr_scheduler = None
        elif scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **scheduler_config["exponential_config"]
            )
        elif scheduler == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_config["plateau_config"]
            )
        else:
            raise ValueError(
                f"Did not recognize scheduler option '{scheduler}''"
            )
        return lr_scheduler

    def score(
        self,
        data,
        metric=["accuracy"],
        break_ties="random",
        verbose=True,
        print_confusion_matrix=True,
        **kwargs,
    ):
        """Scores the predictive performance of the Classifier on all tasks

        Args:
            data: a Pytorch DataLoader, Dataset, or tuple with Tensors (X,Y):
                X: The input for the predict method
                Y: An [n] or [n, 1] torch.Tensor or np.ndarray of target labels
                    in {1,...,k}
            metric: A metric (string) with which to score performance or a
                list of such metrics
            break_ties: A tie-breaking policy (see Classifier._break_ties())
            verbose: The verbosity for just this score method; it will not
                update the class config.
            print_confusion_matrix: Print confusion matrix

        Returns:
            scores: A (float) score or a list of such scores if kwarg metric
                is a list
        """
        Y_p, Y = self._get_predictions(data, break_ties=break_ties, **kwargs)

        # Evaluate on the specified metrics
        metric_list = metric if isinstance(metric, list) else [metric]
        scores = []
        for metric in metric_list:
            score = metric_score(Y, Y_p, metric, ignore_in_gold=[0])
            scores.append(score)
            if verbose:
                print(f"{metric.capitalize()}: {score:.3f}")

        # Optionally print confusion matrix
        if print_confusion_matrix:
            confusion_matrix(Y_p, Y, pretty_print=True)

        if isinstance(scores, list) and len(scores) == 1:
            return scores[0]
        else:
            return scores

    def _get_predictions(self, data, break_ties="random", **kwargs):
        """Computes predictions in batch, given a labeled dataset

        Args:
            data: a Pytorch DataLoader, Dataset, or tuple with Tensors (X,Y):
                X: The input for the predict method
                Y: An [n] or [n, 1] torch.Tensor or np.ndarray of target labels
                    in {1,...,k}
            break_ties: How to break ties when making predictions

        Returns:
            Y_p: A Tensor of predictions
            Y: A Tensor of labels
        """
        data_loader = self._create_data_loader(data)
        Y_p = []
        Y = []

        # Do batch evaluation by default, getting the predictions and labels
        for batch_num, data in enumerate(data_loader):
            Xb, Yb = data
            Y.append(self._to_numpy(Yb))

            # Optionally move to GPU
            if self.config["use_cuda"]:
                Xb = place_on_gpu(Xb)

            # Append predictions and labels from DataLoader
            Y_p.append(
                self._to_numpy(
                    self.predict(Xb, break_ties=break_ties, **kwargs)
                )
            )
        Y_p = np.hstack(Y_p)
        Y = np.hstack(Y)
        return Y_p, Y

    def predict(self, X, break_ties="random", **kwargs):
        """Predicts hard (int) labels for an input X on all tasks

        Args:
            X: The input for the predict_proba method
            break_ties: A tie-breaking policy (see Classifier._break_ties())

        Returns:
            An n-dim np.ndarray of predictions in {1,...k}
        """
        Y_p = self._to_numpy(self.predict_proba(X, **kwargs))
        return self._break_ties(Y_p, break_ties).astype(np.int)

    def predict_proba(self, X, **kwargs):
        """Predicts soft probabilistic labels for an input X on all tasks
        Args:
            X: An appropriate input for the child class of Classifier
        Returns:
            An [n, k] np.ndarray of soft predictions
        """
        raise NotImplementedError

    def _break_ties(self, Y_s, break_ties="random"):
        """Break ties in each row of a tensor according to the specified policy

        Args:
            Y_s: An [n, k] np.ndarray of probabilities
            break_ties: A tie-breaking policy:
                'abstain': return an abstain vote (0)
                'random': randomly choose among the tied options
                    NOTE: if break_ties='random', repeated runs may have
                    slightly different results due to difference in broken ties
        """
        n, k = Y_s.shape
        Y_h = np.zeros(n)
        diffs = np.abs(Y_s - Y_s.max(axis=1).reshape(-1, 1))

        TOL = 1e-5
        for i in range(n):
            max_idxs = np.where(diffs[i, :] < TOL)[0]
            if len(max_idxs) == 1:
                Y_h[i] = max_idxs[0] + 1
            # Deal with 'tie votes' according to the specified policy
            elif break_ties == "random":
                Y_h[i] = np.random.choice(max_idxs) + 1
            elif break_ties == "abstain":
                Y_h[i] = 0
            else:
                ValueError(f"break_ties={break_ties} policy not recognized.")
        return Y_h

    @staticmethod
    def _to_numpy(Z):
        """Converts a None, list, np.ndarray, or torch.Tensor to np.ndarray;
        also handles converting sparse input to dense."""
        if Z is None:
            return Z
        elif issparse(Z):
            return Z.toarray()
        elif isinstance(Z, np.ndarray):
            return Z
        elif isinstance(Z, list):
            return np.array(Z)
        elif isinstance(Z, torch.Tensor):
            return Z.cpu().numpy()
        else:
            msg = (
                f"Expected None, list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead."
            )
            raise Exception(msg)

    @staticmethod
    def _to_torch(Z, dtype=None):
        """Converts a None, list, np.ndarray, or torch.Tensor to torch.Tensor;
        also handles converting sparse input to dense."""
        if Z is None:
            return None
        elif issparse(Z):
            Z = torch.from_numpy(Z.toarray())
        elif isinstance(Z, torch.Tensor):
            pass
        elif isinstance(Z, list):
            Z = torch.from_numpy(np.array(Z))
        elif isinstance(Z, np.ndarray):
            Z = torch.from_numpy(Z)
        else:
            msg = (
                f"Expected list, numpy.ndarray or torch.Tensor, "
                f"got {type(Z)} instead."
            )
            raise Exception(msg)

        return Z.type(dtype) if dtype else Z

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
