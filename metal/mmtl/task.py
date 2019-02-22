from abc import ABC
from functools import partial

import torch
import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.scorer import Scorer


class Task(ABC):
    """A abstract class for tasks in MMTL Metal Model.

    Args:
        name: (str) The name of the task
            TODO: replace this with a more fully-featured path through the network
        input_module: (nn.Module) The input module
        middle_module: (nn.Module) A middle module
        head_module: (nn.Module) The task head module
        data_loaders: A dict of DataLoaders to feed through the network
            Each key in data.keys() should be in ["train", "valid", "test"]
            The DataLoaders should return batches of (X, Ys) pairs, where X[0] returns
                a complete input for feeding into the input_module, and Ys is a tuple
                containing S label sets, such that Y[0][0] is the appropriate label(s)
                to pass into the loss head for the first example and first label set.
        task_names: an [S] list of task names corresponding to the S label sets
        scorer: A Scorer that returns a metrics_dict object.
        loss_hat_func: A function of the form f(forward(X), Y) -> loss (scalar Tensor)
            We recommend returning an average loss per example so that loss magnitude
            is more consistent in the face of batch size changes
        output_hat_func: A function of the form f(forward(X)) -> output (e.g. probs)
    """

    def __init__(
        self,
        name,
        data_loaders,
        input_module,
        middle_module,
        head_module,
        loss_hat_func,
        output_hat_func,
        scorer,
        task_names=None,
    ) -> None:
        self.name = name
        self.data_loaders = data_loaders
        self.input_module = input_module
        self.middle_module = middle_module
        self.head_module = head_module
        self.loss_hat_func = loss_hat_func
        self.output_hat_func = output_hat_func
        self.scorer = scorer
        # TODO: get the task_names from the Payload, not the Task
        self.task_names = tuple(task_names) if task_names is not None else (name,)

    def __repr__(self):
        return (
            f"{self.__name__}(name={self.name}, task_names={','.join(self.task_names)})"
        )


class ClassificationTask(Task):
    """A classification task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        data_loaders,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        loss_hat_func=(lambda out, Y_gold: F.cross_entropy(out, Y_gold - 1)),
        output_hat_func=(partial(F.softmax, dim=1)),
        scorer=Scorer(standard_metrics=["accuracy"]),
        task_names=None,
    ) -> None:

        super(ClassificationTask, self).__init__(
            name,
            data_loaders,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
            task_names,
        )


class RegressionTask(Task):
    """A regression task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        data_loaders,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        # TODO: (@JD): fix this with auxiliary -- removed Y_gold[.float()] for fp16
        loss_hat_func=(lambda out, Y_gold: F.mse_loss(torch.sigmoid(out), Y_gold)),
        output_hat_func=(torch.sigmoid),
        scorer=Scorer(standard_metrics=[]),
        task_names=None,
    ) -> None:

        super(RegressionTask, self).__init__(
            name,
            data_loaders,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
            task_names,
        )


def tokenwise_ce_loss(self, out, Y_gold):
    """Compute the batch- and token-averaged cross-entropy loss

    NOTE:
    First calculate token-averaged loss per example
    Then average these losses over the batch

    We assume the standard MeTaL convention of no 0 labels in Y_gold
    """
    # TEMP
    assert not sum(Y_gold == 0)
    # TEMP
    logits, attention_mask = out
    batch_size, seq_len, num_classes = logits.shape
    active_loss = attention_mask.view(-1) == 1
    active_logits = logits.view(-1, num_classes)[active_loss]
    active_labels = Y_gold.view(-1)[active_loss]
    loss = F.cross_entropy(active_logits, active_labels - 1, reduction="none")
    import ipdb

    ipdb.set_trace()
    # reshape loss for averaging right
    return loss


def tokenwise_accuracy(self, Y, Y_preds, Y_probs):
    raise NotImplementedError
    return {}


class TokenClassificationTask(Task):
    """A single task for predicting a class for multiple tokens (e.g., POS tagging)

    Assumed i/o of head_module:
        (sequence_output, attention_mask) -> (logits, attention_mask)
        logits: [batch_size, seq_len, num_classes]
    """

    def __init__(
        self,
        name,
        data_loaders,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        loss_hat_func=tokenwise_ce_loss,
        output_hat_func=(partial(F.softmax, dim=1)),
        scorer=Scorer(custom_metric_funcs={tokenwise_accuracy: ["token_acc"]}),
        task_names=None,
    ) -> None:

        super().__init__(
            name,
            data_loaders,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
            task_names,
        )
