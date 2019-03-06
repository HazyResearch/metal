from abc import ABC
from functools import partial

import numpy as np
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
        loss_hat_func: A function of the form f(forward(X), Y) -> loss (scalar Tensor)
            We recommend returning an average loss per example so that loss magnitude
            is more consistent in the face of batch size changes
        output_hat_func: A function of the form f(forward(X)) -> output (e.g. probs)
        scorer: A Scorer that returns a metrics_dict object.
    """

    def __init__(
        self,
        name,
        input_module,
        middle_module,
        head_module,
        loss_hat_func,
        output_hat_func,
        scorer,
    ) -> None:
        self.name = name
        self.input_module = input_module
        self.middle_module = middle_module
        self.head_module = head_module
        self.loss_hat_func = loss_hat_func
        self.output_hat_func = output_hat_func
        self.scorer = scorer

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"


class ClassificationTask(Task):
    """A classification task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        loss_hat_func=(lambda out, Y_gold: F.cross_entropy(out, Y_gold - 1)),
        output_hat_func=(partial(F.softmax, dim=1)),
        scorer=Scorer(standard_metrics=["accuracy"]),
    ) -> None:

        super(ClassificationTask, self).__init__(
            name,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
        )


class RegressionTask(Task):
    """A regression task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        loss_hat_func=(lambda out, Y_gold: F.mse_loss(torch.sigmoid(out), Y_gold)),
        output_hat_func=(torch.sigmoid),
        scorer=Scorer(standard_metrics=[]),
    ) -> None:

        super(RegressionTask, self).__init__(
            name,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
        )


def tokenwise_ce_loss(out, Y_gold):
    """Compute the token-averaged cross-entropy loss

    We assume the standard MeTaL convention of no 0 labels in Y_gold
    """
    logits, attention_mask = out
    batch_size, seq_len, num_classes = logits.shape
    active = attention_mask.view(-1) == 1
    active_logits = logits.view(-1, num_classes)[active]
    active_labels = Y_gold.view(-1)[active]
    return F.cross_entropy(active_logits, active_labels - 1, reduction="mean")


def tokenwise_softmax(out):
    """Compute the token-wise class probabilities for each token

    Args:
        out: the output of task head
    Returns:
        probs: [batch_size] list of [seq_len, num_classes] probabilities
            Note that seq_len may vary by instance after this step (padding is removed)
    """
    logits, masks = out
    batch_size, seq_len, num_classes = logits.shape
    probs = F.softmax(logits, dim=2)
    return [probs_matrix[mask == 1] for probs_matrix, mask in zip(probs, masks)]


def tokenwise_accuracy(gold, preds, probs=None):
    """Compute the average token-wise accuracy per example"""
    # HACK: Most unfortunately, incoming gold is padded whereas preds are not
    # For now we just drop the padding on the end by looking up the length of the preds
    # Longer-term, find a more intuitive hard and fast rule for when Y will be padded
    accs = []
    for y, y_preds in zip(gold, preds):
        acc = np.mean(y[: len(y_preds)] == y_preds)
        accs.append(acc)
    return {"token_acc": np.mean(accs)}


class TokenClassificationTask(Task):
    """A single task for predicting a class for multiple tokens (e.g., POS tagging)

    Assumed i/o of head_module:
        (sequence_output, attention_mask) -> (logits, attention_mask)
        logits: [batch_size, seq_len, num_classes]
    """

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        head_module=IdentityModule(),
        loss_hat_func=tokenwise_ce_loss,
        output_hat_func=tokenwise_softmax,
        scorer=Scorer(custom_metric_funcs={tokenwise_accuracy: ["token_acc"]}),
    ) -> None:

        super().__init__(
            name,
            input_module,
            middle_module,
            head_module,
            loss_hat_func,
            output_hat_func,
            scorer,
        )
