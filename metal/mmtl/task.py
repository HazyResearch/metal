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
        attention_module: (nn.Module) An attention module right before the task head
        head_module: (nn.Module) The task head module
        output_hat_func: A function of the form f(forward(X)) -> output (e.g. probs)
        loss_hat_func: A function of the form f(forward(X), Y) -> loss (scalar Tensor)
            We recommend returning an average loss per example so that loss magnitude
            is more consistent in the face of batch size changes
        loss_multiplier: A scalar by which the loss for this task will be multiplied.
            Default is 1 (no scaling effect at all)
        scorer: A Scorer that returns a metrics_dict object.
    """

    def __init__(
        self,
        name,
        input_module,
        middle_module,
        attention_module,
        head_module,
        output_hat_func,
        loss_hat_func,
        loss_multiplier,
        scorer,
    ) -> None:
        self.name = name
        self.input_module = input_module
        self.middle_module = middle_module
        self.attention_module = attention_module
        self.head_module = head_module
        self.output_hat_func = output_hat_func
        self.loss_hat_func = loss_hat_func
        self.loss_multiplier = loss_multiplier
        self.scorer = scorer

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name}, loss_multiplier={self.loss_multiplier})"


class ClassificationTask(Task):
    """A classification task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        attention_module=IdentityModule(),
        head_module=IdentityModule(),
        output_hat_func=(partial(F.softmax, dim=1)),
        loss_hat_func=(
            lambda Y_out, Y_gold: F.cross_entropy(Y_out, Y_gold.view(-1) - 1)
        ),
        loss_multiplier=1.0,
        scorer=Scorer(standard_metrics=["accuracy"]),
    ) -> None:

        super(ClassificationTask, self).__init__(
            name,
            input_module,
            middle_module,
            attention_module,
            head_module,
            output_hat_func,
            loss_hat_func,
            loss_multiplier,
            scorer,
        )


class RegressionTask(Task):
    """A regression task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        input_module=IdentityModule(),
        middle_module=IdentityModule(),
        attention_module=IdentityModule(),
        head_module=IdentityModule(),
        output_hat_func=lambda x: x,
        # w/o sigmoid (target labels can be in any range)
        loss_hat_func=(
            lambda Y_out, Y_gold: F.mse_loss(Y_out.view(-1), Y_gold.view(-1))
        ),
        loss_multiplier=1.0,
        scorer=Scorer(standard_metrics=[]),
    ) -> None:

        super(RegressionTask, self).__init__(
            name,
            input_module,
            middle_module,
            attention_module,
            head_module,
            output_hat_func,
            loss_hat_func,
            loss_multiplier,
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
    return {"token_acc": float(np.mean(accs))}


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
        attention_module=IdentityModule(),
        head_module=IdentityModule(),
        output_hat_func=tokenwise_softmax,
        loss_hat_func=tokenwise_ce_loss,
        loss_multiplier=1.0,
        scorer=Scorer(custom_metric_funcs={tokenwise_accuracy: ["token_acc"]}),
    ) -> None:

        super().__init__(
            name,
            input_module,
            middle_module,
            attention_module,
            head_module,
            output_hat_func,
            loss_hat_func,
            loss_multiplier,
            scorer,
        )
