import numpy as np
import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.scorer import Scorer
from metal.mmtl.task import Task


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
