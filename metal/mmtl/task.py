from abc import ABC
from functools import partial

import torch
import torch.nn.functional as F

from metal.mmtl.scorer import Scorer


class Task(ABC):
    """A abstract class for tasks in MMTL Metal Model.

    Args:
        name: (str) The name of the task
            TODO: replace this with a more fully-featured path through the network
        input_module: (nn.Module) The input module
        attention_module: The attention module between the input and task head
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
        head_module,
        scorer,
        loss_hat_func,
        output_hat_func,
        task_names=None,
        attention_module=None,
    ) -> None:
        self.name = name
        self.data_loaders = data_loaders
        self.input_module = input_module
        self.attention_module = attention_module
        self.head_module = head_module
        self.scorer = scorer
        self.loss_hat_func = loss_hat_func
        self.output_hat_func = output_hat_func
        # TODO: get the task_names from the Payload, not the Task
        self.task_names = tuple(task_names) if task_names is not None else (name,)


class ClassificationTask(Task):
    """A classification task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        data_loaders,
        input_module,
        head_module,
        scorer=Scorer(standard_metrics=["accuracy"]),
        loss_hat_func=(lambda Y_prob, Y_gold: F.cross_entropy(Y_prob, Y_gold - 1)),
        output_hat_func=(partial(F.softmax, dim=1)),
        task_names=None,
        attention_module=None,
    ) -> None:

        super(ClassificationTask, self).__init__(
            name,
            data_loaders,
            input_module,
            head_module,
            scorer,
            loss_hat_func,
            output_hat_func,
            task_names,
            attention_module,
        )


class RegressionTask(Task):
    """A regression task for use in an MMTL MetalModel"""

    def __init__(
        self,
        name,
        data_loaders,
        input_module,
        head_module,
        scorer=Scorer(standard_metrics=[]),
        # TODO: (@JD): fix this with auxiliary -- removed Y_gold[.float()] for fp16
        loss_hat_func=(
            lambda Y_prob, Y_gold: F.mse_loss(Y_prob, Y_gold)
            # lambda Y_prob, Y_gold: F.mse_loss(torch.sigmoid(Y_prob), Y_gold)
        ),
        output_hat_func=lambda x: (
            torch.gt(x, 0).type(x.dtype) * torch.gt(-x, -1).type(x.dtype) * x
        )
        + torch.gt(-x, -1).type(x.dtype),
        task_names=None,
        attention_module=None,
    ) -> None:

        super(RegressionTask, self).__init__(
            name,
            data_loaders,
            input_module,
            head_module,
            scorer,
            loss_hat_func,
            output_hat_func,
            task_names,
            attention_module,
        )
