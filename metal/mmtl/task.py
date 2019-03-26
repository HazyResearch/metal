from abc import ABC

import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.modules import MetalModule, MetalModuleWrapper
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
        self.input_module = self._wrap_module(input_module)
        self.middle_module = self._wrap_module(middle_module)
        self.attention_module = self._wrap_module(attention_module)
        self.head_module = self._wrap_module(head_module)
        self.output_hat_func = output_hat_func
        self.loss_hat_func = loss_hat_func
        self.loss_multiplier = loss_multiplier
        self.scorer = scorer

    @staticmethod
    def _wrap_module(module):
        if isinstance(module, MetalModule):
            return module
        else:
            return MetalModuleWrapper(module)

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
        output_hat_func=(lambda X: F.softmax(X["data"], dim=1)),
        loss_hat_func=(lambda X, Y: F.cross_entropy(X["data"], Y.view(-1) - 1)),
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
        output_hat_func=(lambda X: X["data"]),
        # Note: no sigmoid (target labels can be in any range)
        loss_hat_func=(lambda X, Y: F.mse_loss(X["data"].view(-1), Y.view(-1))),
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
