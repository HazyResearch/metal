from functools import partial

import torch.nn.functional as F

from metal.mmtl.scorer import Scorer


class Task(object):
    """A task for use in an MMTL MetalModel

    Args:
        name: (str) The name of the task
            TODO: replace this with a more fully-featured path through the network
        input_module: (nn.Module) The input module
        head_module: (nn.Module) The task head module
        data: A dict of DataLoaders (instances and labels) to feed through the network
            with keys in ["train", "valid", "test"]
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
        scorer=Scorer(standard_metrics=["accuracy"]),
        loss_hat_func=(lambda X, Y: F.cross_entropy(X, Y - 1, reduction="mean")),
        output_hat_func=(partial(F.softmax, dim=1)),
    ) -> None:
        self.name = name
        self.data_loaders = data_loaders
        self.input_module = input_module
        self.head_module = head_module
        self.scorer = scorer
        self.loss_hat_func = loss_hat_func
        self.output_hat_func = output_hat_func
