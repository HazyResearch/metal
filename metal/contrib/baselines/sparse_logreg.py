from metal.contrib.modules.sparse_linear_module import SparseLinearModule
from metal.end_model import EndModel
from metal.utils import recursive_merge_dicts


class SparseLogisticRegression(EndModel):
    """A _sparse_ logistic regression classifier for a single-task problem

    Args:
        input_dim: The maximum length of each input (a tensor of integer
            indices corresponding to one-hot features)
        output_dim: The cardinality of the classifier
        padding_idx: If not None, the embedding initialized to 0 so no gradient
            will pass through it.
    """

    def __init__(self, input_dim, output_dim=2, padding_idx=0, **kwargs):
        layer_out_dims = [input_dim, output_dim]
        sparse_linear = SparseLinearModule(
            vocab_size=input_dim, embed_size=output_dim, padding_idx=padding_idx
        )
        overrides = {"input_batchnorm": False, "input_dropout": 0.0}
        kwargs = recursive_merge_dicts(
            kwargs, overrides, misses="insert", verbose=False
        )
        super().__init__(layer_out_dims, head_module=sparse_linear, **kwargs)
