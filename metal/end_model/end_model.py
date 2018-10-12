import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.classifier import Classifier
from metal.end_model.em_defaults import em_default_config
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.modules import IdentityModule
from metal.utils import MetalDataset, hard_to_soft, recursive_merge_dicts


class EndModel(Classifier):
    """A dynamically constructed discriminative classifier

        layer_out_dims: a list of integers corresponding to the output sizes
            of the layers of your network. The first element is the
            dimensionality of the input layer, the last element is the
            dimensionality of the head layer (equal to the cardinality of the
            task), and all other elements dictate the sizes of middle layers.
            The number of middle layers will be inferred from this list.
        input_module: (nn.Module) a module that converts the user-provided
            model inputs to torch.Tensors. Defaults to IdentityModule.
        middle_modules: (nn.Module) a list of modules to execute between the
            input_module and task head. Defaults to nn.Linear.
        head_module: (nn.Module) a module to execute right before the final
            softmax that outputs a prediction for the task.
    """

    def __init__(
        self,
        layer_out_dims,
        input_module=None,
        middle_modules=None,
        head_module=None,
        **kwargs,
    ):

        if len(layer_out_dims) < 2:
            raise ValueError(
                "Arg layer_out_dims must have at least two "
                "elements corresponding to the output dim of the input module "
                "and the cardinality of the task. If the input module is the "
                "IdentityModule, then the output dim of the input module will "
                "be equal to the dimensionality of your input data points"
            )

        # Add layer_out_dims to kwargs so it will be merged into the config dict
        kwargs["layer_out_dims"] = layer_out_dims
        config = recursive_merge_dicts(em_default_config, kwargs)
        super().__init__(k=layer_out_dims[-1], config=config)

        self._build(input_module, middle_modules, head_module)

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_module, middle_modules, head_module):
        """
        TBD
        """
        input_layer = self._build_input_layer(input_module)
        middle_layers = self._build_middle_layers(middle_modules)
        head = self._build_task_head(head_module)
        if middle_layers is None:
            self.network = nn.Sequential(input_layer, head)
        else:
            self.network = nn.Sequential(input_layer, *middle_layers, head)

        # Construct loss module
        self.criteria = SoftCrossEntropyLoss(reduction="sum")

    def _build_input_layer(self, input_module):
        if input_module is None:
            input_module = IdentityModule()
        output_dim = self.config["layer_out_dims"][0]
        input_layer = self._make_layer(input_module, output_dim=output_dim)
        return input_layer

    def _build_middle_layers(self, middle_modules):
        layer_out_dims = self.config["layer_out_dims"]
        num_mid_layers = len(layer_out_dims) - 2
        if num_mid_layers == 0:
            return None

        middle_layers = nn.ModuleList()
        for i in range(num_mid_layers):
            if middle_modules is None:
                module = nn.Linear(*layer_out_dims[i : i + 2])
                layer = self._make_layer(
                    module, output_dim=layer_out_dims[i + 1]
                )
            else:
                module = middle_modules[i]
                layer = self._make_layer(module)
            middle_layers.add_module(f"layer{i+1}", layer)
        return middle_layers

    def _build_task_head(self, head_module):
        if head_module is None:
            head = nn.Linear(self.config["layer_out_dims"][-2], self.k)
        else:
            # Note that if head module is provided, it must have input dim of
            # the last middle module and output dim of self.k, the cardinality
            head = head_module
        return head

    def _make_layer(self, module, output_dim=None):
        if isinstance(module, IdentityModule):
            return module
        layer = [module]
        layer.append(nn.ReLU())
        if self.config["batchnorm"] and output_dim:
            layer.append(nn.BatchNorm1d(output_dim))
        if self.config["dropout"]:
            layer.append(nn.Dropout(self.config["dropout"]))
        return nn.Sequential(*layer)

    def _print(self):
        print(self.network)

    def forward(self, x):
        """Returns a list of outputs for tasks 0,...t-1

        Args:
            x: a [batch_size, ...] batch from X
        """
        return self.network(x)

    @staticmethod
    def _reset_module(m):
        """A method for resetting the parameters of any module in the network

        First, handle special cases (unique initialization or none required)
        Next, use built in method if available
        Last, report that no initialization occured to avoid silent failure.

        This will be called on all children of m as well, so do not recurse
        manually.
        """
        if callable(getattr(m, "reset_parameters", None)):
            m.reset_parameters()

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def _preprocess_Y(self, Y, k):
        """Convert Y to soft labels if necessary"""
        Y = Y.clone()

        # If hard labels, convert to soft labels
        if Y.dim() == 1 or Y.shape[1] == 1:
            Y = hard_to_soft(Y.long(), k=k)
        return Y

    def _create_dataset(self, *data):
        return MetalDataset(*data)

    def _get_loss_fn(self):
        if self.config["use_cuda"]:
            criteria = self.criteria.cuda()
        else:
            criteria = self.criteria
        loss_fn = lambda X, Y: criteria(self.forward(X), Y)
        return loss_fn

    def train(self, train_data, dev_data=None, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)

        # If train_data is provided as a tuple (X, Y), we can make sure Y is in
        # the correct format
        # NOTE: Better handling for if train_data is Dataset or DataLoader...?
        if isinstance(train_data, (tuple, list)):
            X, Y = train_data
            Y = self._preprocess_Y(
                self._to_torch(Y, dtype=torch.FloatTensor), self.k
            )
            train_data = (X, Y)

        # Convert input data to data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)

        # Initialize the model
        self.reset()

        # Create loss function
        loss_fn = self._get_loss_fn()

        # Execute training procedure
        self._train(train_loader, loss_fn, dev_data=dev_data)

    def predict_proba(self, X):
        """Returns a [n, k] tensor of soft (float) predictions."""
        return F.softmax(self.forward(X), dim=1).data.cpu().numpy()
