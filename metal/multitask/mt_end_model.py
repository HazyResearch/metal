import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model import EndModel
from metal.end_model.em_defaults import em_default_config
from metal.end_model.identity_module import IdentityModule
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.multitask import MTClassifier
from metal.multitask.mt_em_defaults import mt_em_default_config
from metal.multitask.task_graph import TaskGraph
from metal.utils import recursive_merge_dicts


class MTEndModel(MTClassifier, EndModel):
    """A multi-task discriminative model.

    Note that when looking up methods, MTEndModel will first search in
    MTClassifier, followed by EndModel.

    Args:
        layer_out_dims: a list of integers corresponding to the output sizes
            of the layers of your network. The first element is the
            dimensionality of the input layer, and all other elements dictate
            the sizes of middle layers. The number of middle layers will be
            inferred from this list. The output dimensions of the task heads
            will be inferred from the cardinalities pulled from K or the
            task_graph.
        input_modules: (nn.Module) a list of modules that converts the
            user-provided model inputs to torch.Tensors.
            Defaults to IdentityModule.
        middle_modules: (nn.Module) a list of modules to execute between the
            input_module and task head. Defaults to nn.Linear.
        head_module: (nn.Module) a module to execute right before the final
            softmax that outputs a prediction for the task.
        K: A t-length list of task cardinalities (overrided by task_graph
            if task_graph is not None)
        task_graph: TaskGraph: A TaskGraph which defines a feasible set of
            task label vectors; overrides K
    """

    def __init__(
        self,
        layer_out_dims,
        input_modules=None,
        middle_modules=None,
        head_modules=None,
        K=[],
        task_graph=None,
        **kwargs,
    ):

        kwargs["layer_out_dims"] = layer_out_dims
        # kwargs["input_modules"] = input_modules
        # kwargs["middle_modules"] = middle_modules
        # kwargs["head_modules"] = head_modules

        config = recursive_merge_dicts(
            em_default_config, mt_em_default_config, misses="insert"
        )
        config = recursive_merge_dicts(config, kwargs)
        MTClassifier.__init__(self, K, config)

        if task_graph is None:
            if len(K) == 0:
                raise ValueError(
                    "You must supply either a list of "
                    "cardinalities (K) or a TaskGraph."
                )
            task_graph = TaskGraph(K)
        self.task_graph = task_graph
        self.K = self.task_graph.K  # Cardinalities by task
        self.t = self.task_graph.t  # Total number of tasks
        assert len(self.K) == self.t

        self._build(input_modules, middle_modules, head_modules)

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_modules, middle_modules, head_modules):
        """
        TBD
        """
        self.input_layer = self._build_input_layer(input_modules)
        self.middle_layers = self._build_middle_layers(middle_modules)
        self.heads = self._build_task_heads(head_modules)

        # Construct loss module
        reduction = self.config["train_config"]["loss_fn_reduction"]
        self.criteria = SoftCrossEntropyLoss(reduction=reduction)

    def _build_input_layer(self, input_modules):
        if input_modules is None:
            output_dim = self.config["layer_out_dims"][0]
            input_modules = IdentityModule()

        if isinstance(input_modules, list):
            input_layer = [
                self._make_layer(mod, "input", self.config["input_layer_config"])
                for mod in input_modules
            ]
        else:
            input_layer = self._make_layer(
                input_modules,
                "input",
                self.config["input_layer_config"],
                output_dim=output_dim,
            )

        return input_layer

    def _build_middle_layers(self, middle_modules):
        layer_out_dims = self.config["layer_out_dims"]
        num_mid_layers = len(layer_out_dims) - 1
        if num_mid_layers == 0:
            return None

        middle_layers = nn.ModuleList()
        for i in range(num_mid_layers):
            if middle_modules is None:
                module = nn.Linear(*layer_out_dims[i : i + 2])
                layer = self._make_layer(
                    module,
                    "middle",
                    self.config["middle_layer_config"],
                    output_dim=layer_out_dims[i + 1],
                )
            else:
                module = middle_modules[i]
                layer = self._make_layer(
                    module, "middle", self.config["middle_layer_config"]
                )
            middle_layers.add_module(f"layer{i+1}", layer)
        return middle_layers

    def _build_task_heads(self, head_modules):
        """Creates and attaches task_heads to the appropriate network layers"""
        # Make task head layer assignments
        num_layers = len(self.config["layer_out_dims"])
        task_head_layers = self._set_task_head_layers(num_layers)

        # task_head_layers stores the layer whose output is input to task head t
        # task_map stores the task heads that appear at each layer
        self.task_map = defaultdict(list)
        for t, l in enumerate(task_head_layers):
            self.task_map[l].append(t)

        if any(l == 0 for l in task_head_layers) and head_modules is None:
            raise Exception(
                "If any task head is being attached to layer 0 "
                "(the input modules), then you must provide a t-length list of "
                "head_modules, since the output dimension of each input_module "
                "cannot be inferred."
            )

        # Construct heads
        head_dims = [self.K[t] for t in range(self.t)]

        heads = nn.ModuleList()
        for t in range(self.t):
            input_dim = self.config["layer_out_dims"][task_head_layers[t]]
            if self.config["pass_predictions"]:
                for p in self.task_graph.parents[t]:
                    input_dim += head_dims[p]
            output_dim = head_dims[t]

            if head_modules is None:
                head = nn.Linear(input_dim, output_dim)
            elif isinstance(head_modules, list):
                head = head_modules[t]
            else:
                head = copy.deepcopy(head_modules)
            heads.append(head)
        return heads

    def _set_task_head_layers(self, num_layers):
        head_layers = self.config["task_head_layers"]
        if isinstance(head_layers, list):
            task_head_layers = head_layers
        elif head_layers == "top":
            task_head_layers = [num_layers - 1] * self.t
        else:
            msg = f"Invalid option to 'head_layers' parameter: '{head_layers}'"
            raise ValueError(msg)

        # Confirm that the network does not extend beyond the latest task head
        if max(task_head_layers) < num_layers - 1:
            unused = num_layers - 1 - max(task_head_layers)
            msg = (
                f"The last {unused} layer(s) of your network have no task "
                "heads attached to them"
            )
            raise ValueError(msg)

        # Confirm that parents come b/f children if predictions are passed
        # between tasks
        if self.config["pass_predictions"]:
            for t, l in enumerate(task_head_layers):
                for p in self.task_graph.parents[t]:
                    if task_head_layers[p] >= l:
                        p_layer = task_head_layers[p]
                        msg = (
                            f"Task {t}'s layer ({l}) must be larger than its "
                            f"parent task {p}'s layer ({p_layer})"
                        )
                        raise ValueError(msg)

        return task_head_layers

    def _print(self):
        print("\n--Input Layer--")
        if isinstance(self.input_layer, list):
            for mod in self.input_layer:
                print(mod)
        else:
            print(self.input_layer)

        for t in self.task_map[0]:
            print(f"(head{t})")
            print(self.heads[t])

        print("\n--Middle Layers--")
        for i, layer in enumerate(self.middle_layers, start=1):
            print(f"(layer{i}):")
            print(layer)
            for t in self.task_map[i]:
                print(f"(head{t})")
                print(self.heads[t])
            print()

    def forward(self, x):
        """Returns a list of outputs for tasks 0,...t-1

        Args:
            x: a [batch_size, ...] batch from X
        """
        head_outputs = [None] * self.t

        # Execute input layer
        if isinstance(self.input_layer, list):  # One input_module per task
            input_outputs = [mod(x) for mod, x in zip(self.input_layer, x)]
            x = torch.stack(input_outputs, dim=1)

            # Execute level-0 task heads from their respective input modules
            for t in self.task_map[0]:
                head = self.heads[t]
                head_outputs[t] = head(input_outputs[t])
        else:  # One input_module for all tasks
            x = self.input_layer(x)

            # Execute level-0 task heads from the single input module
            for t in self.task_map[0]:
                head = self.heads[t]
                head_outputs[t] = head(x)

        # Execute middle layers
        for i, layer in enumerate(self.middle_layers, start=1):
            x = layer(x)

            # Attach level-i task heads from the ith middle module
            for t in self.task_map[i]:
                head = self.heads[t]
                # Optionally include as input the predictions of parent tasks
                if self.config["pass_predictions"] and bool(self.task_graph.parents[t]):
                    task_input = [x]
                    for p in self.task_graph.parents[t]:
                        task_input.append(head_outputs[p])
                    task_input = torch.stack(task_input, dim=1)
                else:
                    task_input = x
                head_outputs[t] = head(task_input)
        return head_outputs

    def _preprocess_Y(self, Y, k=None):
        """Convert Y to t-length list of probabilistic labels if necessary"""
        # If not a list, convert to a singleton list
        if not isinstance(Y, list):
            if self.t != 1:
                msg = "For t > 1, Y must be a list of n-dim or [n, K_t] tensors"
                raise ValueError(msg)
            Y = [Y]

        if not len(Y) == self.t:
            msg = f"Expected Y to be a t-length list (t={self.t}), not {len(Y)}"
            raise ValueError(msg)

        return [EndModel._preprocess_Y(self, Y_t, self.K[t]) for t, Y_t in enumerate(Y)]

    def _get_loss_fn(self):
        """Returns the loss function to use in the train_model routine"""
        criteria = self.criteria.to(self.config["device"])
        loss_fn = lambda X, Y: sum(
            criteria(Y_tp, Y_t) for Y_tp, Y_t in zip(self.forward(X), Y)
        )
        return loss_fn

    def predict_proba(self, X):
        """Returns a list of t [n, K_t] tensors of probabilistic (float) predictions."""
        return [
            F.softmax(output, dim=1).data.cpu().numpy() for output in self.forward(X)
        ]

    def predict_task_proba(self, X, t):
        """Returns an n x k matrix of probabilities for each label of task t"""
        return self.predict_proba(X)[t]
