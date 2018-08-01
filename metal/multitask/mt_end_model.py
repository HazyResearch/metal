from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metal.end_model import EndModel
from metal.end_model.em_defaults import em_default_config
from metal.multitask import MTClassifier, TaskGraph, MTMetalDataset
from metal.multitask.mt_em_defaults import mt_em_default_config
from metal.input_modules import IdentityModule
from metal.utils import recursive_merge_dicts

class MTEndModel(MTClassifier, EndModel):
    """A multi-task discriminative model.

    Note that when looking up methods, MTEndModel will first search in 
    MTClassifier, followed by EndModel.
    """
    def __init__(self, task_graph=None, input_module=None, seed=None, **kwargs):
        defaults = recursive_merge_dicts(
            em_default_config, mt_em_default_config, misses='insert')
        self.config = recursive_merge_dicts(defaults, kwargs)

        # If no task_graph is specified, default to a single binary task
        if task_graph is None:
            task_graph = TaskGraph(edges=[], cardinalities=[2])
        self.task_graph = task_graph
        self.K_t = self.task_graph.K_t  # Cardinalities by task
        self.T = self.task_graph.T      # Total number of tasks

        MTClassifier.__init__(self, cardinalities=self.K_t, seed=seed)

        if input_module is None:
            input_module = IdentityModule(self.config['layer_output_dims'][0])

        self._build(input_module)

        # Show network
        if self.config['verbose']:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _set_task_head_layers(self, num_layers):
        head_layers = self.config['task_head_layers']
        if isinstance(head_layers, list):
            task_head_layers = head_layers
        elif head_layers == 'top':
            task_head_layers = [num_layers - 1] * self.T
        elif head_layers == 'auto':
            raise NotImplementedError
        else:
            msg = (f"Invalid option to 'head_layers' parameter: "
                f"{head_layers}")
            raise ValueError(msg)

        # Confirm that the network does not extend beyond the latest task head
        if max(task_head_layers) < num_layers - 1:
            unused = num_layers - 1 - max(task_head_layers)
            msg = (f"The last {unused} layer(s) of your network have no task "
                "heads attached to them")
            raise ValueError(msg)

        # Confirm that parents come b/f children if predictions are passed 
        # between tasks
        if self.config['pass_predictions']:
            for t, l in enumerate(task_head_layers):
                for p in self.task_graph.parents[t]:
                    if task_head_layers[p] >= l:
                        p_layer = task_head_layers[p]
                        msg = (f"Task {t}'s layer ({l}) must be larger than its "
                            f"parent task {p}'s layer ({p_layer})")
                        raise ValueError(msg)

        return task_head_layers

    def _attach_task_heads(self, num_layers):
        """Creates and attaches task_heads to the appropriate network layers"""
        # Make task head layer assignments
        task_head_layers = self._set_task_head_layers(num_layers)

        # task_head_layers stores the layer whose output task head t takes as input
        # task_map stores the task heads that appear at each layer
        self.task_map = defaultdict(list)
        for t, l in enumerate(task_head_layers):
            self.task_map[l].append(t)

        # Construct heads
        # TODO: Try to get heads to show up at proper depth when printing net
        head_dims = self.config['task_head_output_dims']
        if head_dims is None:
            head_dims = [self.K_t[t] for t in range(self.T)]

        self.heads = nn.ModuleList()
        for t in range(self.T):
            input_dim = self.config['layer_output_dims'][task_head_layers[t]]
            if self.config['pass_predictions']:
                for p in self.task_graph.parents[t]:
                    input_dim += head_dims[p]
            output_dim = head_dims[t]
            self.heads.append(nn.Linear(input_dim, output_dim))

    def _print(self):
        print("\n--Trunk--")
        print(self.layers)
        print("\n--Heads--")
        for layer, tasks in self.task_map.items():
            print(f"(layer{layer})+:")
            for t in tasks:
                print(self.heads[t])

    def _preprocess_Y(self, Y):
        """Convert Y to T-dim lists of soft labels if necessary"""
        # If not a list, convert to a singleton list
        if not isinstance(Y, list):
            if self.T != 1:
                msg = "For T > 1, Y must be a list of N-dim or [N, K_t] tensors"
                raise ValueError(msg)
            Y = [Y]
        
        if not len(Y) == self.T:
            msg = "Expected Y to be a list of length T ({self.T}), not {len(Y)}"
            raise ValueError(msg)

        Y = [Y_t.clone() for Y_t in Y]

        return [EndModel._preprocess_Y(self, Y_t) for Y_t in Y]

    def _make_data_loader(self, X, Y, data_loader_config):
        dataset = MTMetalDataset(X, self._preprocess_Y(Y))
        data_loader = DataLoader(dataset, shuffle=True, **data_loader_config)
        return data_loader

    def _get_loss(self, output, Y):
        """Return the loss of Y and the output of the net forward pass.
        
        The returned loss is averaged over items (by the loss function) but
        summed over tasks.
        """
        loss = torch.tensor(0.0)
        for t, Y_tp in enumerate(output):
            loss += self.criteria(Y_tp, Y[t])
        return loss

    def forward(self, x):
        """Returns a list of outputs for tasks t=0,...T-1
        
        Args:
            x: a [batch_size, ...] batch from X
        """
        task_outputs = [None] * self.T
        for i, layer in enumerate(self.layers):
            x = layer(x)
            for t in self.task_map[i]:
                head = self.heads[t]
                if (self.config['pass_predictions'] and 
                    bool(self.task_graph.parents[t])):
                    task_input = [x]
                    for p in self.task_graph.parents[t]:
                        task_input.append(task_outputs[p])
                    task_input = torch.cat(task_input, dim=1)
                else:
                    task_input = x
                task_outputs[t] = head(task_input)
        return task_outputs

    def predict_proba(self, X):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [F.softmax(output, dim=1).data.cpu().numpy() for output in 
            self.forward(X)]

    def predict_task_proba(self, X, t):
        """Returns an N x k matrix of probabilities for each label of task t"""
        return self.predict_tasks_proba(X)[t]