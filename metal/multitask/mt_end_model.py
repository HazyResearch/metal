from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metal.end_model import EndModel
from metal.end_model.em_defaults import em_default_config
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.multitask import TaskHierarchy, MTClassifier
from metal.multitask import MultiYDataset, MultiXYDataset
from metal.multitask.mt_em_defaults import mt_em_default_config
from metal.input_modules import IdentityModule
from metal.utils import recursive_merge_dicts

class MTEndModel(MTClassifier, EndModel):
    """A multi-task discriminative model.

    Note that when looking up methods, MTEndModel will first search in 
    MTClassifier, followed by EndModel.
    """
    def __init__(self, task_graph=None, input_modules=None, head_modules=None,
        seed=None, **kwargs):
        defaults = recursive_merge_dicts(
            em_default_config, mt_em_default_config, misses='insert')
        self.config = recursive_merge_dicts(defaults, kwargs)

        # If no task_graph is specified, default to a single binary task
        if task_graph is None:
            task_graph = TaskHierarchy(edges=[], cardinalities=[2])
        self.task_graph = task_graph
        self.K_t = self.task_graph.K_t  # Cardinalities by task
        self.T = self.task_graph.T      # Total number of tasks

        MTClassifier.__init__(self, cardinalities=self.K_t, seed=seed)

        self._build(input_modules, head_modules)

        # Show network
        if self.config['verbose']:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_modules, head_modules):
        """
        TBD
        """
        self.input_layer = self._build_input_layer(input_modules)
        self.middle_layers = self._build_middle_layers()
        self.heads = self._build_task_heads(head_modules)  

        # Construct loss module
        self.criteria = SoftCrossEntropyLoss()

    def _build_input_layer(self, input_modules):
        if input_modules is None:
            output_dim = self.config['layer_out_dims'][0]
            input_modules = IdentityModule()

        if isinstance(input_modules, list):
            input_layer = [self._make_layer(mod) for mod in input_modules]
        else:
            input_layer = self._make_layer(input_modules, output_dim)

        return input_layer

    def _build_middle_layers(self):
        middle_layers = nn.ModuleList()
        layer_out_dims = self.config['layer_out_dims']
        num_layers = len(layer_out_dims)
        for i in range(1, num_layers):
            module = nn.Linear(*layer_out_dims[i-1:i+1])
            layer = self._make_layer(module, output_dim=layer_out_dims[i])
            middle_layers.add_module(f'layer{i}', layer)
        return middle_layers

    def _build_task_heads(self, head_modules):
        """Creates and attaches task_heads to the appropriate network layers"""
        # Make task head layer assignments
        num_layers = len(self.config['layer_out_dims'])
        task_head_layers = self._set_task_head_layers(num_layers)

        # task_head_layers stores the layer whose output task head t takes as input
        # task_map stores the task heads that appear at each layer
        self.task_map = defaultdict(list)
        for t, l in enumerate(task_head_layers):
            self.task_map[l].append(t)

        # Construct heads
        head_dims = [self.K_t[t] for t in range(self.T)]

        heads = nn.ModuleList()
        for t in range(self.T):
            input_dim = self.config['layer_out_dims'][task_head_layers[t]]
            if self.config['pass_predictions']:
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
        """Returns a list of outputs for tasks t=0,...T-1
        
        Args:
            x: a [batch_size, ...] batch from X
        """
        head_outputs = [None] * self.T
        
        # Execute input layer
        if isinstance(self.input_layer, list): # One input_module per task
            input_outputs = [mod(x) for mod, x in zip(self.input_layer, x)]
            x = torch.stack(input_outputs, dim=1)

            # Execute level-0 task heads from their respective input modules
            for t in self.task_map[0]:
                head = self.heads[t]
                head_outputs[t] = head(input_outputs[t])
        else: # One input_module for all tasks
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
                if (self.config['pass_predictions'] and 
                    bool(self.task_graph.parents[t])):
                    task_input = [x]
                    for p in self.task_graph.parents[t]:
                        task_input.append(head_outputs[p])
                    task_input = torch.stack(task_input, dim=1)
                else:
                    task_input = x
                head_outputs[t] = head(task_input)
        return head_outputs

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
        if isinstance(X, list):
            dataset = MultiXYDataset(X, self._preprocess_Y(Y))
        else:   
            dataset = MultiYDataset(X, self._preprocess_Y(Y))
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

    def predict_proba(self, X):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [F.softmax(output, dim=1).data.cpu().numpy() for output in 
            self.forward(X)]

    def predict_task_proba(self, X, t):
        """Returns an N x k matrix of probabilities for each label of task t"""
        return self.predict_tasks_proba(X)[t]