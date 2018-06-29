from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from metal.classifier import Classifier, multitask
from metal.end_model.em_defaults import em_model_defaults
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.input_modules import IdentityModule
from metal.structs import TaskGraph
from metal.utils import (
    MultilabelDataset, 
    hard_to_soft, 
    recursive_merge_dicts,
)

class EndModel(Classifier):
    def __init__(self, label_map=None, task_graph=None, input_module=None, 
        **kwargs):
        self.config = recursive_merge_dicts(em_model_defaults, kwargs)

        multitask = isinstance(label_map, list) and len(label_map) > 1
        super().__init__(multitask, seed=self.config['seed'])

        if label_map is None:
            label_map = [[1,2]]  # Default to a single binary task
        self.label_map = label_map

        self.K_t = [len(x) for x in label_map]
        self.T = len(label_map)

        if task_graph is None:
            task_graph = TaskGraph(edges=[], cardinalities=self.K_t)
        self.task_graph = task_graph

        if input_module is None:
            input_module = IdentityModule(self.config['layer_output_dims'][0])

        self._build(input_module)

    def _build(self, input_module):
        # Notes:
        # Build in order so that display is correct
        # Build Sequential with OrderedDict so modules have names
        # Build in layers, execute layers 1 at a time
        # We get integer mapping of task heads to layers
        # Layer 0 is the input module

        layer_dims = self.config['layer_output_dims']
        # The number of layers is inferred from the specific layer_output_dims
        num_layers = len(layer_dims)

        # Make task head layer assignments
        if isinstance(self.config['head_layers'], list):
            task_layers = self.config['head_layers']
        elif self.config['head_layers'] == 'top':
            task_layers = [num_layers - 1] * self.T
        elif self.config['head_layers'] == 'auto':
            raise NotImplementedError
        else:
            msg = (f"Invalid option to 'head_layers' parameter: "
                f"{self.config['head_layers']}")
            raise ValueError(msg)

        if max(task_layers) < num_layers - 1:
            unused = num_layers - 1 - max(task_layers)
            msg = (f"The last {unused} layer(s) of your network have no task "
                "heads attached to them")
            raise ValueError(msg)

        # If we're passing predictions, confirm that parents come b/f children
        if self.config['pass_predictions']:
            for t, l in enumerate(task_layers):
                for p in self.task_graph.parents[t]:
                    if task_layers[p] >= l:
                        p_layer = task_layers[p]
                        msg = (f"Task {t}'s layer ({l}) must be larger than its "
                            f"parent task {p}'s layer ({p_layer})")
                        raise ValueError(msg)

        # task_layers stores the layer whose output task head t takes as input
        # task_map stores the task heads that appear at each layer
        self.task_map = defaultdict(list)
        for t, l in enumerate(task_layers):
            self.task_map[l].append(t)

        # Set dropout probabilities for all layers
        dropout = self.config['dropout']
        if isinstance(dropout, float):
            dropouts = [dropout] * num_layers
        elif isinstance(dropout, list):
            dropouts = dropout
        
        # Construct layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = []
            # Input module or Linear
            if i == 0:
                layer.append(input_module)
            else:
                layer.append(nn.Linear(*layer_dims[i-1:i+1]))
            if not isinstance(input_module, IdentityModule):
                layer.append(nn.ReLU())
            if self.config['batchnorm']:
                layer.append(nn.BatchNorm1d(layer_dims[i]))
            if self.config['dropout']:
                layer.append(nn.Dropout(dropouts[i]))
            self.layers.add_module(f'layer{i}', nn.Sequential(*layer))

        # Construct heads
        # TODO: Try to get heads to show up at proper depth when printing net
        # TODO: Allow user to specify output dimensions of task heads
        head_dims = self.config['head_output_dims']
        if head_dims is None:
            head_dims = [self.K_t[t] for t in range(self.T)]

        self.heads = nn.ModuleList()
        for t in range(self.T):
            input_dim = layer_dims[task_layers[t]]
            if self.config['pass_predictions']:
                for p in self.task_graph.parents[t]:
                    input_dim += head_dims[p]
            output_dim = head_dims[t]
            self.heads.append(nn.Linear(input_dim, output_dim))

        # Construct loss module
        self.criteria = SoftCrossEntropyLoss()

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

    @staticmethod
    def _reset_module(m):
        """A method for resetting the parameters of any module in the network

        First, handle special cases (unique initialization or none required)
        Next, use built in method if available
        Last, report that no initialization occured to avoid silent failure.

        This will be called on all children of m as well, so do not recurse
        manually.
        """
        module_name = m.__class__.__name__
        if module_name in ['EndModel', 'Sequential', 'ModuleList', 'ReLU', 
            'Dropout', 'LogisticRegression', 'SoftmaxRegression', 
            'SoftCrossEntropyLoss']:
            pass
        elif callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()
        else:
            # TODO: Once the core library is in place and tested, remove this
            # exception so it doesn't complain on user-provided input modules.
            # Until then though, keep it in place so we notice when a module
            # is not being initialized.
            raise Exception(f"Module {module_name} was not initialized.")

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

        Y = [Yt.clone() for Yt in Y]

        # If hard labels, convert to soft labels
        for t, Y_t in enumerate(Y):
            if Y_t.dim() == 1 or Y_t.shape[1] == 1:
                if not isinstance(Y_t, torch.LongTensor):
                    self._check(Y_t, typ=torch.LongTensor)
                # FIXME: This could fail if last class was never predicted
                Y[t] = hard_to_soft(Y_t, k=Y_t.max().long())
        return Y

    def _set_optimizer(self, optimizer_config):
        opt = optimizer_config['optimizer']
        if opt == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), 
                **optimizer_config['optimizer_common'],
                **optimizer_config['sgd_config']
            )
        else:
            raise ValueError(f"Did not recognize optimizer option '{opt}''") 
        return optimizer

    def _set_scheduler(self, scheduler_config, optimizer):
        scheduler = scheduler_config['scheduler']
        if scheduler is None:
            pass
        elif scheduler == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **scheduler_config['exponential_config'])
        elif scheduler == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_config['plateau_config'])
        else:
            raise ValueError(f"Did not recognize scheduler option '{scheduler}''")
        return lr_scheduler

    def get_loss(self, outputs, Y):
        """Return the loss of Y and the output(s) of the net forward pass.
        
        The returned loss is averaged over items (by the loss function) but
        summed over tasks.
        """
        loss = torch.tensor(0.0)
        for t, Y_tp in enumerate(outputs):
            loss += self.criteria(Y_tp, Y[t])
        return loss

    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)
        train_config = self.config['train_config']

        if train_config['use_cuda']:
            raise NotImplementedError
            # TODO: fix this
            # X = X.cuda(self.gpu_id)
            # Y = Y.cuda(self.gpu_id)
            # TODO: put model on gpu

        # Make data loaders
        dataset = MultilabelDataset(X_train, self._preprocess_Y(Y_train))
        train_loader = DataLoader(dataset, shuffle=True, 
            **train_config['data_loader_config'])

        if X_dev is not None and Y_dev is not None:
            dataset = MultilabelDataset(X_dev, self._preprocess_Y(Y_dev))
            dev_loader = DataLoader(dataset, shuffle=True, 
                **train_config['data_loader_config'])
        else:
            dev_loader = None

        # Show network
        if self.config['verbose']:
            print(self)

        # Set the optimizer
        optimizer_config = train_config['optimizer_config']
        optimizer = self._set_optimizer(optimizer_config)

        # Set the lr scheduler
        scheduler_config = train_config['scheduler_config']
        lr_scheduler = self._set_scheduler(scheduler_config, optimizer)

        # Initialize the model
        self.reset()

        # Train the model
        for epoch in range(train_config['n_epochs']):
            epoch_loss = 0.0
            for i, data in enumerate(train_loader):
                X, Y = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass to calculate outputs
                outputs = self.forward(X)
                loss = self.get_loss(outputs, Y)

                # Backward pass to calculate gradients
                loss.backward()

                # Clip gradients
                # if grad_clip:
                #     torch.nn.utils.clip_grad_norm(
                #        self.net.parameters(), grad_clip)

                # Perform optimizer step
                optimizer.step()

                # Keep running sum of losses
                epoch_loss += loss.detach() * X.shape[0]

            # Calculate average loss per training example
            # Saving division until this stage protects against the potential
            # mistake of averaging batch losses when the last batch is an orphan
            train_loss = epoch_loss / len(train_loader.dataset)

            if dev_loader:
                dev_score = self.score(X_dev, Y_dev, verbose=False)
            
            # Apply learning rate scheduler
            if (lr_scheduler is not None 
                and epoch + 1 >= scheduler_config['lr_freeze']):
                if scheduler_config['scheduler'] == 'reduce_on_plateau':
                    if dev_loader:
                        lr_scheduler.step(dev_score)
                else:
                    lr_scheduler.step()

            # Report progress
            if (self.config['verbose'] and 
                (epoch % train_config['print_at'] == 0 
                or epoch == train_config['n_epochs'] - 1)):
                msg = f'[E:{epoch+1}]\tTrain Loss: {train_loss:.3f}'
                if dev_loader:
                    msg += f'\tDev score: {dev_score:.3f}'
                print(msg)

    @multitask
    def predict_proba(self, X):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [F.softmax(Y_tp, dim=1).data.cpu() for Y_tp in self.forward(X)]

    def predict_task_proba(self, X, t):
        """Returns an N x k matrix of probabilities for each label of task t"""
        return self.predict_tasks_proba(X)[t]