from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from metal.analysis import plot_probabilities_histogram, confusion_matrix
from metal.classifier import Classifier
from metal.end_model.em_defaults import  em_default_config
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.input_modules import IdentityModule
from metal.utils import (
    MetalDataset,
    hard_to_soft, 
    recursive_merge_dicts,
)

class EndModel(Classifier):
    def __init__(self, cardinality=2, input_module=None, **kwargs):
        self.config = recursive_merge_dicts( em_default_config, kwargs)
        super().__init__(cardinality, seed=self.config['seed'])

        if input_module is None:
            input_module = IdentityModule(self.config['layer_output_dims'][0])

        self._build(input_module)

       # Show network
        if self.config['verbose']:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_module):
        """
        TBD
        """
        # The number of layers is inferred from the specified layer_output_dims
        layer_dims = self.config['layer_output_dims']
        num_layers = len(layer_dims)

        if not input_module.get_output_dim() == layer_dims[0]:
            msg = (f"Input module output size != the first layer output size: "
                f"({input_module.get_output_dim()} != {layer_dims[0]})")
            raise ValueError(msg)

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

        self._attach_task_heads(num_layers)

        # Construct loss module
        self.criteria = SoftCrossEntropyLoss()

    def _attach_task_heads(self, num_layers):
        """Create and attach a task head to the end of the network trunk"""
        input_dim = self.config['layer_output_dims'][-1]
        output_dim = self.k
        head = nn.Linear(input_dim, output_dim)
        self.network = nn.Sequential(*(self.layers), head)

    def _print(self):
        print(self.network)

    def forward(self, x):
        """Returns a list of outputs for tasks t=0,...T-1
        
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
        if callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()

    def config_set(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def _preprocess_Y(self, Y):
        """Convert Y to soft labels if necessary"""

        # If hard labels, convert to soft labels
        if Y.dim() == 1 or Y.shape[1] == 1:
            if not isinstance(Y, torch.LongTensor):
                self._check(Y, typ=torch.LongTensor)
            # FIXME: This could fail if last class was never predicted
            Y = hard_to_soft(Y, k=Y.max().long())
            # FIXME: This currently assumes that no model can output a 
            # prediction of 0 (i.e., if cardinality=5, Y[0] corresponds to
            # class 1 instead of 0, since the latter would give the model
            # a 5-dim output space but a 6-dim label space)
            Y = Y[:,1:]
        return Y

    def _make_data_loader(self, X, Y, data_loader_config):
        dataset = MetalDataset(X, self._preprocess_Y(Y))
        data_loader = DataLoader(dataset, shuffle=True, **data_loader_config)
        return data_loader

    def _get_loss(self, output, Y):
        """Return the loss of Y and the output of the net forward pass.
        
        The returned loss is averaged over items (by the loss function) but
        summed over tasks.
        """
        loss = torch.tensor(self.criteria(output, Y), dtype=torch.float)
        return loss
    
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


    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)
        train_config = self.config['train_config']

        Y_train = self._to_torch(Y_train)
        Y_dev = self._to_torch(Y_dev)

        if train_config['use_cuda']:
            raise NotImplementedError
            # TODO: fix this
            # X = X.cuda(self.gpu_id)
            # Y = Y.cuda(self.gpu_id)
            # TODO: put model on gpu

        # Make data loaders
        loader_config = train_config['data_loader_config']
        train_loader = self._make_data_loader(X_train, Y_train, loader_config)
        if X_dev is not None and Y_dev is not None:
            dev_loader = self._make_data_loader(X_dev, Y_dev, loader_config)
        else:
            dev_loader = None

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
                output = self.forward(X)
                loss = self._get_loss(output, Y)

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
                val_metric = train_config['validation_metric']
                dev_score = self.score(X_dev, Y_dev, metric=val_metric, 
                    verbose=False)
            
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
                (epoch % train_config['print_every'] == 0 
                or epoch == train_config['n_epochs'] - 1)):
                msg = f'[E:{epoch+1}]\tTrain Loss: {train_loss:.3f}'
                if dev_loader:
                    msg += f'\tDev score: {dev_score:.3f}'
                print(msg)

        if self.config['verbose']:
            print('Finished Training')
            
            if self.config['show_plots']:
                if self.k == 2:
                    Y_p_train = self.predict_proba(X_train)
                    plot_probabilities_histogram(Y_p_train[:, 0], 
                        title="Training Set Predictions")

            if X_dev is not None and Y_dev is not None:
                Y_ph_dev = self.predict(X_dev)

                print("Confusion Matrix (Dev)")
                mat = confusion_matrix(Y_ph_dev, Y_dev, pretty_print=True)                

    def predict_proba(self, X):
        """Returns a [N, K_t] tensor of soft (float) predictions."""
        return F.softmax(self.forward(X), dim=1).data.cpu().numpy()