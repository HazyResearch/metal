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
    Checkpointer,
    hard_to_soft, 
    recursive_merge_dicts,
)

class EndModel(Classifier):
    def __init__(self, cardinality=2, input_module=None, head_module=None, 
        **kwargs):
        self.config = recursive_merge_dicts( em_default_config, kwargs)
        super().__init__(cardinality, seed=self.config['seed'])

        self._build(input_module, head_module)

       # Show network
        if self.config['verbose']:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_module, head_module):
        """
        TBD
        """
        input_layer = self._build_input_layer(input_module)
        middle_layers = self._build_middle_layers()
        head = self._build_task_head(head_module)  
        self.network = nn.Sequential(input_layer, *middle_layers, head)

        # Construct loss module
        self.criteria = SoftCrossEntropyLoss()

    def _build_input_layer(self, input_module):
        if input_module is None:
            input_module = IdentityModule()
        output_dim = self.config['layer_out_dims'][0]
        input_layer = self._make_layer(input_module, output_dim=output_dim)
        return input_layer

    def _build_middle_layers(self):
        layers = nn.ModuleList()
        layer_out_dims = self.config['layer_out_dims']
        num_layers = len(layer_out_dims)
        for i in range(1, num_layers):
            module = nn.Linear(*layer_out_dims[i-1:i+1])
            layer = self._make_layer(module, output_dim=layer_out_dims[i])
            layers.add_module(f'layer{i}', layer)
        return layers

    def _build_task_head(self, head_module):
        if head_module is None:
            head = nn.Linear(self.config['layer_out_dims'][-1], self.k)
        else:
            # Note that if head module is provided, it must have input dim of
            # the last middle module and output dim of self.k, the cardinality
            head = head_module        
        return head

    def _make_layer(self, module, output_dim=None):
        layer = [module]
        if not isinstance(module, IdentityModule):
            layer.append(nn.ReLU())
            if self.config['batchnorm'] and output_dim:
                layer.append(nn.BatchNorm1d(output_dim))
            if self.config['dropout']:
                layer.append(nn.Dropout(self.config['dropout']))
        return nn.Sequential(*layer)

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

    def update_config(self, update_dict):
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
        elif opt == 'adam':
            optimizer = optim.Adam(
                self.parameters(), 
                **optimizer_config['optimizer_common'],
                **optimizer_config['adam_config']
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
        evaluate_dev = (X_dev is not None and Y_dev is not None)

        # Initialize the model
        self.reset()

        ### CUT HERE ###

        # Set the optimizer
        optimizer_config = train_config['optimizer_config']
        optimizer = self._set_optimizer(optimizer_config)

        # Set the lr scheduler
        scheduler_config = train_config['scheduler_config']
        lr_scheduler = self._set_scheduler(scheduler_config, optimizer)

        # Create the checkpointer if applicable
        if train_config['checkpoint']:
            checkpoint_config = train_config['checkpoint_config']
            model_class = type(self).__name__
            checkpointer = Checkpointer(model_class, **checkpoint_config, 
                verbose=self.config['verbose'])

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
                epoch_loss += loss.detach() * len(output)

            # Calculate average loss per training example
            # Saving division until this stage protects against the potential
            # mistake of averaging batch losses when the last batch is an orphan
            train_loss = epoch_loss / len(train_loader.dataset)

            if evaluate_dev:
                val_metric = train_config['validation_metric']
                dev_score = self.score(X_dev, Y_dev, metric=val_metric, 
                    verbose=False)
                if train_config['checkpoint']:
                    checkpointer.checkpoint(self, epoch, dev_score)
            
            # Apply learning rate scheduler
            if (lr_scheduler is not None 
                and epoch + 1 >= scheduler_config['lr_freeze']):
                if scheduler_config['scheduler'] == 'reduce_on_plateau':
                    if evaluate_dev:
                        lr_scheduler.step(dev_score)
                else:
                    lr_scheduler.step()

            # Report progress
            if (self.config['verbose'] and 
                (epoch % train_config['print_every'] == 0 
                or epoch == train_config['n_epochs'] - 1)):
                msg = f'[E:{epoch+1}]\tTrain Loss: {train_loss:.3f}'
                if evaluate_dev:
                    msg += f'\tDev score: {dev_score:.3f}'
                print(msg)

        # Restore best model if applicable
        if train_config['checkpoint']:
            checkpointer.restore(model=self)

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