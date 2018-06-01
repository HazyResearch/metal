import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from metal.classifier import Classifier
from metal.end_model.em_defaults import em_model_defaults, em_train_defaults
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.modules import IdentityModule
from metal.utils import MultilabelDataset, hard_to_soft, multitask_decorator

class EndModel(Classifier):
    def __init__(self, input_module=IdentityModule, **kwargs):
        # TODO: merge kwargs into model_params
        multitask = False # TODO: actually calculate multitask based on inputs
        super().__init__(multitask)
        self.mp = em_model_defaults # TODO: merge kwargs with defaults here

        # Build the network
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 2),
        )
        self.criteria = SoftCrossEntropyLoss()

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
        if module_name in ['EndModel', 'Sequential', 'SoftCrossEntropyLoss']:
            pass
        elif callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()
        else:
            raise Exception(f"Module {module_name} was not initialized.")

    def _preprocess_Y(self, Y):
        """Convert Y to T-dim lists of soft labels if necessary"""
        # If not a list, convert to a singleton list
        if not isinstance(Y, list):
            Y = [Y]

        # If hard labels, convert to soft labels
        for t, Y_t in enumerate(Y):
            if Y_t.dim() == 1 or Y_t.shape[1] == 1:
                # FIXME: This could fail if last class was never predicted
                Y[t] = hard_to_soft(Y_t, k=Y_t.max())
        return Y

    def get_loss(self, outputs, Y):
        """Return the loss of Y and the output(s) of the net forward pass.
        
        The returned loss is averaged over items (by the loss function) but
        summed over tasks.
        """
        loss = torch.tensor(0.0)
        for t, Y_tp in enumerate(outputs):
            loss += self.criteria(Y_tp, Y[t])
        return loss

    def forward(self, x):
        """Returns a list of outputs for tasks t=0,...T-1
        
        Args:
            x: a [batch_size, ...] batch from X
        """
        return [self.net(x)]
    
    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, 
        train_params=em_train_defaults, **kwargs):
        # TODO: merge kwargs into model_params
        self.tp = train_params

        # Y_train = self._preprocess_Y(Y_train)
        # Y_dev = self._preprocess_Y(Y_dev)

        if self.tp['use_cuda']:
            raise NotImplementedError
            # TODO: fix this
            # X = X.cuda(self.gpu_id)
            # Y = Y.cuda(self.gpu_id)

        # Make data loaders
        dataset = MultilabelDataset(X_train, self._preprocess_Y(Y_train))
        train_loader = DataLoader(dataset, shuffle=True, 
            **self.tp['data_loader_params'])

        if X_dev is not None and Y_dev is not None:
            dataset = MultilabelDataset(X_dev, self._preprocess_Y(Y_dev))
            dev_loader = DataLoader(dataset, shuffle=True, 
                **self.tp['data_loader_params'])
        else:
            dev_loader = None

        # Show network
        if self.tp['verbose']:
            print(self)

        # Put model on GPU 
        # if self.config['use_cuda']:
        #     raise NotImplementedError

        # Set the optimizer
        optimizer = optim.SGD(
            self.parameters(), 
            **self.tp['optimizer_params'],
            **self.tp['sgd_params']
        )

        # Initialize the model
        self.reset()

        # Train the model
        for epoch in range(self.tp['n_epochs']):
            epoch_loss = 0.0
            for i, data in enumerate(train_loader):
                X, Y = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass -> get loss -> backward -> take optimizer step
                outputs = self.forward(X)
                loss = self.get_loss(outputs, Y)
                loss.backward()

                # Gradient clipping
                # if grad_clip:
                #     torch.nn.utils.clip_grad_norm(self.net.parameters(), grad_clip)

                optimizer.step()

                # Print statistics
                epoch_loss += loss.data

            train_loss = epoch_loss / len(train_loader.dataset)
            if dev_loader:
                dev_score = self.score(X_dev, Y_dev, verbose=False)
            
            # Report progress
            if epoch % self.tp['print_every'] == 0:
                msg = f'[E:{epoch+1}]\tTrain Loss: {train_loss:.3f}'
                if dev_loader:
                    msg += f'\tDev score: {dev_score:.3f}'
                print(msg)

    @multitask_decorator
    def predict_proba(self, X):
        """Returns a list of T [N, K_t] tensors of soft (float) predictions."""
        return [F.softmax(Y_tp, dim=1).data.cpu() for Y_tp in self.forward(X)]

    def predict_task_proba(self, X, t):
        """Returns an N x k matrix of probabilities for each label of task t"""
        return self.predict_tasks_proba(X)[t]