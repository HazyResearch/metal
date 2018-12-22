import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model.end_model import EndModel

class LinearModule(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.input_layer(x)


class MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dims=[], bias=False):
        super().__init__()

        # Create layers
        dims = [input_dim] + middle_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i + 1 < len(dims):
                layers.append(nn.Sigmoid())

        self.input_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.input_layer(x)

class SliceDPModel(EndModel):
    """
    Args:
        - input_module: (nn.Module) a module that converts the user-provided
            model inputs to torch.Tensors. Defaults to IdentityModule.
        - accs: The LF accuracies, computed offline
        - r: Intermediate representation dimension
        - rw: Whether to use reweighting of representation for Y_head
        - L_weights: The m-dim vector of weights to use for the LF-head
                loss weighting; defaults to all 1's.
        - middle_modules: (nn.Module) a list of modules to execute between the
            input_module and task head. Defaults to nn.Linear.
        - head_module: (nn.Module) a module to execute right before the final
            softmax that outputs a prediction for the task.
    """
    
    def __init__(self,
        input_module,
        accs,
        r=1,
        rw=False,
        L_weights=None,
        middle_modules=None,
        **kwargs
    ):
        
        self.m = len(accs) # number of labeling sources
        self.r = r
        self.rw = rw
        self.output_dim = 2 # Fixed for binary setting

        # No bias-- only learn weights for L_head
        head_module = nn.Linear(self.r, self.m, bias=False)

        # Initialize EndModel.
        # Note: We overwrite `self.k` which conventionally refers to the 
        # number of tasks to `self.m` which is the number of LFs in our setting.
        super().__init__([self.r, self.m], # layer_out_dims
                         input_module,
                         middle_modules, 
                         head_module,
                         verbose=False, # don't print EndModel params
                         **kwargs)
        
        # Set to "verbose" by default
        self.update_config({"verbose": kwargs.get("verbose", True) if kwargs else True})

        # Redefine loss fn
        self.criteria = nn.BCEWithLogitsLoss(reduce=False)
        
        # For manually reweighting. Default to all ones.
        if L_weights is None:
            self.L_weights = torch.ones(self.m).reshape(-1, 1)
        else:
            self.L_weights = torch.from_numpy(L_weights).reshape(-1, 1)
        
        # Set network and L_head modules
        modules = list(self.network.children())
        self.network = nn.Sequential(*list(modules[:-1]))
        self.L_head = modules[-1]

        # Attach the "DP head" which outputs the final prediction
        y_d = 2 * self.r if self.rw else self.r
        self.Y_head = nn.Linear(y_d, self.output_dim, bias=False)
        
        # Start by getting the DP marginal probability of Y=1, using the
        # provided LF accuracies, accs, and assuming cond. ind., binary LFs
        accs = np.array(accs, dtype=np.float32)
        self.w = torch.from_numpy(np.log(accs / (1-accs))).float()
        
        if self.config["use_cuda"]:
            self.L_weights = self.L_weights.cuda()
            self.w = self.w.cuda()
        
        if self.config["verbose"]:
            print ("Slice Heads:")
            print ("Input Network:", self.network)
            print ("L_head:", self.L_head)
            print ("Y_head:", self.Y_head)
        
  
    def _loss(self, X, L):
        """Returns the loss consisting of summing the LF + DP head losses
        
        Args:
            - X: A [batch_size, d] torch Tensor
            - L: A [batch_size, m] torch Tensor with elements in {-1,0,1}
        """
        L_01 = (L + 1) / 2
        # LF heads loss
        # NOTE: Here, we only add non-abstains to the loss
#         L_mask = torch.abs(L_01)
#         nb = torch.sum(L_mask)
        # loss_1 = torch.sum(self.loss_fn(self.forward_L(x), L_01) * L_mask) / nb
        # NOTE: Here, we add *all* data points to the loss       
        loss_1 = torch.mean(
            self.criteria(self.forward_L(X), L_01) @ self.L_weights
        )

        # Compute the noise-aware DP loss w/ the reweighted representation
        # Note: Need to convert L from {0,1} --> {-1,1}
        label_probs = F.sigmoid(2 * L @ self.w).reshape(-1, 1)
        multiclass_labels = torch.cat((label_probs, 1-label_probs), dim=1)
        loss_2 = torch.mean(
            self.criteria(self.forward_Y(X), multiclass_labels)
        )
        
        # Just take the unweighted sum of these for now...
#         return (10*loss_1 + loss_2) / 2
        return (loss_1 + 10*loss_2) / 2

    
    def _get_loss_fn(self):
        """ Override `EndModel` loss function with custom L_head + Y_head loss"""
        return self._loss
        
    def forward_L(self, x):
        """Returns the unnormalized predictions of the L_head layer."""
        return self.L_head(self.network(x))
    
    def forward_Y(self, x):
        """Returns the output of the Y head only, over re-weighted repr."""
        b = x.shape[0]
        xr = self.network(x)

        # Concatenate with the LF attention-weighted representation as well
        if self.rw:

            # A is the [bach_size, 1, m] Tensor representing the relative
            # "confidence" of each LF on each example
            # NOTE: Should we be taking an absolute value / centering somewhere
            # before here to capture the "confidence" vs. prediction...?
            A = F.softmax(self.forward_L(x)).unsqueeze(1)

            # We then project the A weighting onto the respective features of
            # the L_head layer, and add these attention-weighted features to Xr
            W = self.L_head.weight.repeat(b, 1, 1)
            xr = torch.cat([xr, torch.bmm(A, W).squeeze()], 1)

        # Return the list of head outputs + DP head
        return self.Y_head(xr).squeeze()

    def predict_proba(self, x):
        return F.sigmoid(self.forward_Y(x)).data.cpu().numpy()

    def score_on_LF_slices(self, X, Y, L):
        """Return the score for each coverage set of each LF"""
        m = L.shape[1]
        X_eval = torch.from_numpy(X.astype(np.float32)) 
        Yp = np.tile(self.predict(X_eval), (m, 1))
        Yp[Yp==2] = -1 
        Yp = np.abs(L).T * Yp
        return 0.5 * (Yp @ Y / np.sum(np.abs(L), axis=0) + 1)

