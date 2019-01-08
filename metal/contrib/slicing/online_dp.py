import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model.end_model import EndModel
from metal.end_model.em_defaults import em_default_config
from metal.metrics import metric_score
from metal.utils import recursive_merge_dicts

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
        - slice-weight: Factor to multiply loss for L_heads in joint loss with
                Y_head loss
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
        slice_weight=0.5,
        middle_modules=None,
        verbose=True,
        **kwargs
    ):
        
        self.m = len(accs) # number of labeling sources
        self.r = r
        self.rw = rw
        self.output_dim = 2 # NOTE: Fixed for binary setting
        self.slice_weight = slice_weight

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
        
        # Set reset "verbose" in config
        self.update_config({"verbose": verbose})

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
        self.w[np.abs(self.w) == np.inf] = 0 # set weights from acc==0 to 0
        
        if self.config["use_cuda"]:
            self.L_weights = self.L_weights.cuda()
            self.w = self.w.cuda()
        
        if self.config["verbose"]:
            print ("Slice Heads:")
            print ("Reweighting:", self.rw)
            print ("Slice Weight:", self.slice_weight)
            print ("Input Network:", self.network)
            print ("L_head:", self.L_head)
            print ("Y_head:", self.Y_head)
  
    def _loss(self, X, L, Y_tilde=None):
        """Returns the loss consisting of summing the LF + DP head losses
        
        Args:
            - X: A [batch_size, d] torch Tensor
            - L: A [batch_size, m] torch Tensor with elements in {-1,0,1}
        """
        L_01 = (L + 1) / 2
        # LF heads loss
        # NOTE: Here, we add *all* data points (incl. abstains) to the loss
        loss_1 = torch.mean(
            self.criteria(self.forward_L(X), L_01) @ self.L_weights
        )
        
        # Compute the noise-aware DP loss w/ the reweighted representation
        if Y_tilde is None:
            # Note: Need to convert L from {0,1} --> {-1,1}
            label_probs = F.sigmoid(2 * L @ self.w).reshape(-1, 1)
            Y_tilde = torch.cat((label_probs, 1-label_probs), dim=1)

        loss_2 = torch.mean(
            self.criteria(self.forward_Y(X), Y_tilde)
        )
        
        # Compute the weighted sum of these
        loss_1 /= self.m # normalize by number of LFs
        return (self.slice_weight*loss_1) + ((1-self.slice_weight)*loss_2)

    
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
        outputs = self.Y_head(xr).squeeze()
        return F.softmax(outputs)

    def predict_proba(self, x):
        return self.forward_Y(x).data.cpu().numpy()

    def score_on_slice(
        self, 
        data, 
        selected_idx,
        metric=["f1"], 
        break_ties="random", 
        verbose=True, 
        **kwargs
    ):
        """Returns the slice-specific score as defined by given indexes.
        
        Args:
            data: a pytorch DataLoader, Dataset or tuple with Tensors (X,Y)
            metric: A metric (string) with which to score performance or a list
                of such metrics
            verbose: The verbosity for this score method; it will not update the
                class config
    
        Returns:
            scores: A (float) score of list of such scores if kwarg metric 
                is a list
        """

        # no overlap, return 1.0
        if len(selected_idx) == 0:
            scores = [1.0] * len(metric)

        # otherwise, compute score at overlap
        else:
            # Filter preds/gt by selected_idx
            Y_p, Y, Y_s = self._get_predictions(
                data, break_ties=break_ties, return_probs=True, **kwargs
            ) 
            Y_p, Y, Y_s = Y_p[selected_idx], Y[selected_idx], Y_s[selected_idx]
    
            # Evaluate on selected metrics
            metric_list = metric if isinstance(metric, list) else [metric]
            scores = []
            for metric in metric_list:
                score = metric_score(Y, Y_p, metric, probs=Y_s, ignore_in_gold=[0]) 
                scores.append(score)
    
        if isinstance(scores, list) and len(scores) == 1:
            return scores[0]
        else:
            return scores

