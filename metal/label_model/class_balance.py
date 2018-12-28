from itertools import product

import numpy as np
import torch
from torch import nn, optim


class ClassBalanceModel(nn.Module):
    """A model for learning the class balance, P(Y=y), given a subset of LFs
    which are *conditionally independent*, i.e. \lambda_i \perp \lambda_j | Y,
    for  i != j.

    Learns the model using a tensor factorization approach.

    Note: This approach can also be used for estimation of the LabelModel, may
    want to later refactor and expand this class.
    """

    def __init__(self, k, config=None):
        super().__init__()
        self.config = config
        self.k = k

        # Estimated quantities (np.array)
        self.cond_probs = None
        self.class_balance = None

    def _form_overlaps_tensor(self, L):
        """Transforms the input label matrix to a three-way overlaps tensor.

        Args:
            L: (np.array) An n x m array of LF output labels, in {0,...,k}, from
                m conditionally independent LFs on n data points

        Outputs:
            O: (torch.Tensor) A m x m x m x k x k x k (6-dim) tensor of the the
                label-specific overlap rates.
        """
        # TODO
        pass

    @staticmethod
    def get_loss(O, Q, mask):
        # Main constraint: match empirical three-way overlaps matrix
        # (entries O_{ijk} for i != j != k)
        diffs = (O - torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q]))[mask]
        return torch.norm(diffs) ** 2

    def train_model(self, L=None, O=None, lr=1):

        # Get overlaps tensor if L provided else use O directly (e.g. for tests)
        if O is not None:
            pass
        elif L is not None:
            O = self._form_overlaps_tensor(L)
        else:
            raise ValueError("L or O required as input.")
        self.m = O.shape[0]

        # Compute mask
        self.mask = torch.ones(
            (self.m, self.m, self.m, self.k, self.k, self.k)
        ).byte()
        for i, j, k in product(range(self.m), repeat=3):
            if len(set((i, j, k))) < 3:
                self.mask[i, j, k, :, :, :] = 0

        # Initialize parameters
        self.Q = nn.Parameter(
            torch.from_numpy(np.random.rand(self.m, self.k, self.k)).float()
        ).float()

        # Use L-BFGS here
        # Seems to be a tricky problem for simple 1st order approaches, and
        # small enough for quasi-Newton... L-BFGS seems to work well here
        optimizer = optim.LBFGS([self.Q], lr=lr, max_iter=1000)

        # The closure computes the loss
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(O, self.Q, self.mask)
            loss.backward()
            print(f"Loss: {loss.detach():.8f}")
            return loss

        # Perform optimizer step
        optimizer.step(closure)

        # Recover the class balance
        # Note that the columns are not necessarily ordered correctly at this
        # point, since there's a column-wise symmetry remaining
        q = self.Q.detach().numpy()
        p_y = np.mean(q.sum(axis=1) ** 3, axis=0)

        # Resolve remaining col-wise symmetry
        # We do this by first estimating the conditional probabilities (accs.)
        # P(\lambda_i = y' | Y = y) of the labeling functions, *then leveraging
        # the assumption that they are better than random* to resolve col-wise
        # symmetries here
        # Note we then store both the estimated conditional probs, and the class
        # balance

        # Recover the estimated cond probs: Q = C(P^{1/3}) --> C = Q(P^{-1/3})
        cps = q @ np.diag(1 / p_y ** (1 / 3))

        # Re-order cps and p_y using assumption and store np.array values
        # TODO: Take the *most common* ordering
        col_order = cps[0].argmax(axis=0)
        self.class_balance = p_y[col_order]
        self.cond_probs = cps[col_order]
