import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


class Classifier(nn.Module):
    def predict(self, x):
        yp = self.predict_proba(x).squeeze().detach().numpy()
        return np.where(yp > 0.5, 1, -1)

    def score(self, X_np, Y_np):
        X = self._convert_np_data(X_np)
        return np.where(self.predict(X) == Y_np, 1, 0).sum() / X.shape[0]

    def score_on_LF_slices(self, X_np, Y_np, L_np):
        """Return the score for each coverage set of each LF"""
        m = L_np.shape[1]
        Yp = np.tile(self.predict(self._convert_np_data(X_np)), (m, 1))
        Yp = np.abs(L_np).T * Yp
        return 0.5 * (Yp @ Y_np / np.sum(np.abs(L_np), axis=0) + 1)

    def train(
        self,
        X_np,
        L_np,
        batch_size=10,
        n_epochs=10,
        lr=0.01,
        momentum=0.9,
        print_every=10,
    ):
        """Train a standard supervised model using SGD with momentum."""
        X, L = map(self._convert_np_data, [X_np, L_np])

        # Create DataLoader
        train_loader = DataLoader(TensorDataset(X, L), batch_size=batch_size)

        # Set optimizer as SGD w/ momentum
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        # Train model
        for epoch in range(n_epochs):
            running_loss = 0.0
            for batch, data in enumerate(train_loader):

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                loss = self.loss(*data)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach()

            # Print loss every 10 epochs
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                avg_loss = running_loss / batch
                print(f"[Epoch {epoch}] Loss: {avg_loss:0.3f}")

        print("Finished Training")

    def _convert_np_data(self, X_np, convert_binary=False):
        X_np = np.copy(X_np)

        # Optionally: Convert from {-1,1} --> {0,1}
        if convert_binary:
            X_np[X_np == -1] = 0

        return torch.from_numpy(X_np).float()

    def print_params(self):
        for name, param in self.named_parameters():
            print("\n", name, param)


class SliceDPModel(Classifier):
    def __init__(
        self,
        input_dim,
        input_module_class,
        m,
        accs,
        r=1,
        rw=False,
        L_weights=[],
    ):
        """Online / joint data programming model
        Assumes balanced, binary class problem with conditionally ind. LFs that
        output binary labels or abstain, \lambda_i \in {-1,0,1}
        Args:
            - input_dim: Input data vector dimension
            - input_module_class: Class that initializes with args (input_dim,
                output_dim)
            - m: Number of label sources
            - accs: The LF accuracies, computed offline
            - r: Intermediate representation dimension
            - rw: Whether to use reweighting of representation for Y_head
            - L_weights: If provided, manually weight L_heads using L_weights
        """
        super().__init__()
        self.k = 1  # Fixed here for binary setting
        self.m = m
        self.r = r
        self.rw = rw
        self.L_weights = torch.from_numpy(np.array(L_weights, dtype=np.float32))

        # Basic binary loss function
        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False)

        # Input module
        self.input_layer = nn.Sequential(
            input_module_class(input_dim, self.r), nn.Sigmoid()
        )

        # Attach an [r, m] linear layer to predict the labels of the LFs
        self.L_head = nn.Linear(self.r, self.m, bias=False)

        # Attach the "DP head" which outputs the final prediction
        y_d = 2 * self.r if self.rw else self.r
        self.Y_head = nn.Linear(y_d, self.k, bias=False)

        # Start by getting the DP marginal probability of Y=1, using the
        # provided LF accuracies, accs, and assuming cond. ind., binary LFs
        self.w = torch.from_numpy(np.log(accs / (1 - accs))).float()

    def forward_L(self, x):
        """Returns the unnormalized predictions of the L_head layer."""
        return self.L_head(self.input_layer(x))

    def forward_Y(self, x):
        """Returns the output of the Y head only, over re-weighted repr."""
        b = x.shape[0]
        xr = self.input_layer(x)

        # Concatenate with the LF attention-weighted representation as well
        if self.rw:

            # A is the [bach_size, 1, m] Tensor representing the relative
            # "confidence" of each LF on each example
            # NOTE: Should we be taking an absolute value / centering somewhere
            # before here to capture the "confidence" vs. prediction...?
            A = F.softmax(self.forward_L(x)).unsqueeze(1)

            # We then project the A weighting onto the respective features of
            # the L_head layer, and add these attention-weighted features to Xr
            if self.L_weights.shape[0] > 0:
                # Manually reweight
                W = self.L_weights.repeat(4, 1).transpose(0, 1).repeat(b, 1, 1)
            else:
                # Use learned weights from L_head
                W = self.L_head.weight.repeat(b, 1, 1)

            xr = torch.cat([xr, torch.bmm(A, W).squeeze()], 1)

        # Return the list of head outputs + DP head
        return self.Y_head(xr).squeeze()

    def loss(self, x, L):
        """Returns the loss consisting of summing the LF + DP head losses
        Args:
            - x: A [batch_size, d] torch Tensor
            - L: A [batch_size, m] torch Tensor with elements in {-1,0,1}
        """

        # Convert label matrix to [0,1] scale, and create abstain mask
        L_01 = (L + 1) / 2

        # LF heads loss
        # NOTE: Here, we only add non-abstains to the loss
        # L_mask = torch.abs(L_01)
        # nb = torch.sum(L_mask)
        # loss_1 = torch.sum(self.loss_fn(self.forward_L(x), L_01) * L_mask) / nb
        # NOTE: Here, we add *all* data points to the loss
        loss_1 = torch.mean(self.loss_fn(self.forward_L(x), L_01))

        # Compute the noise-aware DP loss w/ the reweighted representation
        # Note: Need to convert L from {0,1} --> {-1,1}
        loss_2 = torch.mean(
            self.loss_fn(self.forward_Y(x), F.sigmoid(2 * L @ self.w))
        )

        # Just take the unweighted sum of these for now...
        return (10 * loss_1 + loss_2) / 2

    def predict_proba(self, x):
        return F.sigmoid(self.forward_Y(x))
