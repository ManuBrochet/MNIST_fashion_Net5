import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_Riem_opti, utils_math

# ====================================================================================
# 1. Définition du réseau
# ====================================================================================

# Défninition des couches cachées avec les poids : W = U * Sigma * V.T
# dim(Sigma) = r, dim(U) = (in_features, r), dim(V) = (out_features, r)
class ReducedLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_size, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Q parameter (orthogonal part)
        self.U = nn.Parameter(torch.randn(out_features, sigma_size) * 0.05)
        self.V = nn.Parameter(torch.randn(in_features, sigma_size) * 0.05)

        # diagonal scaling
        self.log_Sigma = nn.Parameter(torch.randn(sigma_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None


    def weight(self):
        # W = QD
        Sigma = torch.exp(self.log_Sigma)
        return self.U @ torch.diag(Sigma) @ self.V.T

    def forward(self, x):

        W = self.weight()

        return F.linear(
            x,
            W,
            self.bias
        )


class LeNet5(nn.Module):

    def __init__(self, rank_fc, optimizer, taille_couche1 = 120, taille_couche2 = 84):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # 16 x 4 x 4 = 256

        if optimizer == "Reduced_network":
            self.fc1 = ReducedLinear(
                256,
                taille_couche1,
                rank_fc[0]
            )

            self.fc2 = ReducedLinear(
                taille_couche1,
                taille_couche2,
                rank_fc[1]
            )

            self.fc3 = ReducedLinear(
                taille_couche2,
                10,
                rank_fc[2]
            )

        elif optimizer == "Pytorch":
            self.fc1 = nn.Linear(
                256,
                taille_couche1
            )

            self.fc2 = nn.Linear(
                taille_couche1,
                taille_couche2
            )

            self.fc3 = nn.Linear(
                taille_couche2,
                10
            )

    def forward(self, x):

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# ====================================================================================
# 2. Optimizer dédiés
# ====================================================================================

def reduced_network_optimizer(
    model, learning_rate, LR_UV, beta_momentum, use_momentum = True, first_iteration = True
    ):

    with torch.no_grad():
        for name, param in model.named_parameters():

            # Si pas de gradients, pas d'optimisation
            if param.grad is None:
                continue

            if use_momentum:
                # ----------------------------------------
                # 1. Momentum buffer (créé si inexistant)
                # ----------------------------------------
                if not hasattr(param, "momentum_buffer"):
                    param.momentum_buffer = torch.zeros_like(param)
                momentum = param.momentum_buffer
            else: momentum = None

            # Only Stiefel optimize Q
            if "U" in name or "V" in name:

                X = param
                G = param.grad

                X_perp = utils_math.compute_X_perp_torch(X)

                # Stiefel update
                if use_momentum:
                    X_new, momentum = utils_Riem_opti.stiefel_step_torch_momentum(
                        X, X_perp, G, momentum, LR_UV, beta_momentum, first_iteration
                        )
                else:
                    X_new = utils_Riem_opti.stiefel_step_torch(X, X_perp, G, LR_UV)

                param.copy_(X_new)

                param.grad.zero_()

                if use_momentum:
                    param.momentum_buffer.copy_(momentum)

            else :
                utils_math.opti_euclidienn(
                    param=param, learning_rate=learning_rate, use_momentum=use_momentum,
                    momentum=momentum, beta_momentum=beta_momentum
                    )
                