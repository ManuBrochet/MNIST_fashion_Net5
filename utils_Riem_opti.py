import torch

# ====================================================================================
# Optimizer steps for Riemannian optimization
# ====================================================================================

def stiefel_step_torch(X, X_perp, G_euc, step):

    # # ------------------
    # XtG = X.T @ G_euc
    # sym = 0.5 * (XtG + XtG.T)
    # G = G_euc - X @ sym   # Riemannian gradient
    # # ------------------
    G = - G_euc

    # print(G - G_euc)

    A = X.T @ G - G.T @ X           # (p,p)
    B = X_perp.T @ G                # (n-p,p)

    p = A.shape[0]
    n_p = B.shape[0]

    # Build K
    K = torch.zeros((p + n_p, p + n_p), device=X.device, dtype=X.dtype)

    K[:p, :p] = A
    K[:p, p:] = -B.T
    K[p:, :p] = B

    # Exponential
    E = torch.matrix_exp(step * K)

    # Extract blocks : we take to :p because of the I_n*p matrix
    E11 = E[:p, :p]
    E21 = E[p:, :p]

    # Update WITHOUT forming Q
    X_new = X @ E11 + X_perp @ E21

    return X_new

def stiefel_step_torch_momentum(X, X_perp, G_euc, momentum, step, beta_momentum, first_iteration):

    # # ------------------
    # XtG = X.T @ G_euc
    # sym = 0.5 * (XtG + XtG.T)
    # G = G_euc - X @ sym   # Riemannian gradient
    # # ------------------
    G = - G_euc

    # print(G - G_euc)

    A = X.T @ G - G.T @ X           # (p,p)
    B = X_perp.T @ G                # (n-p,p)

    if first_iteration:
        A_final = A
        B_final = B
    
    else:
        # grad_riem = X @ A + X_perp @ B
        # print("La shape du momentum : ", momentum.shape)

        momentum_transported = momentum - X @ (X.T @ momentum + momentum.T @ X) / 2

        A_transported = X.T @ momentum_transported
        B_transported = X_perp.T @ momentum_transported

        A_final = beta_momentum * A_transported + (1-beta_momentum) * A
        B_final = beta_momentum * B_transported + (1-beta_momentum) * B

    # Maj momentum : prise en compte du grad riem que l'on vient de calculer
    momentum = X @ A_final + X_perp @ B_final

    p = A_final.shape[0]
    n_p = B_final.shape[0]

    # Build K
    K = torch.zeros((p + n_p, p + n_p), device=X.device, dtype=X.dtype)

    K[:p, :p] = A_final
    K[:p, p:] = -B_final.T
    K[p:, :p] = B_final

    # Exponential
    E = torch.matrix_exp(step * K)

    # Extract blocks : we take to :p because of the I_n*p matrix
    E11 = E[:p, :p]
    E21 = E[p:, :p]

    # Update WITHOUT forming Q
    X_new = X @ E11 + X_perp @ E21

    return X_new, momentum


# ====================================================================================
# Initialisation
# ====================================================================================

def project_to_stiefel(X):

    if X.shape[0] >= X.shape[1]:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return U @ Vh

    else:
        U, _, Vh = torch.linalg.svd(X.T, full_matrices=False)
        return (U @ Vh).T

def initialize_reduced_model(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            
            # Only project weight matrices
            if ("U" in name or "V" in name) and param.ndim == 2:
                param.copy_(project_to_stiefel(param))