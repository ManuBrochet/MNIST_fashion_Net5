import torch

def opti_euclidienn(param, learning_rate, use_momentum, momentum, beta_momentum):
    "MAJ des poids gradient euclidien"
    if use_momentum:
        vect_descente =  beta_momentum * momentum + (1-beta_momentum) * param.grad
    else :
        vect_descente = param.grad
    param -= learning_rate * vect_descente
    if use_momentum:
        param.momentum_buffer.copy_(vect_descente)
    param.grad.zero_()


def compute_X_perp_torch(X):
    # Full QR decomposition
    Q, _ = torch.linalg.qr(X, mode='complete')
    _, p = X.shape
    return Q[:, p:]