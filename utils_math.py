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

def compute_dead_neuron_stats(model, dataloader):

    model.eval()

    device = next(model.parameters()).device

    dead_counts = {
        "fc1": None,
        "fc2": None,
    }

    total_samples = 0

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(device)

            _, (a1, a2) = model(images, return_activations=True)

            if dead_counts["fc1"] is None:
                dead_counts["fc1"] = torch.zeros(
                    a1.shape[1], device=device
                )
                dead_counts["fc2"] = torch.zeros(
                    a2.shape[1], device=device
                )

            dead_counts["fc1"] += (a1 == 0).sum(dim=0)
            dead_counts["fc2"] += (a2 == 0).sum(dim=0)

            total_samples += images.size(0)

    stats = {
        layer: counts / total_samples
        for layer, counts in dead_counts.items()
    }

    return stats
