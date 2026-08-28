import torch

def opti_euclidienn(param, learning_rate, use_momentum, momentum, beta_momentum,
                     clamp_range=None):
    "MAJ des poids gradient euclidien"
    if use_momentum:
        vect_descente =  beta_momentum * momentum + (1-beta_momentum) * param.grad
    else :
        vect_descente = param.grad
    param -= learning_rate * vect_descente
    if clamp_range is not None:
        # Prevents runaway growth of exponential-domain parameters
        # (log_Sigma): without this, d(Sigma)/d(log_Sigma) = Sigma makes
        # the update self-reinforcing once log_Sigma drifts in one
        # direction, which is what causes the seed-dependent blow-ups.
        param.clamp_(*clamp_range)
    if use_momentum:
        param.momentum_buffer.copy_(vect_descente)
    param.grad.zero_()


def log_training_diagnostics(model):
    """
    Snapshot of the quantities that matter for diagnosing the
    instability: Sigma's scale, the gradient norm feeding log_Sigma,
    how far U/V have drifted from orthonormality, and whether any
    parameter has already gone non-finite.

    Call this once per batch (or per epoch), right after
    loss.backward() and BEFORE the optimizer step, so the gradients
    reflect the step that's about to be taken. Cheap enough to run
    every batch for a single debug seed; keep it off (cfg
    "debug_diagnostics": False) for full benchmark sweeps.
    """
    diag = {}
    for name, param in model.named_parameters():

        if "log_Sigma" in name:
            with torch.no_grad():
                sigma = torch.exp(param)
            diag[f"{name}/sigma_max"] = sigma.max().item()
            diag[f"{name}/sigma_mean"] = sigma.mean().item()
            diag[f"{name}/logsigma_grad_norm"] = (
                param.grad.norm().item() if param.grad is not None else float("nan")
            )

        elif name.endswith(".U") or name.endswith(".V"):
            with torch.no_grad():
                gram = param.T @ param
                eye = torch.eye(gram.shape[0], device=param.device, dtype=param.dtype)
                diag[f"{name}/ortho_residual"] = (gram - eye).norm().item()
            diag[f"{name}/grad_norm"] = (
                param.grad.norm().item() if param.grad is not None else float("nan")
            )

        if not torch.isfinite(param).all():
            diag[f"{name}/has_nonfinite"] = True

    return diag


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
