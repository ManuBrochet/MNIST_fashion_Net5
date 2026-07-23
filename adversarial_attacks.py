"""
adversarial_attacks.py
=======================

Attaques adversariales white-box "simples" (FGSM / PGD, denses ou éparses)
pour évaluer la robustesse de modèles LeNet5.
Inclut le calcul de la constante de Lipschitz pour la partie MLP via l'algorithme SeqLip.

Le "budget" d'une attaque est défini par deux quantités, comme demandé :
  - epsilon  : amplitude maximale de modification autorisée par pixel touché
               (norme L∞, par pixel)
  - n_pixels : nombre de pixels spatiaux (H×W) que l'attaque a le droit de
               modifier. None (ou >= H*W) => attaque dense.
"""

import os
import itertools

import torch
import torch.nn as nn
import pandas as pd

import load_data
import utils_pytorch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bornes valides des pixels selon le dataset (pour le clamp final)
# ─────────────────────────────────────────────────────────────────────────────

_CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR_STD  = (0.2675, 0.2565, 0.2761)


def _valid_range(dataset_name, device):
    if dataset_name in ("CIFAR10", "CIFAR100"):
        mean = torch.tensor(_CIFAR_MEAN, device=device).view(-1, 1, 1)
        std  = torch.tensor(_CIFAR_STD,  device=device).view(-1, 1, 1)
        min_val = (0.0 - mean) / std
        max_val = (1.0 - mean) / std
    else:
        min_val = torch.zeros(1, 1, 1, device=device)
        max_val = torch.ones(1, 1, 1, device=device)

    return min_val, max_val


def _get_test_loader(dataset_name, batch_size):
    if dataset_name == "CIFAR10":
        _, test_loader = load_data.load_CIFAR_10(batch_size)
    elif dataset_name == "CIFAR100":
        _, test_loader = load_data.load_CIFAR_100(batch_size)
    else:
        _, test_loader = load_data.load_MNIST_fashion(batch_size)
    return test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attaques Adversariales
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_mask(grad, n_pixels):
    B, C, H, W = grad.shape
    n_spatial = H * W

    if n_pixels is None or n_pixels >= n_spatial:
        return torch.ones(B, 1, H, W, device=grad.device)

    grad_importance = grad.abs().sum(dim=1).view(B, n_spatial)
    topk_idx = grad_importance.topk(n_pixels, dim=1).indices

    mask_flat = torch.zeros(B, n_spatial, device=grad.device)
    mask_flat.scatter_(1, topk_idx, 1.0)

    return mask_flat.view(B, 1, H, W)


def pgd_attack(model, images, labels, epsilon, n_pixels=None,
               n_steps=10, step_size=None, dataset_name="CIFAR10"):
    device = images.device
    labels = labels.to(device)

    if step_size is None:
        step_size = epsilon if n_steps == 1 else epsilon / max(1, n_steps // 2)

    min_val, max_val = _valid_range(dataset_name, device)

    images_orig = images.clone().detach()
    images_adv  = images.clone().detach()

    criterion = nn.CrossEntropyLoss()
    mask = None

    for _ in range(n_steps):
        images_adv.requires_grad_(True)
        logits = model(images_adv)
        loss = criterion(logits, labels)
        grad = torch.autograd.grad(loss, images_adv)[0]

        if mask is None:
            mask = _pixel_mask(grad, n_pixels)

        with torch.no_grad():
            images_adv = images_adv + step_size * grad.sign() * mask
            perturbation = torch.clamp(images_adv - images_orig, -epsilon, epsilon)
            images_adv = images_orig + perturbation * mask
            images_adv = torch.max(torch.min(images_adv, max_val), min_val)

        images_adv = images_adv.detach()

    return images_adv


def fgsm_attack(model, images, labels, epsilon, n_pixels=None, dataset_name="CIFAR10"):
    return pgd_attack(
        model, images, labels, epsilon,
        n_pixels=n_pixels, n_steps=1, step_size=epsilon,
        dataset_name=dataset_name,
    )


def _get_clean_subset(test_loader, model, device, n_images):
    images_kept, labels_kept = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct_mask = preds == labels

            images_kept.append(images[correct_mask])
            labels_kept.append(labels[correct_mask])

            if sum(t.shape[0] for t in images_kept) >= n_images:
                break

    images_all = torch.cat(images_kept, dim=0)[:n_images]
    labels_all = torch.cat(labels_kept, dim=0)[:n_images]
    return images_all, labels_all


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calcul de la Constante de Lipschitz (AutoLip / SeqLip)
# ─────────────────────────────────────────────────────────────────────────────

def seqlip_pair_objective(layer1, layer2, sigma, is_first=False, is_last=False):
    """Calcule la matrice d'interaction SeqLip entre deux couches ReducedLinear."""
    U1 = layer1.U
    V2 = layer2.V
    Sigma1 = torch.exp(layer1.log_Sigma)
    Sigma2 = torch.exp(layer2.log_Sigma)
    
    Sigma1_tilde = Sigma1 if is_first else torch.sqrt(Sigma1)
    Sigma2_tilde = Sigma2 if is_last else torch.sqrt(Sigma2)
    
    M1 = U1 * Sigma1_tilde.unsqueeze(0) 
    M2 = V2 * Sigma2_tilde.unsqueeze(0) 
    
    diag_sigma = torch.diag(sigma)
    core_matrix = M2.T @ diag_sigma @ M1
    
    return torch.linalg.norm(core_matrix, ord=2)


def compute_mlp_lipschitz(model, device, n_steps=50, lr=0.1):
    """
    Estime la constante de Lipschitz du sous-réseau MLP (fc1, fc2, fc3).
    - Utilise Greedy SeqLip pour les couches ReducedLinear.
    - Utilise AutoLip pour les couches standard nn.Linear.
    """
    is_reduced = hasattr(model.fc1, 'U')

    if not is_reduced:
        # Borne AutoLip classique pour les modèles de référence (produit des normes spectrales)
        with torch.no_grad():
            L1 = torch.linalg.norm(model.fc1.weight, ord=2).item()
            L2 = torch.linalg.norm(model.fc2.weight, ord=2).item()
            L3 = torch.linalg.norm(model.fc3.weight, ord=2).item()
        return L1 * L2 * L3

    # Approche Greedy SeqLip pour les couches avec U, Sigma, V
    # --------------------------------------------------------
    
    # Paire 1 : fc1 -> fc2
    sigma1 = torch.rand(model.fc1.out_features, device=device, requires_grad=True)
    optimizer1 = torch.optim.Adam([sigma1], lr=lr)

    for _ in range(n_steps):
        optimizer1.zero_grad()
        with torch.no_grad():
            sigma1.clamp_(0, 1) # On restreint l'espace de recherche à la dérivée des activations (ReLU)
        
        # Maximisation de l'objectif d'alignement des vecteurs singuliers
        loss = -seqlip_pair_objective(model.fc1, model.fc2, sigma1, is_first=True, is_last=False)
        loss.backward()
        optimizer1.step()

    with torch.no_grad():
        sigma1.clamp_(0, 1)
        bound1 = seqlip_pair_objective(model.fc1, model.fc2, sigma1, is_first=True, is_last=False).item()

    # Paire 2 : fc2 -> fc3
    sigma2 = torch.rand(model.fc2.out_features, device=device, requires_grad=True)
    optimizer2 = torch.optim.Adam([sigma2], lr=lr)

    for _ in range(n_steps):
        optimizer2.zero_grad()
        with torch.no_grad():
            sigma2.clamp_(0, 1)
            
        loss = -seqlip_pair_objective(model.fc2, model.fc3, sigma2, is_first=False, is_last=True)
        loss.backward()
        optimizer2.step()

    with torch.no_grad():
        sigma2.clamp_(0, 1)
        bound2 = seqlip_pair_objective(model.fc2, model.fc3, sigma2, is_first=False, is_last=True).item()

    return bound1 * bound2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Boucle d'expérience globale
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_experiment(
    checkpoint_paths,
    n_images=200,
    epsilons=(0.01, 0.05, 0.1, 0.2),
    n_pixels_list=(None, 50, 100, 300),
    n_steps=10,
    batch_size=64,
    output_csv="results/adv_results.csv",
    device=None,
    verbose=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []

    for ckpt_path in checkpoint_paths:

        model, meta = utils_pytorch.load_lenet5(ckpt_path, device=device)
        cfg = meta["model_config"]
        dataset_name = cfg["dataset_name"]
        optimizer_choice = cfg["optimizer_choice"]
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

        if verbose:
            print(f"\n[Modèle] {model_name}  (dataset={dataset_name}, "
                  f"optimizer={optimizer_choice})")

        # Calcul et affichage de la constante de Lipschitz
        lip_cst = compute_mlp_lipschitz(model, device)
        if verbose:
            print(f"  -> Constante de Lipschitz (MLP) estimée : {lip_cst:.4e}")

        test_loader = _get_test_loader(dataset_name, batch_size)
        images, labels = _get_clean_subset(test_loader, model, device, n_images)

        for epsilon, n_pixels in itertools.product(epsilons, n_pixels_list):

            attack_type = "dense" if n_pixels is None else "sparse"

            images_adv = pgd_attack(
                model, images, labels, epsilon,
                n_pixels=n_pixels, n_steps=n_steps, dataset_name=dataset_name,
            )

            with torch.no_grad():
                adv_preds = model(images_adv).argmax(dim=1)

            success = (adv_preds != labels).cpu()

            for idx in range(images.shape[0]):
                rows.append({
                    "model_name":       model_name,
                    "dataset":          dataset_name,
                    "optimizer_choice": optimizer_choice,
                    "attack_type":      attack_type,
                    "epsilon":          epsilon,
                    "n_pixels":         n_pixels if n_pixels is not None else "all",
                    "lip_constant":     lip_cst,  # Ajout au CSV
                    "image_idx":        idx,
                    "true_label":       labels[idx].item(),
                    "adv_label":        adv_preds[idx].item(),
                    "success":          bool(success[idx].item()),
                })

            if verbose:
                rate = success.float().mean().item() * 100
                px_label = "toutes" if n_pixels is None else n_pixels
                print(f"  eps={epsilon:<6} pixels={px_label:<6} "
                      f"-> succès attaque: {rate:5.1f} %")

    df = pd.DataFrame(rows)

    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Résultats sauvegardés → {output_csv}")

    return df


if __name__ == "__main__":
    run_adversarial_experiment(
        checkpoint_paths= [
            "checkpoints_benchmark/CIFAR10_Adam_layerSize_69_40.pth",
            "checkpoints_benchmark/CIFAR10_Adam_layerSize_120_84.pth",
            "checkpoints_benchmark/CIFAR10_SGD_layerSize_69_40.pth",
            "checkpoints_benchmark/CIFAR10_SGD_layerSize_120_84.pth",
            "checkpoints_benchmark/CIFAR10_Reduced_network_momentum_True_adaptStep_False_SigSize_40_24_4.pth",
            "checkpoints_benchmark/CIFAR10_Reduced_network_momentum_True_adaptStep_True_SigSize_40_24_4.pth",
            "checkpoints_benchmark/CIFAR10_Reduced_network_iso_momentum_True_adaptStep_False_SigSize_40_24_4.pth"
            ],
        n_images=200,
        epsilons=[0.01, 0.05, 0.1, 0.2],
        n_pixels_list=[None],
        output_csv="results/adv_results.csv",
    )