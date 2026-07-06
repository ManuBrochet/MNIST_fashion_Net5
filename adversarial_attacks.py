"""
adversarial_attacks.py
=======================

Attaques adversariales white-box "simples" (FGSM / PGD, denses ou éparses)
pour évaluer la robustesse de modèles LeNet5 sauvegardés avec
utils_pytorch.save_mlp().

Le "budget" d'une attaque est défini par deux quantités, comme demandé :
  - epsilon  : amplitude maximale de modification autorisée par pixel touché
               (norme L∞, par pixel)
  - n_pixels : nombre de pixels spatiaux (H×W) que l'attaque a le droit de
               modifier. None (ou >= H*W) => attaque dense (tous les pixels
               peuvent être modifiés).

Toutes les attaques sont white-box : les gradients sont calculés directement
à partir des paramètres du modèle (torch.autograd), sans aucune approximation
ni requête boîte noire. L'objectif est simplement de faire changer la
prédiction (attaque non-ciblée).

Utilisation typique
--------------------
    from adversarial_attacks import run_adversarial_experiment

    run_adversarial_experiment(
        checkpoint_paths=[
            "checkpoints/CIFAR10_Pytorch.pth",
            "checkpoints/CIFAR10_Reduced_network.pth",
        ],
        n_images=200,
        epsilons=[0.01, 0.05, 0.1, 0.2],
        n_pixels_list=[None, 50, 100, 300],
        output_csv="results/adv_results.csv",
    )
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
    """
    Retourne (min_val, max_val), deux tenseurs broadcastables sur (C,H,W),
    donnant la plage de valeurs atteignables par un pixel réel (dans [0, 1]
    avant normalisation) une fois passé par la normalisation utilisée dans
    load_data.py. Sert à garder l'image adversariale "physiquement valide".
    """
    if dataset_name in ("CIFAR10", "CIFAR100"):
        mean = torch.tensor(_CIFAR_MEAN, device=device).view(-1, 1, 1)
        std  = torch.tensor(_CIFAR_STD,  device=device).view(-1, 1, 1)
        min_val = (0.0 - mean) / std
        max_val = (1.0 - mean) / std
    else:
        # MNIST_fashion : simple ToTensor(), pixels dans [0, 1]
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
# 2. Attaque : PGD éparse (dense si n_pixels=None) — cœur de l'algorithme
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_mask(grad, n_pixels):
    """
    Construit un masque (B, 1, H, W) sélectionnant, pour chaque image du
    batch, les n_pixels pixels spatiaux ayant le plus grand gradient (valeur
    absolue, sommée sur les canaux). C'est ce masque qui matérialise le
    budget "nombre de pixels modifiables".

    Si n_pixels vaut None ou dépasse le nombre de pixels de l'image, le
    masque est entièrement à 1 (attaque dense, tous les pixels autorisés).
    """
    B, C, H, W = grad.shape
    n_spatial = H * W

    if n_pixels is None or n_pixels >= n_spatial:
        return torch.ones(B, 1, H, W, device=grad.device)

    grad_importance = grad.abs().sum(dim=1).view(B, n_spatial)  # (B, H*W)
    topk_idx = grad_importance.topk(n_pixels, dim=1).indices    # (B, n_pixels)

    mask_flat = torch.zeros(B, n_spatial, device=grad.device)
    mask_flat.scatter_(1, topk_idx, 1.0)

    return mask_flat.view(B, 1, H, W)


def pgd_attack(model, images, labels, epsilon, n_pixels=None,
               n_steps=10, step_size=None, dataset_name="CIFAR10"):
    """
    Attaque white-box de type PGD (Projected Gradient Descent), avec un
    budget L∞ (epsilon, par pixel touché) restreint à `n_pixels` pixels
    spatiaux au maximum.

    - n_pixels=None (ou >= H*W)        -> attaque dense (PGD classique)
    - n_steps=1 et step_size=epsilon   -> équivaut à une FGSM (éventuellement
                                           éparse si n_pixels est précisé)

    Le masque des pixels "autorisés" est calculé une seule fois (à partir du
    gradient de la première itération) : c'est le choix le plus simple pour
    respecter un budget fixe de pixels tout au long des itérations.

    Retourne
    --------
    images_adv : images perturbées, mêmes bornes de valeurs que le dataset
                 d'origine (donc directement utilisables par le modèle)
    """
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

            # Projection L∞ : la perturbation totale reste bornée par epsilon
            perturbation = torch.clamp(images_adv - images_orig, -epsilon, epsilon)
            images_adv = images_orig + perturbation * mask

            # L'image reste dans la plage de valeurs physiquement valides
            images_adv = torch.max(torch.min(images_adv, max_val), min_val)

        images_adv = images_adv.detach()

    return images_adv


def fgsm_attack(model, images, labels, epsilon, n_pixels=None, dataset_name="CIFAR10"):
    """Cas particulier du PGD à un seul pas : Fast Gradient Sign Method."""
    return pgd_attack(
        model, images, labels, epsilon,
        n_pixels=n_pixels, n_steps=1, step_size=epsilon,
        dataset_name=dataset_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boucle d'expérience : plusieurs modèles × plusieurs budgets → CSV
# ─────────────────────────────────────────────────────────────────────────────

def _get_clean_subset(test_loader, model, device, n_images):
    """
    Récupère `n_images` images du set de test qui sont CORRECTEMENT
    classifiées par le modèle avant attaque. Attaquer une image déjà mal
    classée n'a pas de sens ici puisque le but est de mesurer la capacité de
    l'attaque à "faire changer d'avis" le modèle.
    """
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
    """
    Évalue plusieurs modèles sauvegardés (`checkpoint_paths`) contre des
    attaques adversariales white-box PGD/FGSM, denses ou éparses, pour
    différentes combinaisons de budget (epsilon, n_pixels), sur le même jeu
    d'images de test (correctement classées à l'origine). Écrit les
    résultats détaillés (un succès/échec par image et par budget) dans un
    fichier CSV.

    Paramètres
    ----------
    checkpoint_paths : liste de chemins vers des .pth sauvegardés avec
                        utils_pytorch.save_mlp
    n_images         : nombre d'images de test utilisées par modèle
    epsilons         : liste des amplitudes de perturbation à tester
    n_pixels_list    : liste des budgets "nombre de pixels" à tester
                        (mettre None dans la liste pour une attaque dense)
    n_steps          : nombre d'itérations PGD (1 = FGSM)
    output_csv       : chemin du fichier CSV de sortie

    Colonnes du CSV
    ---------------
    model_name, dataset, optimizer_choice, attack_type,
    epsilon, n_pixels, image_idx, true_label, adv_label, success
    """
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
    # Exemple d'utilisation — à adapter avec vos propres checkpoints
    # (générés en passant save_model=True à Run_experiment.run_experiment)
    run_adversarial_experiment(
        # checkpoint_paths=[
        #     "checkpoints/MNIST_fashion_Pytorch.pth",
        #     "checkpoints/MNIST_fashion_Reduced_network.pth",
        #     "checkpoints/MNIST_fashion_no_constraints.pth",
        # ],
        checkpoint_paths= ["checkpoints/CIFAR10_Reduced_network_momentum_True_adaptStep_True_SigSize_40_24_4.pth"],
        n_images=200,
        epsilons=[0.01, 0.05, 0.1, 0.2],
        #n_pixels_list=[None, 20, 50, 100],
        n_pixels_list=[None],
        output_csv="results/adv_results.csv",
    )
