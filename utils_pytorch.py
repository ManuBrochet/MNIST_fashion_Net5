import torch
import os
from datetime import datetime

from building_network import LeNet5
import building_network

def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()

            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy



# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _extract_config(model: LeNet5) -> dict:
    """
    Déduit les arguments du constructeur en inspectant les types de couches.
    Fonctionne quelle que soit la variante entraînée.
    """
    from building_network import ReducedLinear

    reduced_network     = isinstance(model.fc2, ReducedLinear)

    # sigma_size : présent sur ReducedLinear et ReducedBisLinear seulement
    if reduced_network:
        sigma_size = model.fc2.log_Sigma.shape[0]
    else:
        sigma_size = 0

    # input_dim : récupéré depuis la première couche
    input_dim  = model.fc1.in_features   # fonctionne sur Linear et OrthogonalLinear
    hidden_size = model.fc2.out_features

    # input_dim=2 signifie entrée (x,y) → on conserve la valeur réelle
    return {
        "input_dim":           input_dim,
        "reduced_network":     reduced_network,
        "sigma_size":          sigma_size,
        "hidden_size":         hidden_size,
    }


def _mode_label(cfg: dict) -> str:
    if cfg.get("learn_QD"):
        return "OrthogonalLinear (QD)"
    if cfg.get("reduced_network"):
        return f"ReducedLinear (σ={cfg.get('sigma_size','')})"
    if cfg.get("reduced_bis_network"):
        return f"ReducedBisLinear (σ={cfg.get('sigma_size','')})"
    return "Linear standard"

def _print_summary(meta: dict) -> None:
    cfg = meta.get("model_config", {})
    lines = [
        f"  Date         : {meta.get('saved_at', '?')}",
        f"  Mode         : {_mode_label(cfg)}",
        f"  input_dim    : {cfg.get('input_dim', '?')}",
        f"  hidden_size  : {cfg.get('hidden_size', '?')}",
    ]
    if "loss"  in meta: lines.append(f"  Loss         : {meta['loss']:.6f}")
    if "epoch" in meta: lines.append(f"  Epoch        : {meta['epoch']}")
    if "image" in meta: lines.append(f"  Image        : {meta['image']}")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_mlp(model: building_network.LeNet5,
             filepath: str,
             meta: dict = None) -> None:
    """
    Sauvegarde un MLP_ImageCompressor_PT dans un fichier .pth.

    Ce qui est sauvegardé :
      - state_dict       : tous les paramètres entraînés (Q, d, U, V,
                           log_Sigma, A, D, weight, bias selon le mode)
      - model_config     : les 5 arguments du constructeur
                           (input_dim, reduced_network, reduced_bis_network,
                            learn_QD, sigma_size, hidden_size)
      - meta             : métriques et informations libres sur l'entraînement

    Paramètres
    ----------
    model    : instance de MLP_ImageCompressor_PT après entraînement
    filepath : chemin de destination, ex. "checkpoints/run1.pth"
    meta     : dict optionnel, ex. {"loss": 0.0042, "epoch": 300}
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    if not filepath.endswith(".pth"):
        filepath += ".pth"

    # Reconstitue la config du constructeur depuis les attributs du modèle
    model_config = _extract_config(model)

    metadata = {
        "saved_at":    datetime.now().isoformat(timespec="seconds"),
        "model_config": model_config,
    }
    if meta:
        metadata.update(meta)

    torch.save({"state_dict": model.state_dict(), "meta": metadata}, filepath)

    print(f"[✓] Réseau sauvegardé → {filepath}")
    _print_summary(metadata)