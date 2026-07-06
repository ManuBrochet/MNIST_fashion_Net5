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

def _mode_label(cfg: dict) -> str:
    opt = cfg.get("optimizer_choice")
    if opt == "Reduced_network":
        return f"ReducedLinear (rank_fc={cfg.get('rank_fc', '')})"
    if opt == "no_constraints":
        return "Linear standard (no_constraints)"
    return "Linear standard (Pytorch/Adam)"


def _print_summary(meta: dict) -> None:
    cfg = meta.get("model_config", {})
    lines = [
        f"  Date         : {meta.get('saved_at', '?')}",
        f"  Dataset      : {cfg.get('dataset_name', '?')}",
        f"  Mode         : {_mode_label(cfg)}",
        f"  taille_1/2   : {cfg.get('taille_couche1', '?')} / {cfg.get('taille_couche2', '?')}",
    ]
    if "loss"     in meta: lines.append(f"  Loss         : {meta['loss']:.6f}")
    if "test_acc" in meta: lines.append(f"  Test acc     : {meta['test_acc']:.2f}%")
    if "epoch"    in meta: lines.append(f"  Epoch        : {meta['epoch']}")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_mlp(model: building_network.LeNet5,
             filepath: str,
             dataset_sizes: list,
             cfg: dict,
             meta: dict = None) -> None:
    """
    Sauvegarde un LeNet5 (poids + configuration) dans un fichier .pth, afin
    de pouvoir le recharger plus tard sans refaire l'entraînement
    (voir load_lenet5).

    Ce qui est sauvegardé :
      - state_dict   : tous les paramètres entraînés (U, V, log_Sigma, bias,
                       ou weight/bias selon le mode)
      - model_config : les arguments nécessaires pour reconstruire le modèle
                       à l'identique (rank_fc, optimizer_choice,
                       taille_couche1/2, dataset_sizes, dataset_name)
      - meta         : métriques et informations libres sur l'entraînement

    Paramètres
    ----------
    model         : instance de LeNet5 après entraînement
    filepath      : chemin de destination, ex. "checkpoints/run1.pth"
    dataset_sizes : [in_channels, taille_flatten, n_classes] passé à LeNet5
                    à la construction (cf. Run_experiment.run_experiment)
    cfg           : dict de config de l'expérience (sigma_sizes,
                    optimizer_choice, taille_couches, dataset, ...)
    meta          : dict optionnel, ex. {"loss": 0.0042, "epoch": 300}
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    if not filepath.endswith(".pth"):
        filepath += ".pth"

    model_config = {
        "rank_fc":          cfg["sigma_sizes"],
        "optimizer_choice": cfg["optimizer_choice"],
        "taille_couche1":   cfg["taille_couches"][0],
        "taille_couche2":   cfg["taille_couches"][1],
        "dataset_sizes":    dataset_sizes,
        "dataset_name":     cfg.get("dataset", "MNIST_fashion"),
    }

    metadata = {
        "saved_at":    datetime.now().isoformat(timespec="seconds"),
        "model_config": model_config,
    }
    if meta:
        metadata.update(meta)

    torch.save({"state_dict": model.state_dict(), "meta": metadata}, filepath)

    print(f"[✓] Réseau sauvegardé → {filepath}")
    _print_summary(metadata)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_lenet5(filepath: str, device=None):
    """
    Recharge un LeNet5 sauvegardé avec save_mlp : reconstruit l'architecture
    exacte (Reduced_network / Pytorch / no_constraints) à partir de la config
    stockée, puis charge les poids entraînés.

    Paramètres
    ----------
    filepath : chemin vers le fichier .pth
    device   : torch.device cible (par défaut : cuda si disponible, sinon cpu)

    Retourne
    --------
    model : instance de LeNet5, poids chargés, en mode eval(), sur `device`
    meta  : dict des métadonnées sauvegardées (loss, epoch, model_config, ...)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)
    meta = checkpoint["meta"]
    cfg = meta["model_config"]

    model = LeNet5(
        rank_fc=cfg["rank_fc"],
        optimizer=cfg["optimizer_choice"],
        taille_couche1=cfg["taille_couche1"],
        taille_couche2=cfg["taille_couche2"],
        dataset_sizes=cfg["dataset_sizes"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, meta