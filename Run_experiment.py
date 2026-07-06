import os
import torch
import torch.nn as nn
import time

import building_network, utils_Riem_opti, load_data, utils_pytorch

# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────


def apply_optimizer(model, cfg, pt_optimizer, first_iteration, adaptative_step, beta2):
    
    if cfg["optimizer_choice"] == "Pytorch":
        pt_optimizer.step()
    elif cfg["optimizer_choice"] == "Reduced_network":
        building_network.reduced_network_optimizer(
            model, cfg["LR"], cfg["LR_UV"], 
            cfg["beta_momentum"], cfg["use_momentum"], first_iteration,
            adaptative_step, beta2
            )
    elif cfg["optimizer_choice"] == "no_constraints":
        building_network.basic_optimizer(
            model, cfg["LR"], cfg["beta_momentum"], 
            cfg["use_momentum"], first_iteration
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(cfg: dict, verbose = False, save_model = False,
                    checkpoint_dir = "checkpoints", run_name = None):
    """
    Train the model for one (image, config) combination.

    Parameters
    ----------
    save_model     : si True, le modèle entraîné est sauvegardé sur disque
                     une fois l'entraînement terminé (voir utils_pytorch.save_mlp).
    checkpoint_dir : dossier de destination des checkpoints.
    run_name       : nom du fichier de checkpoint (sans extension). Si None,
                     un nom est généré automatiquement à partir du dataset et
                     de l'optimizer_choice (utile pour ne pas écraser un
                     modèle par un autre quand on compare plusieurs runs).

    Returns
    -------
    loss_curve   : list[float]  – one entry per epoch
    dead_stats   : list[dict]   – one entry every STATS_EVERY epochs
    final_metrics: dict         – PSNR, SSIM, final_L1
    """
    t0 = time.time()
    torch.manual_seed(42)


    # ── data ──────────────────────────────────────────────────────────────────
    if cfg["dataset"] == "CIFAR10":
        dataset_sizes = [3, 400, 10]
        train_loader, test_loader = load_data.load_CIFAR_10(cfg["BATCH_SIZE"])
    elif cfg["dataset"] == "CIFAR100":
        dataset_sizes = [3, 400, 100]
        train_loader, test_loader = load_data.load_CIFAR_100(cfg["BATCH_SIZE"])
    else :
        dataset_sizes = [1, 256, 10]
        train_loader, test_loader = load_data.load_MNIST_fashion(cfg["BATCH_SIZE"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = building_network.LeNet5(
            rank_fc=cfg['sigma_sizes'],
            optimizer=cfg['optimizer_choice'],
            taille_couche1=cfg["taille_couches"][0],
            taille_couche2=cfg["taille_couches"][1],
            dataset_sizes=dataset_sizes,
        ).to(device)
    

    if cfg["optimizer_choice"] in ("Reduced_network"):
        utils_Riem_opti.initialize_reduced_model(model)

    # ── loss  ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer  ──────────────────────────────────────────────────────
    pt_optimizer = None
    if cfg["optimizer_choice"] == "Pytorch":
        pt_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])


    # ── History ─────────────────────────────────────────────────────────
    loss_curve  = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg["EPOCHS"]):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            if pt_optimizer:
                pt_optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            # Stats de la loss function
            if epoch % cfg["STATS_EVERY"] == 0:
                loss_curve.append([epoch, loss.item()])

            loss.backward()

            # weight update
            apply_optimizer(model, cfg, pt_optimizer, epoch==0, cfg["adaptive_step"], cfg["beta2"])


        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    if verbose:
        print("\nTraining done !")


    test_loss, test_acc = utils_pytorch.evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # Le temps pris par l'algo en secondes
    elapsed = round(time.time() - t0, 1)

    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.2f}%")
    print(f"Elapsed time (s) : {elapsed}")


    final_metrics = {
        "test_loss":    test_loss,
        "test_acc":     test_acc,
        "elapsed_s" :   elapsed,
    }

    if save_model:
        # Nom de fichier auto-généré si non fourni, pour pouvoir sauvegarder
        # plusieurs modèles (différents datasets / optimizers) sans les
        # écraser les uns les autres.
        if run_name is None:
            if cfg['optimizer_choice'] in ("Pytorch", "no_constraints"):
                run_name = f"{cfg.get('dataset', 'MNIST_fashion')}_{cfg['optimizer_choice']}_layerSize_{cfg["taille_couches"][0]}_{cfg["taille_couches"][1]}"
            else :
                run_name = f"{cfg.get('dataset', 'MNIST_fashion')}_{cfg['optimizer_choice']}_momentum_{cfg["use_momentum"]}_adaptStep_{cfg["adaptive_step"]}_SigSize_{cfg["sigma_sizes"][0]}_{cfg["sigma_sizes"][1]}_{cfg["sigma_sizes"][2]}"

        filepath = os.path.join(checkpoint_dir, f"{run_name}.pth")

        # Sauvegarde finale avec toutes les métadonnées utiles
        utils_pytorch.save_mlp(
            model,
            filepath=filepath,
            dataset_sizes=dataset_sizes,
            cfg=cfg,
            meta={
                "loss":       final_metrics["test_loss"],
                "test_acc":   final_metrics["test_acc"],
                "epoch":      cfg["EPOCHS"],
                "notes":      "Réseau final",
            },
        )

        final_metrics["checkpoint_path"] = filepath

    return loss_curve, final_metrics