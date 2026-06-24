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


def run_experiment(cfg: dict, verbose = False, save_model = False):
    """
    Train the model for one (image, config) combination.

    Returns
    -------
    loss_curve   : list[float]  – one entry per epoch
    dead_stats   : list[dict]   – one entry every STATS_EVERY epochs
    final_metrics: dict         – PSNR, SSIM, final_L1
    """
    t0 = time.time()
    torch.manual_seed(42)


    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = load_data.load_CIFAR_100(cfg["BATCH_SIZE"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = building_network.LeNet5(
            rank_fc=cfg['sigma_sizes'],
            optimizer=cfg['optimizer_choice'],
            taille_couche1=cfg["taille_couches"][0],
            taille_couche2=cfg["taille_couches"][1]
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


        if verbose and epoch % 10 == 0:
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
        # Sauvegarde finale avec toutes les métadonnées utiles
        utils_pytorch.save_mlp(
            model,
            filepath="checkpoints/mlp_final.pth",
            meta={
                "loss":       final_metrics["test_loss"],
                "epoch":      cfg["EPOCHS"],
                "notes":      "Réseau final",
            },
        )

    return loss_curve, final_metrics