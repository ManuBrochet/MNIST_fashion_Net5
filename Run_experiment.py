import os
import torch
import torch.nn as nn
import time
import copy

import utils_math
import building_network, utils_Riem_opti, load_data, utils_pytorch, utils_math
import fast_data


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────


def apply_optimizer(model, cfg, pt_optimizer, first_iteration, adaptative_step, beta2):

    if cfg["optimizer_choice"] == "Adam":
        pt_optimizer.step()
    elif cfg["optimizer_choice"] == "Reduced_network":
        building_network.reduced_network_optimizer(
            model, cfg["LR"], cfg["LR_UV"],
            cfg["beta_momentum"], cfg["use_momentum"], first_iteration,
            adaptative_step, beta2, logsigma_clamp=cfg.get("logsigma_clamp")
        )
    elif cfg["optimizer_choice"] == "SGD":
        building_network.basic_optimizer(
            model, cfg["LR"], cfg["beta_momentum"],
            cfg["use_momentum"], first_iteration
        )
    elif cfg["optimizer_choice"] == "Reduced_network_iso":
        building_network.reduced_network_optimizer_iso(
            model, cfg["LR"], cfg["LR_UV_iso"],
            cfg["beta_momentum"], cfg["use_momentum"], first_iteration,
            logsigma_clamp=cfg.get("logsigma_clamp")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────



def _gpu_loaders(dataset_name, batch_size, validation_split, seed):
    """Build GPU-resident loaders for `dataset_name`.

    Uses torchvision only to read the raw arrays; no transform pipeline runs,
    because normalisation happens on the device inside fast_data.GPULoader.
    """
    from torchvision import datasets as _tv

    if dataset_name in ("CIFAR10", "CIFAR100"):
        cls = _tv.CIFAR10 if dataset_name == "CIFAR10" else _tv.CIFAR100
        train = cls(root="./data", train=True, download=True)
        test = cls(root="./data", train=False, download=True)
    elif dataset_name == "SVHN":
        train = _tv.SVHN(root="./data", split="train", download=True)
        test = _tv.SVHN(root="./data", split="test", download=True)
    else:
        train = _tv.FashionMNIST(root="./data", train=True, download=True)
        test = _tv.FashionMNIST(root="./data", train=False, download=True)

    return fast_data.build_loaders(train, test, dataset_name, batch_size,
                                   validation_split=validation_split, seed=seed)


def run_experiment(cfg: dict, verbose=False, save_model=False,
                   checkpoint_dir="checkpoints", run_name=None):
    """
    Train the model for one (image, config) combination.

    The original training set is split into:
      - training set: used for parameter updates;
      - validation set: used for monitoring / early stopping;
      - test set: kept completely untouched until the final evaluation.

    Parameters
    ----------
    save_model : bool
        If True, the final model is saved to disk.
    checkpoint_dir : str
        Directory where checkpoints are stored.
    run_name : str or None
        Checkpoint filename without extension.
    """

    t0 = time.time()
    seed = cfg["seed"]

    # Reproducibility of model initialization and validation split.
    torch.manual_seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    # validation_split is the fraction of the original training set reserved
    # for validation. Default: 10%.
    validation_split = cfg.get("validation_split", 0.1)

    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            "validation_split must be strictly between 0 and 1 "
            "when using a validation set."
        )

    if cfg["dataset"] == "CIFAR10":
        dataset_sizes = [3, 400, 10]
        train_loader, validation_loader, test_loader = _gpu_loaders(
            "CIFAR10", cfg["BATCH_SIZE"], validation_split, seed
        )
    elif cfg["dataset"] == "CIFAR100":
        dataset_sizes = [3, 400, 100]
        train_loader, validation_loader, test_loader = _gpu_loaders(
            "CIFAR100", cfg["BATCH_SIZE"], validation_split, seed
        )
    elif cfg["dataset"] == "SVHN":
        dataset_sizes = [3, 400, 10]
        train_loader, validation_loader, test_loader = _gpu_loaders(
            "SVHN", cfg["BATCH_SIZE"], validation_split, seed
        )
    # else:
    #     dataset_sizes = [1, 256, 10]
    #     train_loader, validation_loader, test_loader = load_data.load_MNIST_fashion(
    #         cfg["BATCH_SIZE"],
    #         validation_split=validation_split,
    #         seed=seed,
    #     )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = building_network.LeNet5(
        rank_fc=cfg["sigma_sizes"],
        optimizer=cfg["optimizer_choice"],
        taille_couche1=cfg["taille_couches"][0],
        taille_couche2=cfg["taille_couches"][1],
        dataset_sizes=dataset_sizes,
    ).to(device)

    if cfg["optimizer_choice"] in ("Reduced_network", "Reduced_network_iso"):
        utils_Riem_opti.initialize_reduced_model(model)

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ──────────────────────────────────────────────────────────────
    pt_optimizer = None
    if cfg["optimizer_choice"] == "Adam":
        pt_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])

    # ── History ───────────────────────────────────────────────────────────────
    loss_curve = []
    val_curve = []
    dead_stats = []
    diagnostics_log = []  # populated only if cfg["debug_diagnostics"] is True

    # ── Early stopping variables ──────────────────────────────────────────────
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None
    epochs_without_improvement = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(cfg["EPOCHS"]):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            if pt_optimizer:
                pt_optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Snapshot Sigma / gradient / orthogonality stats BEFORE the
            # step, while the gradients that are about to be applied are
            # still populated. Cheap enough to run every batch for a
            # single debug seed; leave cfg["debug_diagnostics"] unset
            # (or False) for full benchmark sweeps.
            if cfg.get("debug_diagnostics", False):
                diagnostics_log.append({
                    "epoch": epoch,
                    "loss": loss.item() if torch.isfinite(loss) else float("nan"),
                    **utils_math.log_training_diagnostics(model),
                })

            if not torch.isfinite(loss):
                print(
                    f"[seed {seed}] epoch {epoch}: loss is non-finite "
                    f"({loss.item()}) — aborting this run."
                )
                elapsed = round(time.time() - t0, 1)
                final_metrics = {
                    "test_loss": float("nan"),
                    "test_acc": float("nan"),
                    "elapsed_s": elapsed,
                    "best_epoch": best_epoch,
                    "diverged": True,
                    "diverged_epoch": epoch,
                }
                if cfg.get("debug_diagnostics", False):
                    final_metrics["diagnostics"] = diagnostics_log
                return loss_curve, val_curve, final_metrics, dead_stats

            grad_clip_norm = cfg.get("grad_clip_norm")
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            apply_optimizer(
                model,
                cfg,
                pt_optimizer,
                epoch == 0,
                cfg["adaptive_step"],
                cfg["beta2"]
            )

        loss_curve.append([epoch, loss.item()])

        # ── Dead neuron statistics: training subset only ──────────────────────
        stats_every = cfg.get("STATS_EVERY", 1)
        if epoch % stats_every == 0 or epoch == cfg["EPOCHS"] - 1:
            stats = utils_math.compute_dead_neuron_stats(
                model,
                validation_loader
            )

            dead_stats.append({
                "epoch": epoch,
                "fc1": stats["fc1"].cpu(),
                "fc2": stats["fc2"].cpu(),
            })

            # A layer where ~every unit is dead is an absorbing state for
            # ReLU: no gradient can flow back through it, so the run can
            # never recover via gradient descent. Rather than waiting out
            # `patience` epochs on a network that's already decided,
            # detect it directly and stop.
            dead_layer_threshold = cfg.get("dead_layer_threshold")
            if dead_layer_threshold is not None:
                fc1_dead = stats["fc1"].mean().item()
                fc2_dead = stats["fc2"].mean().item()
                if fc1_dead >= dead_layer_threshold or fc2_dead >= dead_layer_threshold:
                    print(
                        f"[seed {seed}] epoch {epoch}: fc1={fc1_dead:.1%} / "
                        f"fc2={fc2_dead:.1%} of units dead — aborting run."
                    )
                    elapsed = round(time.time() - t0, 1)
                    final_metrics = {
                        "test_loss": float("nan"),
                        "test_acc": float("nan"),
                        "elapsed_s": elapsed,
                        "best_epoch": best_epoch,
                        "diverged": False,
                        "dead": True,
                        "dead_epoch": epoch,
                    }
                    if cfg.get("debug_diagnostics", False):
                        final_metrics["diagnostics"] = diagnostics_log
                    return loss_curve, val_curve, final_metrics, dead_stats

        # ── Validation evaluation ────────────────────────────────────────────
        # The validation set comes exclusively from the original training set.
        val_loss, val_acc = utils_pytorch.evaluate(
            model,
            validation_loader,
            criterion,
            device
        )

        val_curve.append([epoch, val_loss, val_acc])

        # ── Early stopping on validation loss ─────────────────────────────────
        if cfg.get("early_stopping", False):

            min_delta = cfg.get("min_delta", 0.0)
            patience = cfg["patience"]

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

                if verbose:
                    print(
                        f"No validation improvement for "
                        f"{epochs_without_improvement}/{patience} epochs."
                    )

                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        model.train()

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train loss: {loss.item():.6f} | "
                f"Val loss: {val_loss:.6f} | "
                f"Val accuracy: {val_acc:.2f}%"
            )

    if verbose:
        print("\nTraining done !")

    # Restore the model corresponding to the best validation loss.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Final test evaluation ─────────────────────────────────────────────────
    # IMPORTANT: test_loader is used only here, after training and model
    # selection are completely finished.
    test_loss, test_acc = utils_pytorch.evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    elapsed = round(time.time() - t0, 1)

    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.2f}%")
    print(f"Elapsed time (s) : {elapsed}")

    final_metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_s": elapsed,
        "best_epoch": best_epoch,
        "diverged": False,
    }
    if cfg.get("debug_diagnostics", False):
        final_metrics["diagnostics"] = diagnostics_log

    if save_model:
        if run_name is None:
            if cfg["optimizer_choice"] in ("Adam", "SGD"):
                run_name = (
                    f"{cfg.get('dataset', 'MNIST_fashion')}_"
                    f"{cfg['optimizer_choice']}_"
                    f"layerSize_{cfg['taille_couches'][0]}_"
                    f"{cfg['taille_couches'][1]}"
                )
            else:
                run_name = (
                    f"{cfg.get('dataset', 'MNIST_fashion')}_"
                    f"{cfg['optimizer_choice']}_"
                    f"momentum_{cfg['use_momentum']}_"
                    f"adaptStep_{cfg['adaptive_step']}_"
                    f"SigSize_{cfg['sigma_sizes'][0]}_"
                    f"{cfg['sigma_sizes'][1]}_"
                    f"{cfg['sigma_sizes'][2]}"
                )

        filepath = os.path.join(checkpoint_dir, f"{run_name}.pth")

        utils_pytorch.save_mlp(
            model,
            filepath=filepath,
            dataset_sizes=dataset_sizes,
            cfg=cfg,
            meta={
                "loss": final_metrics["test_loss"],
                "test_acc": final_metrics["test_acc"],
                "epoch": best_epoch if best_epoch is not None else cfg["EPOCHS"] - 1,
                "notes": "Réseau final sélectionné sur le jeu de validation",
            },
        )

        final_metrics["checkpoint_path"] = filepath

    return loss_curve, val_curve, final_metrics, dead_stats
