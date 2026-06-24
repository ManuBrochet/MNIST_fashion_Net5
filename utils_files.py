from pathlib import Path
import csv
import argparse


def make_run_id(cfg: dict) -> str:
    opt = cfg["optimizer_choice"]
    lr  = cfg["LR"]
    use_momentum = "momentum" if cfg["use_momentum"] else "no_momentum"
    extra = ""
    extra_momentum = ""
    if opt == "Reduced_network":
        extra = f"__lr_uv{cfg['LR_UV']}"
    if cfg["use_momentum"] :
        extra_momentum = str(cfg["beta_momentum"])
    return f"{opt}__lr{lr}__{use_momentum}_{extra_momentum}{extra}"


def build_meta_row(run_id: str, cfg: dict) -> dict:
    """Flat dict of all config fields — goes into every CSV row as metadata."""
    return {
        "run_id":           run_id,
        "optimizer_choice":     cfg["optimizer_choice"],
        "LR":                   cfg["LR"],
        "EPOCHS":               cfg["EPOCHS"],
        "use_momentum":         cfg["use_momentum"],
        "beta_momentum":        cfg["beta_momentum"],
        "LR_UV":                cfg["LR_UV"],
        "sigma_size_1":         cfg["sigma_sizes"][0],
        "sigma_size_2":         cfg["sigma_sizes"][1],
        "sigma_size_3":         cfg["sigma_sizes"][2],
        "taille_couche1":       cfg["taille_couches"][0],
        "taille_couche1":       cfg["taille_couches"][1],
        "adaptive_step":        cfg["adaptive_step"], 
        "beta2":                cfg["beta2"], 
    }

def open_csv(path: Path, fieldnames: list):
    """Open (or append to) a CSV; return (file_handle, DictWriter)."""
    write_header = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    return fh, writer


def create_output_dir(cfg):
    run_id = make_run_id(cfg)
    meta   = build_meta_row(run_id, cfg)


    # if cfg['use_momentum']:
    #     output_dir = f"results_main/{cfg['optimizer_choice']}/momentum_True/beta_momentum_{cfg['beta_momentum']}"
    # else :
    #     output_dir = f"results_main/{cfg['optimizer_choice']}/momentum_False"

    output_dir = "benchmark_results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, meta


def save_loss_curve(cfg, loss_curve):

    output_dir, meta = create_output_dir(cfg)

    loss_csv   = output_dir / "loss_curves.csv"

    loss_fields   = ["run_id", "optimizer_choice", "LR", "LR_UV", "use_momentum", "beta_momentum", 
                    "LR_UV", "sigma_size_1", "sigma_size_2", "sigma_size_3", "taille_couche1", "taille_couche2", "adaptive_step", "beta2", "EPOCHS", "epoch", "loss"]

    loss_fh,   loss_writer   = open_csv(loss_csv,   loss_fields)

    # ── write loss_curves.csv ─────────────────────────────
    for epoch_loss in loss_curve:
        loss_writer.writerow({**meta, "epoch": epoch_loss[0], "loss": epoch_loss[1]})
    loss_fh.flush()

    return loss_csv, output_dir


def save_final_metrics(cfg, final_metrics):

    output_dir, meta = create_output_dir(cfg)

    loss_csv   = output_dir / "final_metrics.csv"

    loss_fields   = ["run_id", "optimizer_choice", "LR", "LR_UV", "use_momentum", "beta_momentum", 
                    "LR_UV", "sigma_size_1", "sigma_size_2", "sigma_size_3", "taille_couche1", "taille_couche2", "adaptive_step", "beta2", "EPOCHS", "test_loss", "test_acc", "elapsed_s"]

    metrics_fh,   metrics_writer   = open_csv(loss_csv,   loss_fields)

    # ── write final_metrics.csv ─────────────────────────────
    metrics_writer.writerow({**meta, "test_loss": final_metrics["test_loss"], "test_acc": final_metrics["test_acc"], "elapsed_s": final_metrics["elapsed_s"]})
    metrics_fh.flush()

    return loss_csv, output_dir
