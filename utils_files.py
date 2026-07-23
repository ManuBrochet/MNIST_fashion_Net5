from pathlib import Path
import csv
import pandas  as pd

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
        "run_id":               run_id,
        "dataset":              cfg["dataset"],
        "seed":                 cfg["seed"],
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
        "taille_couche2":       cfg["taille_couches"][1],
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


def create_output_dir(cfg, benchmark):
    run_id = make_run_id(cfg)
    meta   = build_meta_row(run_id, cfg)


    if benchmark:
        output_dir = f"benchmark_results/{cfg['dataset']}"
    else :
        output_dir = f"main_results/{cfg['dataset']}"


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, meta


def save_loss_curve(cfg, loss_curve, benchmark = False, wich_curve = "loss_curve.csv"):

    output_dir, meta = create_output_dir(cfg, benchmark)

    loss_csv = output_dir / wich_curve


    loss_fields   = ["run_id", "dataset", "seed", "optimizer_choice", "LR", "LR_UV", "use_momentum", "beta_momentum", 
                    "LR_UV", "sigma_size_1", "sigma_size_2", "sigma_size_3", "taille_couche1", "taille_couche2", "adaptive_step", "beta2", "EPOCHS", "epoch", "loss"]

    loss_fh,   loss_writer   = open_csv(loss_csv,   loss_fields)

    # ── write loss_curves.csv ─────────────────────────────
    for epoch_loss in loss_curve:
        loss_writer.writerow({**meta, "epoch": epoch_loss[0], "loss": epoch_loss[1]})
    loss_fh.flush()

    return loss_csv, output_dir


def save_final_metrics(cfg, final_metrics, benchmark = False):

    output_dir, meta = create_output_dir(cfg, benchmark)

    loss_csv   = output_dir / "final_metrics.csv"

    loss_fields   = ["run_id", "dataset", "seed", "optimizer_choice", "LR", "LR_UV", "use_momentum", "beta_momentum", 
                    "LR_UV", "sigma_size_1", "sigma_size_2", "sigma_size_3", "taille_couche1", "taille_couche2", "adaptive_step", "beta2", "EPOCHS", "test_loss", "test_acc", "elapsed_s"]

    metrics_fh,   metrics_writer   = open_csv(loss_csv,   loss_fields)

    # ── write final_metrics.csv ─────────────────────────────
    metrics_writer.writerow({**meta, "test_loss": final_metrics["test_loss"], "test_acc": final_metrics["test_acc"], "elapsed_s": final_metrics["elapsed_s"]})
    metrics_fh.flush()

    return loss_csv, output_dir

def save_dead_neuron_stats(cfg, dead_stats, filename, benchmark = True):
    """
    dead_stats is a list like

    [
        {
            "epoch": 0,
            "fc1": tensor(...),
            "fc2": tensor(...)
        },
        ...
    ]
    """

    output_dir, meta = create_output_dir(cfg, benchmark)

    dead_stats_csv = output_dir / filename

    dead_stats_fields   = ["run_id", "dataset", "seed", "optimizer_choice", "LR", "LR_UV", "use_momentum", "beta_momentum", 
                    "LR_UV", "sigma_size_1", "sigma_size_2", "sigma_size_3", "taille_couche1", "taille_couche2", "adaptive_step", "beta2", "EPOCHS", "epoch", "layer", "neuron", "dead_ratio"]

    dead_stats_fh, dead_stats_writer = open_csv(dead_stats_csv, dead_stats_fields)

    rows = []

    for epoch_stats in dead_stats:

        epoch = epoch_stats["epoch"]

        for layer in ("fc1", "fc2"):

            values = epoch_stats[layer].cpu().numpy()

            for neuron, ratio in enumerate(values):

                rows.append({
                    "epoch": epoch,
                    "layer": layer,
                    "neuron": neuron,
                    "dead_ratio": float(ratio)
                })

                dead_stats_writer.writerow({**meta, "epoch": epoch, "layer": layer, "neuron": neuron, "dead_ratio": float(ratio)})
    dead_stats_fh.flush()
        

    df = pd.DataFrame(rows)
    # df.to_csv(filename, index=False)

    
    return df
