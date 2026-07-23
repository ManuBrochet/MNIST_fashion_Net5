import argparse
import itertools
from pathlib import Path
import torch

import Run_experiment, utils_files, plot_results

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────

# Learning rates are set per-optimizer as requested.
LR_MAP = {
    "Adam":      0.0005,
    "Reduced_network": 0.1,
    "Reduced_network_iso": 0.01,
    "SGD": 0.001,
}

# Each dict in this list is one experimental configuration.
# Fields that are not present fall back to the defaults defined in DEFAULT_CFG.
PARAM_GRID = {
    # Can be "Adam", "Reduced_network", "SGD"
    # "optimizer_choice": ["Adam", "SGD", "Reduced_network", "Reduced_network_iso"],
    "optimizer_choice": ["Adam"],
    "tanh_loss":        [False],
    "use_momentum":     [True],
    # Ignored when use_momentum == False
    "beta_momentum":    [0.9],
    # "beta_momentum":    [0.1, 0.3, 0.5, 0.7, 0.9],
    "proportion_dead_neurons" : [0.5],
    "LR_UV":            [0.1],
    "LR_UV_iso":            [0.1],
    "sigma_sizes":       [[40, 24, 4]],
    # à checker
    "taille_couches":      [[120, 84], [69, 40]],
    "adaptive_step":    [False, True],
    "beta2":            [0.9],
    # "seed":             list(range(2))
    "seed":             [42]
}

DEFAULT_CFG = dict(
    # HIDDEN_SIZE             = 44,     # Approx 5000 parameters
    EPOCHS                  = 3,
    STATS_EVERY             = 5,
    BATCH_SIZE              = 128,
    dataset                 = "CIFAR10",
    # Early stopping params
    early_stopping          = True,
    patience                = 25,
    min_delta               = 1e-4
)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeNet5 reduced network")

    parser.add_argument("--output_dir",  default="./benchmark_results",
                        help="Where to write CSV files")
    parser.add_argument("--extensions",  default="jpg,jpeg,png,bmp",
                        help="Comma-separated list of image extensions to include")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {f".{e.strip().lower()}" for e in args.extensions.split(",")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Build full config list from grid
    MOMENTUM_OPTIMIZERS = {"Reduced_network", "Reduced_network_iso", "SGD"}
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    configs = []
    for combo in combos:
        cfg = dict(DEFAULT_CFG)           # start from defaults
        cfg.update(dict(zip(keys, combo)))
        cfg["LR"] = LR_MAP[cfg["optimizer_choice"]]

        # Momentum only applies to Ortho/Ortho_proj/Ortho_QD/Rank:
        # skip configs that request momentum for unsupported optimizers,
        # and deduplicate by treating all no-momentum configs as equivalent
        # (beta_momentum is irrelevant when use_momentum=False).
        if cfg["optimizer_choice"] not in MOMENTUM_OPTIMIZERS and cfg["use_momentum"]!=PARAM_GRID["use_momentum"][0]:
            continue
        if not cfg["use_momentum"]:
            # Avoid running the same no-momentum config once per beta value
            if cfg["beta_momentum"] != PARAM_GRID["beta_momentum"][0]:
                continue
            cfg["beta_momentum"] = ""   # not applicable — keep CSV clean
        if cfg["optimizer_choice"] not in ("Reduced_network", "Reduced_network_iso"):
            if cfg["LR_UV"] != PARAM_GRID["LR_UV"][0] or cfg["sigma_sizes"] != PARAM_GRID["sigma_sizes"][0]:
                continue
        if cfg["optimizer_choice"] != "Reduced_network":
            if cfg["adaptive_step"] != PARAM_GRID["adaptive_step"][0]:
                continue
        if not cfg["optimizer_choice"] in ("SGD", "Adam"):
            if cfg["taille_couches"] != PARAM_GRID["taille_couches"][0]:
                continue

        configs.append(cfg)

    nb_total_configs = len(configs)

    print(f"Configurations : {nb_total_configs}")

    for (nb_config, cfg) in enumerate(configs):

        print("Config ", nb_config + 1, " / ", nb_total_configs)

        loss_curve, val_curve, final_metrics, dead_stats = Run_experiment.run_experiment(
            cfg=cfg, verbose=False, save_model=True, checkpoint_dir="checkpoints_benchmark"
        )

        # save loss curve
        path_csv, path_dir = utils_files.save_loss_curve(cfg, loss_curve, benchmark=True, wich_curve="loss_curve.csv")
        # df_loss    = plot_results.load_and_clean(path_csv)
        # param_cols_loss = plot_results.detect_param_cols(df_loss)
        # print("→ Generating loss curves…")
        # plot_results.plot_loss_curves(df_loss, str(path_dir) + "loss_curve", param_cols_loss)

        # save validation curve
        path_csv, path_dir = utils_files.save_loss_curve(cfg, val_curve, benchmark=True, wich_curve="val_curve.csv")
        # df_loss    = plot_results.load_and_clean(path_csv)
        # param_cols_loss = plot_results.detect_param_cols(df_loss)
        # print("→ Generating loss curves…")
        # plot_results.plot_loss_curves(df_loss, str(path_dir) + "val_curve", param_cols_loss)

        # save dead neurons stats
        df = utils_files.save_dead_neuron_stats(cfg, dead_stats, "dead_neurons.csv")
        # plot_results.plot_mean_dead_ratio(df, path_dir / "dead_neurons.png")
        # plot_results.plot_dead_neuron_count(df, path_dir / "dead_neurons_count.png")
        # plot_results.plot_dead_histogram(df, epoch, fname)


        # Final metrics
        path_csv, path_dir = utils_files.save_final_metrics(cfg, final_metrics, benchmark=True)
        # df_loss    = plot_results.load_and_clean(path_csv)
        # param_cols_loss = plot_results.detect_param_cols(df_loss)
        # print("→ Generating metrics curves…")
        # plot_results.plot_final_metrics(df_loss, str(path_dir) + "final_metrics", param_cols_loss)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
