"""
Benchmark visualization script for image compression algorithm.

Detects parameter columns automatically and generates:
  - Loss curves (averaged over images) per config group
  - Dead neuron stats curves per config group
  - Final metrics bar charts per config group

At most MAX_CONFIGS_PER_PLOT configurations per figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Configuration ──────────────────────────────────────────────────────────────

# LOSS_CSV        = "benchmark_results/loss_curves.csv"
# METRICS_CSV     = "benchmark_results/final_metrics.csv"
# DEAD_CSV        = "benchmark_results/dead_neuron_stats.csv"
# OUTPUT_DIR      = Path("benchmark_results")
MAX_CONFIGS     = 5
FIGSIZE         = (10, 5)
FIGSIZE_METRICS = (12, 5)

# Columns that are never parameters
NON_PARAM_COLS = {
    "run_id", "image", "seed",
    # metrics
    "final_L1", "elapsed_s",
    "test_loss", "test_acc",
    # series
    "epoch", "loss",
    "fc1_mean_inactivity", "fc2_mean_inactivity", "fc3_mean_inactivity",
}

NAN_SENTINEL = "__NA__"   # safe fill value for NaN in param cols before groupby

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    ("SGD", False),
    ("Adam", False),
    ("Reduced_network", False),
    ("Reduced_network", True),
    ("Reduced_network_iso", False),
]

METHOD_COLORS = {
    ("SGD", False): "#1f77b4",          # bleu
    # ("SGD", True): "#1f77b4",          # bleu
    ("Adam", False): "#ff7f0e",         # orange
    # ("Adam", True): "#ff7f0e",         # orange
    ("Reduced_network", False): "#2ca02c",  # vert
    ("Reduced_network", True): "#d62728",   # rouge
    ("Reduced_network_iso", False): "#A020F0",   # violet

}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, strip whitespace from column names and string values."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()
    return df


def detect_param_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that are parameters (not fixed metadata/metrics)."""
    return [c for c in df.columns if c not in NON_PARAM_COLS]


def fill_params(df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    """Fill NaN in param columns so groupby works correctly."""
    df = df.copy()
    for col in param_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(NAN_SENTINEL)
    return df


def find_varying_params(df: pd.DataFrame, param_cols: list[str]) -> list[str]:
    """Return only params that take more than one distinct value in df."""
    return [c for c in param_cols if df[c].nunique() > 1]


def get_config_groups(df: pd.DataFrame, param_cols: list[str]):
    """Yield (config_dict, sub_df) for each unique parameter combination."""
    df_filled = fill_params(df, param_cols)
    if not param_cols:
        yield {}, df
        return
    for keys, group in df_filled.groupby(param_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cfg = dict(zip(param_cols, keys))
        # Restore NaN sentinel to None for display
        cfg_display = {k: (None if v == NAN_SENTINEL else v) for k, v in cfg.items()}
        # Return rows from original df using the group's index
        yield cfg_display, df.loc[group.index]


def chunk_configs(configs: list, n: int):
    for i in range(0, len(configs), n):
        yield configs[i: i + n]


def sort_configs(all_configs):
    """
    Trie les configurations selon un ordre fixe des méthodes.

    all_configs est une liste de tuples (cfg, sub_df)
    renvoyée par get_config_groups().
    """

    order_dict = {
        method: idx
        for idx, method in enumerate(METHOD_ORDER)
    }

    def sort_key(item):
        cfg, _ = item

        optimizer = cfg.get("optimizer_choice")

        # adaptive_step n'a de sens que pour Reduced_network
        adaptive = (
            bool(cfg.get("adaptive_step", False))
            if optimizer == "Reduced_network"
            else False
        )

        # Ordre principal : la méthode
        method_rank = order_dict.get(
            (optimizer, adaptive),
            len(order_dict)
        )

        # Ordre secondaire : les autres paramètres
        # (pour garder un ordre déterministe)
        other_params = tuple(
            str(cfg[k])
            for k in sorted(cfg)
            if k not in {"optimizer_choice", "adaptive_step"}
        )

        return (method_rank, other_params)

    return sorted(all_configs, key=sort_key)


# def color_cycle(n: int):
#     return [cm.tab10(i % 10) for i in range(n)]


# ── 1. Loss curves ─────────────────────────────────────────────────────────────

SEED_COL = "seed"


def _mean_std_over_seeds(sub: pd.DataFrame, value_col: str, group_cols: list[str]):
    """
    Aggregate a quantity in two steps:
      1. Average all observations belonging to the same seed.
      2. Compute mean and std across seeds.

    This prevents runs with more observations/images from receiving more weight
    than the other seeds.
    """
    if SEED_COL not in sub.columns:
        raise ValueError(f"Column '{SEED_COL}' is required to compute statistics over seeds.")

    # ``group_cols`` can be empty for scalar/final metrics. In that case we
    # only need to group by seed for the first aggregation, then aggregate
    # directly over the resulting seed-level values.
    if group_cols:
        per_seed = (
            sub.groupby(group_cols + [SEED_COL], sort=False)[value_col]
               .mean()
               .reset_index()
        )

        summary = (
            per_seed.groupby(group_cols, sort=False)[value_col]
                     .agg(mean="mean", std="std", n="count")
                     .reset_index()
        )
    else:
        per_seed = (
            sub.groupby(SEED_COL, sort=False)[value_col]
               .mean()
               .reset_index()
        )

        summary = pd.DataFrame({
            "mean": [per_seed[value_col].mean()],
            "std": [per_seed[value_col].std()],
            "n": [per_seed[value_col].count()],
        })
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _config_label(cfg, varying):
    # return " | ".join(
    #     f"{k}={v}" for k, v in cfg.items()
    #     if k in varying and v is not None
    # ) or "default"
    # return cfg["optimizer_choice"] + "_" + str(cfg["taille_couche1"]) + "_" + ("True" if cfg["adaptive_step"] else "False")
    return cfg["optimizer_choice"]

def _get_method_color(cfg):
    optimizer = cfg.get("optimizer_choice")
    adaptive = bool(cfg.get("adaptive_step", False)) if optimizer == "Reduced_network" else False
    return METHOD_COLORS[(optimizer, adaptive)]


def plot_loss_curves(df: pd.DataFrame, fname, param_cols: list[str]):
    """
    Plot loss curves using mean ± std over seeds.

    For each configuration and epoch, the loss is first averaged within each
    seed (e.g. over images), then the mean and std are computed across seeds.
    """
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for loss curves.")
        return

    all_configs = sort_configs(all_configs)

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for cfg, sub in chunk:
            summary = _mean_std_over_seeds(
                sub,
                value_col="loss",
                group_cols=["epoch"],
            )

            label = _config_label(cfg, varying)
            color = _get_method_color(cfg)

            ax.plot(
                summary["epoch"],
                summary["mean"],
                label=label,
                color=color,
                linewidth=1.8,
            )
            ax.fill_between(
                summary["epoch"],
                summary["mean"] - summary["std"],
                summary["mean"] + summary["std"],
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (mean over images)")
        ax.set_title(f"Loss curves – mean ± std over seeds – group {chunk_idx + 1}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname_save = fname + f"_{chunk_idx + 1:02d}.png"
        fig.savefig(fname_save, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname_save}")


# ── 3. Final metrics ───────────────────────────────────────────────────────────

METRIC_COLS   = ["test_loss", "test_acc", "elapsed_s"]
METRIC_LABELS = {
    "test_loss": "Loss for test data",
    "test_acc": "Accuracy for test data",
    "elapsed_s": "Time (s)",
}


def plot_final_metrics(df: pd.DataFrame, fname, param_cols: list[str]):
    """
    Plot final metrics as mean +/- std over seeds.

    Important: ``seed`` is NEVER part of the configuration. Each unique
    combination of experimental parameters gives one bar, while the
    different seeds of that configuration are used to compute the error bar.

    If several rows are present for the same seed/configuration, they are
    first averaged within that seed. Then mean/std are computed across seeds.
    """

    # Safety measure: even if detect_param_cols() was called on a CSV with
    # an unexpected column setup, seed must never define a configuration.
    config_param_cols = [c for c in param_cols if c != SEED_COL]

    df_filled = fill_params(df, config_param_cols)
    varying = find_varying_params(df_filled, config_param_cols)

    # One group = one experimental configuration, independently of seed.
    all_configs = list(get_config_groups(df, config_param_cols))

    if not all_configs:
        print("  [warning] No configs found for final metrics.")
        return

    all_configs = sort_configs(all_configs)

    print(f"  Found {len(all_configs)} configurations (seeds are aggregated).")

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=FIGSIZE_METRICS)

        config_labels = []
        for cfg, _ in chunk:
            config_labels.append(_config_label(cfg, varying))

        x = np.arange(len(chunk))
        bar_width = 0.6

        # One color per configuration, using the fixed method color.
        colors = [_get_method_color(cfg) for cfg, _ in chunk]

        for ax, metric in zip(axes, METRIC_COLS):
            means, stds = [], []

            for cfg, sub in chunk:
                # First average over observations belonging to each seed,
                # then compute mean/std across seeds.
                seed_summary = _mean_std_over_seeds(
                    sub,
                    value_col=metric,
                    group_cols=[],
                )

                means.append(seed_summary["mean"].iloc[0])
                stds.append(seed_summary["std"].iloc[0])

            err_kw = {"elinewidth": 1.2, "capsize": 4}
            bars = ax.bar(
                x,
                means,
                width=bar_width,
                color=colors,
                yerr=stds,
                error_kw=err_kw,
            )

            # Put the mean value above each bar.
            max_std = max(stds) if stds else 0.0
            max_mean = max(abs(m) for m in means) if means else 0.0
            y_offset = max_std * 0.1 if max_std > 0 else max_mean * 0.01
            if y_offset == 0:
                y_offset = 0.01

            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f"{mean:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            short_labels = [
                f"cfg{chunk_idx * MAX_CONFIGS + i + 1}"
                for i in range(len(chunk))
            ]

            ax.set_title(METRIC_LABELS[metric])
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=8)
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(True, axis="y", alpha=0.3)

        legend_text = "\n".join(
            f"cfg{chunk_idx * MAX_CONFIGS + i + 1}: {lbl}"
            for i, lbl in enumerate(config_labels)
        )

        fig.text(
            0.5,
            -0.06,
            legend_text,
            ha="center",
            va="top",
            fontsize=7.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightyellow",
                alpha=0.8,
            ),
        )

        fig.suptitle(
            f"Final metrics (mean +/- std over seeds) – group {chunk_idx + 1}"
        )
        fig.tight_layout()

        fname_save = fname + f"_{chunk_idx + 1:02d}.png"
        fig.savefig(fname_save, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname_save}")


# ── Dead-neuron configuration helpers ─────────────────────────────────────────

DEAD_CONFIG_NON_PARAM_COLS = {
    "run_id",
    "dataset",
    "seed",
    "epoch",
    "layer",
    "neuron",
    "dead_ratio",
}

def _get_dead_neuron_param_cols(df):
    """
    Return configuration columns for the dead-neuron CSV.

    `seed` is deliberately excluded: different seeds of the same experiment
    belong to the same configuration.
    """
    return [
        c for c in df.columns
        if c not in DEAD_CONFIG_NON_PARAM_COLS
    ]


def _dead_config_groups(df):
    """
    Yield (config_index, config_dict, sub_df) for each unique configuration.

    A configuration is defined by all columns except run_id, dataset, seed,
    epoch, layer, neuron and dead_ratio.
    """
    param_cols = _get_dead_neuron_param_cols(df)

    if not param_cols:
        yield 0, {}, df
        return

    filled = fill_params(df, param_cols)

    for config_idx, (keys, group) in enumerate(
        filled.groupby(param_cols, sort=False),
        start=1,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        cfg = dict(zip(param_cols, keys))
        cfg = {
            k: (None if v == NAN_SENTINEL else v)
            for k, v in cfg.items()
        }

        yield config_idx, cfg, df.loc[group.index]


def _format_dead_config_label(cfg):
    """Format a configuration dictionary for plot titles."""
    # if not cfg:
    #     return "default"

    # return " | ".join(
    #     f"{key}={value}"
    #     for key, value in cfg.items()
    #     if value is not None
    # )
    #return cfg["optimizer_choice"] + "_" + str(cfg["taille_couche1"]) + "_" + ("True" if cfg["adaptive_step"] else "False")
    return cfg["optimizer_choice"]


def _safe_filename(text_value):
    """Convert arbitrary text into a filesystem-safe filename component."""
    text_value = str(text_value)
    text_value = re.sub(r"[^\w.-]+", "_", text_value)
    return text_value.strip("_") or "config"


def plot_mean_dead_ratio(df, fname):
    """
    Create one dead-ratio plot per configuration.

    For each configuration:
      1. Compute the mean dead ratio for each (seed, epoch, layer).
      2. Compute mean ± std across seeds.
      3. Plot one curve per layer with a mean ± std band.

    `seed` is NOT part of the configuration.
    """
    groups = list(_dead_config_groups(df))

    if not groups:
        print("  [warning] No configurations found for dead-ratio plots.")
        return

    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)

    for config_idx, cfg, sub in groups:
        per_seed = (
            sub.groupby(["epoch", "layer", SEED_COL], sort=False)["dead_ratio"]
               .mean()
               .reset_index()
        )

        summary = (
            per_seed.groupby(["epoch", "layer"], sort=False)["dead_ratio"]
                     .agg(mean="mean", std="std")
                     .reset_index()
        )
        summary["std"] = summary["std"].fillna(0.0)

        fig, ax = plt.subplots(figsize=(8, 5))

        for layer in summary["layer"].unique():
            layer_sub = summary[summary["layer"] == layer].sort_values("epoch")

            ax.plot(
                layer_sub["epoch"],
                layer_sub["mean"],
                label=layer,
                linewidth=2,
            )

            ax.fill_between(
                layer_sub["epoch"],
                layer_sub["mean"] - layer_sub["std"],
                layer_sub["mean"] + layer_sub["std"],
                alpha=0.18,
                linewidth=0,
            )

        label = _format_dead_config_label(cfg)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean dead ratio")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Mean dead ratio – config {config_idx}\n{label}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out = fname.parent / f"{fname.stem}_config_{config_idx:02d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved {out}")


def plot_dead_neuron_count(df, fname):
    """
    Create one completely-dead-neuron-count plot per configuration.

    For each configuration:
      1. Determine whether each neuron is completely dead.
      2. Count dead neurons for each (seed, epoch, layer).
      3. Compute mean ± std across seeds.
      4. Plot one curve per layer.

    `seed` is NOT part of the configuration.
    """
    groups = list(_dead_config_groups(df))

    if not groups:
        print("  [warning] No configurations found for dead-neuron-count plots.")
        return

    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)

    for config_idx, cfg, sub in groups:
        sub = sub.copy()
        sub["dead"] = sub["dead_ratio"] == 1

        per_seed = (
            sub.groupby(["epoch", "layer", SEED_COL], sort=False)["dead"]
               .sum()
               .reset_index()
        )

        summary = (
            per_seed.groupby(["epoch", "layer"], sort=False)["dead"]
                    .agg(mean="mean", std="std")
                    .reset_index()
        )
        summary["std"] = summary["std"].fillna(0.0)

        fig, ax = plt.subplots(figsize=(8, 5))

        for layer in summary["layer"].unique():
            layer_sub = summary[summary["layer"] == layer].sort_values("epoch")

            ax.plot(
                layer_sub["epoch"],
                layer_sub["mean"],
                label=layer,
                linewidth=2,
            )

            ax.fill_between(
                layer_sub["epoch"],
                layer_sub["mean"] - layer_sub["std"],
                layer_sub["mean"] + layer_sub["std"],
                alpha=0.18,
                linewidth=0,
            )

        label = _format_dead_config_label(cfg)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("# completely dead neurons")
        ax.set_title(
            f"Completely dead neurons – config {config_idx}\n{label}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out = fname.parent / f"{fname.stem}_config_{config_idx:02d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved {out}")


def plot_dead_histogram(df, epoch, fname):
    """
    Create one dead-ratio histogram per configuration for a given epoch.

    The histogram shows the neuron-level dead-ratio distribution over all
    available seeds for that configuration. `seed` is not used to define
    separate configurations.
    """
    groups = list(_dead_config_groups(df))

    if not groups:
        print("  [warning] No configurations found for dead-ratio histograms.")
        return

    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)

    for config_idx, cfg, sub in groups:
        subset = sub[sub["epoch"] == epoch]

        if subset.empty:
            print(
                f"  [warning] No data for epoch {epoch} "
                f"in configuration {config_idx}."
            )
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            subset["dead_ratio"].dropna(),
            bins=20,
        )

        label = _format_dead_config_label(cfg)

        ax.set_xlabel("Dead ratio")
        ax.set_ylabel("Number of neurons")
        ax.set_title(
            f"Dead-ratio distribution – config {config_idx}, epoch {epoch}\n"
            f"{label}",
            fontsize=10,
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = (
            fname.parent
            / f"{fname.stem}_config_{config_idx:02d}_epoch_{epoch}.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved {out}")


if __name__ == "__main__":
    
    path_dir = "benchmark_results/CIFAR10/"
    # path_dir = "benchmark_results/SVHN/"

    path_loss = path_dir + "loss_curve.csv"
    path_val = path_dir + "val_curve.csv"
    path_dead_stats = path_dir + "dead_neurons.csv"
    path_metrics = path_dir + "final_metrics.csv"

    # save loss curve
    df_loss    = load_and_clean(path_loss)
    param_cols_loss = detect_param_cols(df_loss)
    print("→ Generating loss curves…")
    plot_loss_curves(df_loss, path_dir + "loss_curve", param_cols_loss)

    # save validation curve
    df_val    = load_and_clean(path_val)
    param_cols_val = detect_param_cols(df_val)
    print("→ Generating loss curves…")
    plot_loss_curves(df_val, path_dir + "val_curve", param_cols_val)

    # Dead neurons stats
    df_dead_stats = load_and_clean(path_dead_stats)
    param_cols_val = detect_param_cols(df_dead_stats)
    plot_mean_dead_ratio(df_dead_stats, path_dir + "dead_neurons.png")
    plot_dead_neuron_count(df_dead_stats, path_dir + "dead_neurons_count.png")
    # plot_dead_histogram(df_dead_stats, epoch, path_dir + "dead_neurons_hist")

    # Final metrics
    df_loss    = load_and_clean(path_metrics)
    param_cols_loss = detect_param_cols(df_loss)
    print("→ Generating metrics curves…")
    plot_final_metrics(df_loss, path_dir + "final_metrics", param_cols_loss)
