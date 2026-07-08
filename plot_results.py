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
    "run_id", "image",
    # metrics
    "final_L1", "PSNR", "SSIM", "elapsed_s",
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
]

METHOD_COLORS = {
    ("SGD", False): "#1f77b4",          # bleu
    ("Adam", False): "#ff7f0e",         # orange
    ("Reduced_network", False): "#2ca02c",  # vert
    ("Reduced_network", True): "#d62728",   # rouge
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

def plot_loss_curves(df: pd.DataFrame, fname, param_cols: list[str]):
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for loss curves.")
        return

    all_configs = sort_configs(
        all_configs
    )

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        

        for cfg, sub in chunk :
            avg = sub.groupby("epoch")["loss"].mean().reset_index()
            label = " | ".join(
                f"{k}={v}" for k, v in cfg.items()
                if k in varying and v is not None
            ) or "default"
            ax.plot(avg["epoch"], avg["loss"], label=label, 
                color=METHOD_COLORS[(cfg["optimizer_choice"],cfg.get("adaptive_step", False))], 
                linewidth=1.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (mean over images)")
        ax.set_title(f"Loss curves – group {chunk_idx + 1}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # fname = OUTPUT_DIR / f"loss_curves_group{chunk_idx + 1:02d}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname.name}")


# ── 3. Final metrics ───────────────────────────────────────────────────────────

METRIC_COLS   = ["test_loss", "test_acc", "elapsed_s"]
METRIC_LABELS = {"test_loss": "Loss for test data", "test_acc": "Accuracy for test data", "elapsed_s": "Time (s)"}


def plot_final_metrics(df: pd.DataFrame, fname, param_cols: list[str]):
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for final metrics.")
        return

    all_configs = sort_configs(
        all_configs
    )

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=FIGSIZE_METRICS)
        # colors = color_cycle(len(chunk))
        colors = []
        for opt in METHOD_COLORS:
            colors.append(METHOD_COLORS[opt])

        config_labels = []
        for cfg, _ in chunk:
            lbl = " | ".join(
                f"{k}={v}" for k, v in cfg.items()
                if k in varying and v is not None
            ) or "default"
            config_labels.append(lbl)

        x = np.arange(len(chunk))
        bar_width = 0.6

        for ax, metric in zip(axes, METRIC_COLS):
            means, stds = [], []
            for cfg, sub in chunk:
                vals = sub[metric].dropna()
                means.append(vals.mean())
                stds.append(vals.std() if len(vals) > 1 else 0.0)

            err_kw = {"linewidth": 1.2, "capsize": 4}
            bars = ax.bar(x, means, width=bar_width, color=colors,
                          yerr=stds, error_kw=err_kw)

            y_offset = max(stds) * 0.1 if any(s > 0 for s in stds) else max(means) * 0.01
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_offset,
                        f"{mean:.3g}", ha="center", va="bottom", fontsize=8)

            short_labels = [f"cfg{chunk_idx * MAX_CONFIGS + i + 1}"
                            for i in range(len(chunk))]
            ax.set_title(METRIC_LABELS[metric])
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=8)
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(True, axis="y", alpha=0.3)

        # Legend key below figure
        legend_text = "\n".join(
            f"cfg{chunk_idx * MAX_CONFIGS + i + 1}: {lbl}"
            for i, lbl in enumerate(config_labels)
        )
        fig.text(0.5, -0.06, legend_text, ha="center", va="top",
                 fontsize=7.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

        fig.suptitle(f"Final metrics (mean ± std over images) – group {chunk_idx + 1}")
        fig.tight_layout()

        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")
