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

LOSS_CSV        = "benchmark_results/loss_curves.csv"
METRICS_CSV     = "benchmark_results/final_metrics.csv"
DEAD_CSV        = "benchmark_results/dead_neuron_stats.csv"
OUTPUT_DIR      = Path("benchmark_results")
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def color_cycle(n: int):
    return [cm.tab10(i % 10) for i in range(n)]


# ── 1. Loss curves ─────────────────────────────────────────────────────────────

def plot_loss_curves(df: pd.DataFrame, fname, param_cols: list[str]):
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for loss curves.")
        return

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        colors = color_cycle(len(chunk))

        for (cfg, sub), color in zip(chunk, colors):
            avg = sub.groupby("epoch")["loss"].mean().reset_index()
            label = " | ".join(
                f"{k}={v}" for k, v in cfg.items()
                if k in varying and v is not None
            ) or "default"
            ax.plot(avg["epoch"], avg["loss"], label=label, color=color, linewidth=1.8)

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


# ── 2. Dead neuron stats ───────────────────────────────────────────────────────

DEAD_COLS = ["fc1_mean_inactivity", "fc2_mean_inactivity", "fc3_mean_inactivity"]


def plot_dead_neurons(df: pd.DataFrame, param_cols: list[str]):
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for dead neuron stats.")
        return

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, axes = plt.subplots(1, len(DEAD_COLS), figsize=(14, 5), sharey=True)
        colors = color_cycle(len(chunk))

        for ax, dead_col in zip(axes, DEAD_COLS):
            for (cfg, sub), color in zip(chunk, colors):
                avg = sub.groupby("epoch")[dead_col].mean().reset_index()
                label = " | ".join(
                    f"{k}={v}" for k, v in cfg.items()
                    if k in varying and v is not None
                ) or "default"
                ax.plot(avg["epoch"], avg[dead_col], label=label, color=color, linewidth=1.8)

            layer = dead_col.replace("_mean_inactivity", "")
            ax.set_title(layer)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Mean inactivity (avg over images)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8,
                   bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Dead neuron stats – group {chunk_idx + 1}", y=1.02)
        fig.tight_layout()

        fname = OUTPUT_DIR / f"dead_neurons_group{chunk_idx + 1:02d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")


# ── 3. Final metrics ───────────────────────────────────────────────────────────

METRIC_COLS   = ["PSNR", "SSIM", "elapsed_s"]
METRIC_LABELS = {"PSNR": "PSNR (dB)", "SSIM": "SSIM", "elapsed_s": "Time (s)"}


def plot_final_metrics(df: pd.DataFrame, param_cols: list[str]):
    df_filled = fill_params(df, param_cols)
    varying = find_varying_params(df_filled, param_cols)
    all_configs = list(get_config_groups(df, param_cols))

    if not all_configs:
        print("  [warning] No configs found for final metrics.")
        return

    for chunk_idx, chunk in enumerate(chunk_configs(all_configs, MAX_CONFIGS)):
        fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=FIGSIZE_METRICS)
        colors = color_cycle(len(chunk))

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

        fname = OUTPUT_DIR / f"final_metrics_group{chunk_idx + 1:02d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df_loss    = load_and_clean(LOSS_CSV)
    df_metrics = load_and_clean(METRICS_CSV)
    df_dead    = load_and_clean(DEAD_CSV)

    param_cols_loss    = detect_param_cols(df_loss)
    param_cols_metrics = detect_param_cols(df_metrics)
    param_cols_dead    = detect_param_cols(df_dead)

    print(f"Detected param columns (loss):    {param_cols_loss}")
    print(f"Detected param columns (metrics): {param_cols_metrics}")
    print(f"Detected param columns (dead):    {param_cols_dead}")
    print()

    print("→ Generating loss curves…")
    plot_loss_curves(df_loss, param_cols_loss)

    print("→ Generating dead neuron plots…")
    plot_dead_neurons(df_dead, param_cols_dead)

    print("→ Generating final metric plots…")
    plot_final_metrics(df_metrics, param_cols_metrics)

    print(f"\nDone. All figures saved to {OUTPUT_DIR}/")
