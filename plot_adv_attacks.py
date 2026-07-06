"""
plot_adv_attacks.py
=====================

Fonctions de visualisation pour comparer la robustesse de plusieurs modèles
face aux attaques adversariales, à partir du CSV produit par
adversarial_attacks.run_adversarial_experiment.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path):
    return pd.read_csv(csv_path)


def _success_rate(df, group_cols):
    return (
        df.groupby(group_cols)["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )


def plot_success_vs_epsilon(df, attack_type="dense", n_pixels=None, save_path=None):
    """
    Courbe : taux de succès de l'attaque (%) en fonction de epsilon,
    une courbe par modèle.

    attack_type : "dense" ou "sparse". Si "sparse", `n_pixels` doit être
                  précisé pour ne garder qu'un seul budget de pixels.
    """
    subset = df[df["attack_type"] == attack_type].copy()

    if attack_type == "sparse":
        if n_pixels is None:
            raise ValueError("Précisez n_pixels pour un attack_type='sparse'.")
        subset = subset[subset["n_pixels"].astype(str) == str(n_pixels)]

    rates = _success_rate(subset, ["model_name", "epsilon"])

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, group in rates.groupby("model_name"):
        group = group.sort_values("epsilon")
        ax.plot(group["epsilon"], group["success_rate"] * 100,
                marker="o", label=model_name)

    title = f"Taux de succès de l'attaque ({attack_type}"
    title += f", n_pixels={n_pixels})" if attack_type == "sparse" else ")"

    ax.set_xlabel("Epsilon (budget de perturbation par pixel)")
    ax.set_ylabel("Taux de succès de l'attaque (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Figure sauvegardée → {save_path}")

    return fig, ax


def plot_success_vs_n_pixels(df, epsilon, save_path=None):
    """
    Courbe : taux de succès (%) en fonction du nombre de pixels autorisés,
    pour un epsilon fixé, une courbe par modèle (attaques "sparse" uniquement).
    """
    subset = df[(df["attack_type"] == "sparse") & (df["epsilon"] == epsilon)].copy()
    subset["n_pixels"] = pd.to_numeric(subset["n_pixels"], errors="coerce")

    rates = _success_rate(subset, ["model_name", "n_pixels"])

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, group in rates.groupby("model_name"):
        group = group.sort_values("n_pixels")
        ax.plot(group["n_pixels"], group["success_rate"] * 100,
                marker="o", label=model_name)

    ax.set_xlabel("Nombre de pixels modifiés")
    ax.set_ylabel("Taux de succès de l'attaque (%)")
    ax.set_title(f"Taux de succès vs nombre de pixels (epsilon={epsilon})")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Figure sauvegardée → {save_path}")

    return fig, ax


def plot_heatmap(df, model_name, save_path=None):
    """
    Heatmap (epsilon x n_pixels) du taux de succès pour un modèle donné,
    sur les attaques "sparse" — utile pour voir d'un coup d'œil quelles
    combinaisons de budget font céder le modèle.
    """
    subset = df[(df["model_name"] == model_name) & (df["attack_type"] == "sparse")].copy()
    subset["n_pixels"] = pd.to_numeric(subset["n_pixels"], errors="coerce")

    pivot = subset.pivot_table(
        index="epsilon", columns="n_pixels", values="success", aggfunc="mean"
    ) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Nombre de pixels modifiés")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"Taux de succès (%) — {model_name}")

    fig.colorbar(im, ax=ax, label="Taux de succès (%)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Figure sauvegardée → {save_path}")

    return fig, ax


if __name__ == "__main__":
    df = load_results("results/adv_results.csv")

    os.makedirs("results/figures", exist_ok=True)

    plot_success_vs_epsilon(df, attack_type="dense",
                            save_path="results/figures/dense_vs_epsilon.png")

    sparse_df = df[df["attack_type"] == "sparse"]
    if not sparse_df.empty:
        best_epsilon = sparse_df["epsilon"].max()
        plot_success_vs_n_pixels(df, epsilon=best_epsilon,
                                  save_path="results/figures/sparse_vs_n_pixels.png")

    for model_name in df["model_name"].unique():
        if not df[(df.model_name == model_name) & (df.attack_type == "sparse")].empty:
            plot_heatmap(df, model_name,
                         save_path=f"results/figures/heatmap_{model_name}.png")

    plt.show()
