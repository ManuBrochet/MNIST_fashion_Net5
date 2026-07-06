from pathlib import Path

from torch.nn.modules import adaptive

import Run_experiment, utils_files, plot_results

if __name__ == "__main__":

    cfg = dict(
        # ── Batch params ────────────────────────────────────────────────────
        BATCH_SIZE           = 128,     # Can be 32, 64 or 128

        # ML param
        EPOCHS          = 2,
        LR              = 0.05,
        # Can be "Pytorch", "Reduced_network", "no_constraints"
        optimizer_choice = "Reduced_network",
        
        # Momentum param
        use_momentum    = True,
        beta_momentum   = 0.9,

        # Reduced network parameters
        sigma_sizes = [40, 24, 4],  # 50% de params en moins
        # sigma_sizes = [20, 12, 4],  # 75% de params en moins
        LR_UV = 0.1,

        # Others
        STATS_EVERY     = 10,

        # 0% params en moins
        taille_couches = [120, 84],
        # 50% params en moins
        # taille_couches = [69, 40],
        # 75% params en moins
        # taille_couches = [38, 14],

        # adaptive_step params
        adaptive_step = True,
        beta2 = 0.9,

        dataset = "CIFAR10",  # "CIFAR10", "CIFAR100", "MNIST_fashion"
    )

    print("L'optimizer utilisé est : ", cfg["optimizer_choice"])

    loss_curve, final_metrics = Run_experiment.run_experiment(cfg=cfg, verbose=True, save_model=True)


    path_csv, path_dir = utils_files.save_loss_curve(cfg, loss_curve, benchmark=False)
    df_loss    = plot_results.load_and_clean(path_csv)
    param_cols_loss = plot_results.detect_param_cols(df_loss)
    print("→ Generating loss curves…")
    plot_results.plot_loss_curves(df_loss, path_dir / "loss_curve.png", param_cols_loss)


    # Final metrics
    path_csv, path_dir = utils_files.save_final_metrics(cfg, final_metrics, benchmark=False)
    df_loss    = plot_results.load_and_clean(path_csv)
    param_cols_loss = plot_results.detect_param_cols(df_loss)
    print("→ Generating metrics curves…")
    plot_results.plot_final_metrics(df_loss, path_dir / "final_metrics.png", param_cols_loss)
