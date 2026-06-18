from pathlib import Path

import Run_experiment, utils_files, plot_results

if __name__ == "__main__":

    cfg = dict(
        # ── Batch params ────────────────────────────────────────────────────
        BATCH_SIZE           = 128,     # Can be 32, 64 or 128

        # ML param
        EPOCHS          = 151,
        LR              = 0.001,
        # Can be "Pytorch", "Reduced_network"
        optimizer_choice = "Reduced_network",
        
        # Momentum param
        use_momentum    = True,
        beta_momentum   = 0.9,

        # Reduced network parameters
        # sigma_sizes = [40, 24, 4],  # 50% de params en moins
        sigma_sizes = [20, 12, 4],  # 75% de params en moins
        LR_UV = 0.1,

        # Others
        STATS_EVERY     = 1,

        # # 50% params en moins
        # taille_couche1 = 69,    # 120 par défaut
        # taille_couche2 = 40,    # 84 par défaut

        # 50% params en moins
        taille_couche1 = 38,    # 120 par défaut
        taille_couche2 = 14,    # 84 par défaut
    )

    loss_curve, final_metrics = Run_experiment.run_experiment(cfg=cfg, verbose=True, save_model=False)


    path_csv, path_dir = utils_files.save_loss_curve(cfg, loss_curve)
    df_loss    = plot_results.load_and_clean(path_csv)
    param_cols_loss = plot_results.detect_param_cols(df_loss)
    print("→ Generating loss curves…")
    plot_results.plot_loss_curves(df_loss, path_dir / "loss_curve.png", param_cols_loss)


    # Final metrics
    path_csv, path_dir = utils_files.save_final_metrics(cfg, final_metrics)
    df_loss    = plot_results.load_and_clean(path_csv)
    param_cols_loss = plot_results.detect_param_cols(df_loss)
    print("→ Generating metrics curves…")
    plot_results.plot_final_metrics(df_loss, path_dir / "final_metrics.png", param_cols_loss)
