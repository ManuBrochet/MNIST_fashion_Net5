from pathlib import Path

import Run_experiment, utils_files, plot_results

if __name__ == "__main__":

    cfg = dict(
        # ── Batch params ────────────────────────────────────────────────────
        BATCH_SIZE           = 128,     # Can be 32, 64 or 128

        # ML param
        EPOCHS          = 101,
        LR              = 0.05,
        # Can be "Pytorch", "Reduced_network"
        optimizer_choice = "Reduced_network",
        
        # Momentum param
        use_momentum    = False,
        beta_momentum   = 0.9,

        # Reduced network parameters
        sigma_sizes = [40, 24, 4],
        LR_UV = 0.1,

        # Others
        STATS_EVERY     = 1,
    )

    loss_curve, final_metrics = Run_experiment.run_experiment(cfg=cfg, verbose=True, save_model=False)

    path_csv, path_dir = utils_files.save_loss_curve(cfg, loss_curve)

    df_loss    = plot_results.load_and_clean(path_csv)
    param_cols_loss = plot_results.detect_param_cols(df_loss)

    print("→ Generating loss curves…")
    plot_results.plot_loss_curves(df_loss, path_dir / "loss_curve.png", param_cols_loss)
