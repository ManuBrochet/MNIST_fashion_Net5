"""
Run a single seed with full diagnostics enabled, and dump a CSV you can
plot to see exactly when/why it diverges (or confirm it stays healthy).

Usage:
    python debug_seed.py --seed 7          # a seed you know fails
    python debug_seed.py --seed 0          # a seed you know is healthy

Compare sigma_max, the log_Sigma grad norm, and the U/V ortho_residual
columns between a failing and a healthy seed around the epoch where the
loss curve spikes. If sigma_max and logsigma_grad_norm both start
climbing several epochs before the spike, that confirms the log_Sigma
runaway. If ortho_residual jumps first (or in step with sigma_max),
that confirms it's the corruption cascading into U/V via matrix_exp on
huge-magnitude gradients.
"""

import argparse
import csv

import Run_experiment

BASE_CFG = dict(
    EPOCHS=501,
    STATS_EVERY=1,
    BATCH_SIZE=1024,
    dataset="CIFAR10",
    early_stopping=True,
    patience=15,
    min_delta=1e-4,
    optimizer_choice="Reduced_network",
    use_momentum=True,
    beta_momentum=0.9,
    LR_UV=0.1,
    LR_UV_iso=0.1,
    sigma_sizes=[36, 24, 4],
    taille_couches=[120, 84],
    adaptive_step=True,
    beta2=0.9,
    LR=0.1,
    # ---- diagnostics / fixes toggles ----
    debug_diagnostics=True,     # <- turn on per-batch logging
    grad_clip_norm=None,        # <- set e.g. 5.0 to test the fix
    logsigma_clamp=None,        # <- set e.g. (-3.0, 3.0) to test the fix
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default=None,
                         help="CSV output path (default: diagnostics_seed{seed}.csv)")
    parser.add_argument("--grad_clip_norm", type=float, default=None,
                         help="Set to test the gradient-clipping fix, e.g. 5.0")
    parser.add_argument("--logsigma_clamp", type=float, nargs=2, default=None,
                         metavar=("MIN", "MAX"),
                         help="Set to test the log_Sigma clamp fix, e.g. -3 3")
    args = parser.parse_args()

    cfg = dict(BASE_CFG)
    cfg["seed"] = args.seed
    if args.grad_clip_norm is not None:
        cfg["grad_clip_norm"] = args.grad_clip_norm
    if args.logsigma_clamp is not None:
        cfg["logsigma_clamp"] = tuple(args.logsigma_clamp)

    loss_curve, val_curve, final_metrics, dead_stats = Run_experiment.run_experiment(
        cfg=cfg, verbose=True, save_model=False
    )

    print("\nfinal_metrics (excluding diagnostics):",
          {k: v for k, v in final_metrics.items() if k != "diagnostics"})

    diagnostics = final_metrics.get("diagnostics", [])
    if not diagnostics:
        print("No diagnostics recorded (debug_diagnostics was off?).")
        return

    out_path = args.out or f"diagnostics_seed{args.seed}.csv"
    fieldnames = sorted({key for row in diagnostics for key in row.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"Wrote {len(diagnostics)} rows to {out_path}")


if __name__ == "__main__":
    main()
