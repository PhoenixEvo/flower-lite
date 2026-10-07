from pathlib import Path

import pandas as pd

from tools.make_report_tables import generate_learning_curves, generate_tables


# NOT A RESULT - SYNTHETIC TEST FIXTURE ONLY
def test_make_report_tables(tmp_path):
    # Setup synthetic results
    sweep_csv = tmp_path / "results_sweep.csv"
    final_csv = tmp_path / "results_final.csv"
    output_dir = tmp_path / "report"
    runs_dir = tmp_path / "runs"

    df_sw = pd.DataFrame([{
        "dataset": "cifar10",
        "optimizer": "sgd",
        "lr": 0.1,
        "epochs": 30,
        "best_val_top1_raw": 85.0,
        "best_val_top1_ema": 85.5,
        "final_val_top1_raw": 84.8,
        "final_val_top1_ema": 85.4,
        "best_epoch": 28,
        "wall_time": 120.0,
        "peak_vram_mb": 1500.0
    }])
    df_sw.to_csv(sweep_csv, index=False)

    df_fn = pd.DataFrame([{
        "dataset": "cifar10",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 100,
        "best_val_top1_raw": 90.0,
        "best_val_top1_ema": 91.2,
        "final_val_top1_raw": 89.8,
        "final_val_top1_ema": 91.1,
        "best_epoch": 95,
        "wall_time": 450.0,
        "peak_vram_mb": 1500.0
    }])
    df_fn.to_csv(final_csv, index=False)

    # Synthetic run history
    r_dir = runs_dir / "c10_sgd_0.10"
    r_dir.mkdir(parents=True)
    df_hist = pd.DataFrame({
        "epoch": [0, 1, 2],
        "train_loss": [2.3, 1.8, 1.2],
        "val_loss": [2.2, 1.7, 1.1],
        "val_top1_raw": [50.0, 70.0, 85.0],
        "val_top1_ema": [51.0, 71.0, 85.5],
        "lr": [0.01, 0.05, 0.1],
        "time": [10.0, 10.0, 10.0]
    })
    df_hist.to_csv(r_dir / "history.csv", index=False)

    # Run functions
    df_out = generate_tables(str(sweep_csv), str(final_csv), str(output_dir))
    assert df_out is not None
    assert len(df_out) == 2

    csv_out = output_dir / "report_summary.csv"
    md_out = output_dir / "report_summary.md"
    assert csv_out.exists()
    assert md_out.exists()

    content_md = md_out.read_text(encoding="utf-8")
    assert "cifar10" in content_md
    assert "0.9 (Nesterov)" in content_md
    assert "5e-2 (decoupled)" in content_md

    generate_learning_curves(str(runs_dir), str(output_dir))
    png_out = output_dir / "cifar10_learning_curves.png"
    assert png_out.exists()
