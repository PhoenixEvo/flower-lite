import argparse
import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary report tables and learning curve plots.")
    parser.add_argument("--results-sweep", type=str, default="results_sweep.csv", help="Path to results_sweep.csv")
    parser.add_argument("--results-final", type=str, default="results_final.csv", help="Path to results_final.csv")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Path to runs/ folder containing history.csv")
    parser.add_argument("--output-dir", type=str, default="report", help="Directory to save report tables and plots")
    return parser.parse_args()


def get_hyperparams(optimizer):
    if optimizer.lower() == "sgd":
        return "0.9 (Nesterov)", "5e-4"
    elif optimizer.lower() == "adam":
        return "(0.9, 0.999)", "5e-2 (decoupled)"
    return "N/A", "N/A"


def generate_tables(sweep_csv, final_csv, output_dir):
    rows = []
    
    # Process sweep results
    if sweep_csv and os.path.exists(sweep_csv):
        df_sw = pd.read_csv(sweep_csv)
        for _, r in df_sw.iterrows():
            m_beta, wd = get_hyperparams(str(r["optimizer"]))
            rows.append({
                "dataset": r["dataset"],
                "stage": "sweep",
                "optimizer": r["optimizer"],
                "lr": r["lr"],
                "epochs": r["epochs"],
                "momentum/betas": m_beta,
                "weight_decay": wd,
                "best_val_top1_raw": r["best_val_top1_raw"],
                "best_val_top1_ema": r["best_val_top1_ema"]
            })

    # Process final results
    if final_csv and os.path.exists(final_csv):
        df_fn = pd.read_csv(final_csv)
        for _, r in df_fn.iterrows():
            m_beta, wd = get_hyperparams(str(r["optimizer"]))
            rows.append({
                "dataset": r["dataset"],
                "stage": "final",
                "optimizer": r["optimizer"],
                "lr": r["lr"],
                "epochs": r["epochs"],
                "momentum/betas": m_beta,
                "weight_decay": wd,
                "best_val_top1_raw": r["best_val_top1_raw"],
                "best_val_top1_ema": r["best_val_top1_ema"]
            })

    if not rows:
        print("No rows found from sweep or final CSVs.")
        return None

    df_out = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "report_summary.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Wrote CSV table to {csv_path}")

    md_path = out_dir / "report_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# FlowerLite-L Experimental Evaluation Summary\n\n")
        f.write(df_out.to_markdown(index=False))
        f.write("\n")
    print(f"Wrote Markdown table to {md_path}")

    return df_out


def generate_learning_curves(runs_dir, output_dir):
    rdir = Path(runs_dir)
    if not rdir.exists():
        print(f"Runs directory {runs_dir} does not exist. Skipping plots.")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all history.csv files
    cifar10_curves = {}
    cifar100_curves = {}

    for run_folder in rdir.iterdir():
        if run_folder.is_dir():
            hist_file = run_folder / "history.csv"
            if hist_file.exists():
                try:
                    df_h = pd.read_csv(hist_file)
                    run_name = run_folder.name
                    if "c100" in run_name or "cifar100" in run_name:
                        cifar100_curves[run_name] = df_h
                    else:
                        cifar10_curves[run_name] = df_h
                except Exception as e:
                    print(f"Error reading {hist_file}: {e}")

    # Plot CIFAR-10
    if cifar10_curves:
        plt.figure(figsize=(10, 6))
        for name, df_h in cifar10_curves.items():
            if "val_top1_ema" in df_h.columns:
                plt.plot(df_h["epoch"], df_h["val_top1_ema"], label=f"{name} (EMA)")
            elif "val_top1" in df_h.columns:
                plt.plot(df_h["epoch"], df_h["val_top1"], label=name)
        plt.title("CIFAR-10 Learning Curves (Validation Top-1 Accuracy)")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Top-1 (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        p10 = out_dir / "cifar10_learning_curves.png"
        plt.savefig(p10, dpi=150)
        plt.close()
        print(f"Saved CIFAR-10 learning curves to {p10}")

    # Plot CIFAR-100
    if cifar100_curves:
        plt.figure(figsize=(10, 6))
        for name, df_h in cifar100_curves.items():
            if "val_top1_ema" in df_h.columns:
                plt.plot(df_h["epoch"], df_h["val_top1_ema"], label=f"{name} (EMA)")
            elif "val_top1" in df_h.columns:
                plt.plot(df_h["epoch"], df_h["val_top1"], label=name)
        plt.title("CIFAR-100 Learning Curves (Validation Top-1 Accuracy)")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Top-1 (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        p100 = out_dir / "cifar100_learning_curves.png"
        plt.savefig(p100, dpi=150)
        plt.close()
        print(f"Saved CIFAR-100 learning curves to {p100}")


def main():
    args = parse_args()
    generate_tables(args.results_sweep, args.results_final, args.output_dir)
    generate_learning_curves(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
