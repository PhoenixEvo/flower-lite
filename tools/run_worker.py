import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    gpu_id = sys.argv[1]
    configs_json = sys.argv[2]
    data_root = sys.argv[3]
    results_csv_path = sys.argv[4]

    configs = json.loads(configs_json)
    log_file = Path(f"/kaggle/working/log_gpu{gpu_id}.txt")
    if not log_file.parent.exists():
        log_file = Path(f"log_gpu{gpu_id}.txt")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== Starting GPU {gpu_id} Worker ===\n")
        f.flush()

    for cfg in configs:
        run_id = cfg["run_id"]
        dataset = cfg["dataset"]
        opt = cfg["optimizer"]
        lr = cfg["lr"]
        epochs = cfg["epochs"]

        marker_local = Path("experiments") / run_id / "FINISHED"
        marker_work = Path("/kaggle/working/runs") / run_id / "FINISHED"
        if marker_local.exists() or marker_work.exists():
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[GPU {gpu_id}] Run {run_id} is already FINISHED. Skipping.\n")
                f.flush()
            continue

        cmd = [
            sys.executable, "train.py",
            "--config", "configs/base.yaml",
            "--dataset", dataset,
            "--optimizer", opt,
            "--lr", str(lr),
            "--epochs", str(epochs),
            "--run-id", run_id,
            "--data-root", data_root,
            "--mode", "dev",
            "--resume",
            "--results-csv", results_csv_path,
            "--export-dir", f"/kaggle/working/runs/{run_id}"
        ]

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[GPU {gpu_id}] Launching: {' '.join(cmd)}\n")
            f.flush()
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            if ret.returncode != 0:
                f.write(f"[GPU {gpu_id}] Run {run_id} FAILED with code {ret.returncode}\n")
                f.flush()
                sys.exit(ret.returncode)
            else:
                f.write(f"[GPU {gpu_id}] Run {run_id} COMPLETED successfully.\n")
                f.flush()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== GPU {gpu_id} All Configs Done ===\n")
        f.flush()


if __name__ == "__main__":
    main()
