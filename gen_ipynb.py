import os

import nbformat as nbf

# ==========================================
# 1. kaggle/03_main.ipynb
# ==========================================
nb_main = nbf.v4.new_notebook()

nb_main.cells.append(nbf.v4.new_markdown_cell("""# 03_main.ipynb — Single-Stage Dual-GPU Training (CIFAR-10 on GPU 0, CIFAR-100 on GPU 1)
This notebook trains 4 configurations per GPU concurrently:
- **GPU 0**: CIFAR-10 (SGD lr 0.10, SGD lr 0.15, Adam lr 0.001, Adam lr 0.002)
- **GPU 1**: CIFAR-100 (SGD lr 0.10, SGD lr 0.15, Adam lr 0.001, Adam lr 0.002)
- Mode: **dev** (45,000 train / 5,000 validation)
- Validation every epoch measuring both RAW and EMA metrics
- Checkpointing and full resumption support with deterministic run IDs
- Time Guard calibration and safe epoch projection
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""!nvidia-smi
import torch
print(f"PyTorch Version: {torch.__version__}, Available GPUs: {torch.cuda.device_count()}")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import shutil

REPO_SOURCE = "git"  # Options: "git" or "kaggle"
REPO_URL = "https://github.com/PhoenixEvo/flower-lite.git"

if REPO_SOURCE == "git":
    if not os.path.exists("flower-lite"):
        os.system(f"git clone {REPO_URL} flower-lite")
    os.chdir("flower-lite")
else:
    if os.path.exists("/kaggle/input/flowerlite-repo"):
        os.system("cp -r /kaggle/input/flowerlite-repo /kaggle/working/flower-lite")
    os.chdir("/kaggle/working/flower-lite")

os.system("pip install -e .")
import flowerlite
print("FlowerLite package successfully imported!")

DATA_ROOT = "/kaggle/working/data"
import torchvision
torchvision.datasets.CIFAR10(root=DATA_ROOT, download=True)
torchvision.datasets.CIFAR100(root=DATA_ROOT, download=True)
print("CIFAR-10 and CIFAR-100 verified.")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import subprocess
import time
import math

EPOCHS = None  # Human: Set this to an integer (e.g., 60, 80, 100) or leave as None to see safe recommendation
BUDGET_HOURS = 8.0  # Maximum allowed budget for the entire dual-GPU run

def measure_sec_per_epoch(gpu_id="0", batch_size=256, data_root="/kaggle/working/data"):
    print(f"Measuring real execution timing on GPU {gpu_id}...")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
    probe_id = f"timing_probe_gpu{gpu_id}_{int(time.time())}"
    cmd = [
        "python", "train.py",
        "--config", "configs/base.yaml",
        "--dataset", "cifar10",
        "--optimizer", "sgd",
        "--lr", "0.1",
        "--epochs", "1",
        "--run-id", probe_id,
        "--max-steps", "115",
        "--data-root", data_root,
        "--mode", "dev"
    ]
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    elapsed_100_steps = None
    val_time = None
    for line in res.stdout.split("\\n"):
        if "Timing (Steps 10-110)" in line:
            elapsed_100_steps = float(line.split(":")[1].replace("s", "").replace("-", "").strip())
        if "Val Time" in line:
            val_time = float(line.split(":")[1].replace("s", "").replace("-", "").strip())
            
    if elapsed_100_steps is None or val_time is None:
        print("Probe Output STDOUT:\\n", res.stdout)
        print("Probe Output STDERR:\\n", res.stderr)
        raise RuntimeError("Failed to extract 100-step timing and validation timing from probe output.")
        
    steps_per_epoch = math.ceil(45000 / batch_size) # 176 steps per epoch
    train_time_per_epoch = (elapsed_100_steps / 100.0) * steps_per_epoch
    sec_per_epoch = train_time_per_epoch + val_time
    steps_per_sec = 100.0 / elapsed_100_steps
    
    print(f"--> Measured steps/s: {steps_per_sec:.2f}")
    print(f"--> Train time per epoch (176 steps): {train_time_per_epoch:.2f}s")
    print(f"--> Validation time per epoch (5k images): {val_time:.2f}s")
    print(f"--> Total seconds/epoch: {sec_per_epoch:.2f}s")
    return sec_per_epoch

# Measure timing on GPU 0
sec_per_epoch = measure_sec_per_epoch("0", data_root=DATA_ROOT)

# Calculate safe N (largest N fitting BUDGET_HOURS for 4 configs per GPU)
seconds_per_config_epoch = 4.0 * sec_per_epoch
max_safe_N = math.floor((BUDGET_HOURS * 3600.0) / seconds_per_config_epoch)

print(f"\\n================ TIME GUARD SUMMARY ================")
print(f"Measured sec/epoch: {sec_per_epoch:.2f}s")
print(f"Budget: {BUDGET_HOURS} hours")
print(f"Largest safe N (epochs) that fits budget for 4 runs: {max_safe_N}")
print(f"====================================================\\n")

if EPOCHS is None:
    raise ValueError(
        f"EPOCHS is currently None! Based on measured {sec_per_epoch:.2f} s/epoch, "
        f"the largest safe epoch count fitting BUDGET_HOURS ({BUDGET_HOURS}h) is {max_safe_N}. "
        f"Please set EPOCHS = {max_safe_N} (or any integer <= {max_safe_N}) in this cell and re-run."
    )

projected_hours = (seconds_per_config_epoch * EPOCHS) / 3600.0
print(f"Configured EPOCHS: {EPOCHS}")
print(f"Projected wall-clock time: {projected_hours:.2f} hours (Budget: {BUDGET_HOURS} hours)")

if projected_hours > BUDGET_HOURS:
    raise RuntimeError(
        f"Projected runtime ({projected_hours:.2f}h) EXCEEDS budget ({BUDGET_HOURS}h)! "
        f"The maximum safe epoch count is {max_safe_N}. Please reduce EPOCHS <= {max_safe_N}."
    )
print("Time guard passed successfully! Proceeding to execution.")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import json
import subprocess
from pathlib import Path

working_dir = Path("/kaggle/working")
runs_dir = working_dir / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)
results_csv = working_dir / "results.csv"

# Configuration sequence (4 configs per GPU)
configs_gpu0 = [
    {"dataset": "cifar10", "optimizer": "sgd", "lr": 0.10, "run_id": "c10_sgd_0.10"},
    {"dataset": "cifar10", "optimizer": "sgd", "lr": 0.15, "run_id": "c10_sgd_0.15"},
    {"dataset": "cifar10", "optimizer": "adam", "lr": 0.001, "run_id": "c10_adam_0.001"},
    {"dataset": "cifar10", "optimizer": "adam", "lr": 0.002, "run_id": "c10_adam_0.002"},
]

configs_gpu1 = [
    {"dataset": "cifar100", "optimizer": "sgd", "lr": 0.10, "run_id": "c100_sgd_0.10"},
    {"dataset": "cifar100", "optimizer": "sgd", "lr": 0.15, "run_id": "c100_sgd_0.15"},
    {"dataset": "cifar100", "optimizer": "adam", "lr": 0.001, "run_id": "c100_adam_0.001"},
    {"dataset": "cifar100", "optimizer": "adam", "lr": 0.002, "run_id": "c100_adam_0.002"},
]

# Create worker runner script
runner_script = Path("/kaggle/working/worker_runner.py")
runner_code = '''
import sys
import os
import json
import subprocess
from pathlib import Path

gpu_id = sys.argv[1]
configs_json = sys.argv[2]
epochs = int(sys.argv[3])
data_root = sys.argv[4]

configs = json.loads(configs_json)
log_file = Path(f"/kaggle/working/log_gpu{gpu_id}.txt")

with open(log_file, "a") as f:
    f.write(f"=== Starting GPU {gpu_id} Worker ===\\n")

for cfg in configs:
    run_id = cfg["run_id"]
    dataset = cfg["dataset"]
    opt = cfg["optimizer"]
    lr = cfg["lr"]
    
    marker_local = Path("experiments") / run_id / "FINISHED"
    marker_work = Path("/kaggle/working/runs") / run_id / "FINISHED"
    if marker_local.exists() or marker_work.exists():
        with open(log_file, "a") as f:
            f.write(f"[GPU {gpu_id}] Run {run_id} is already FINISHED. Skipping.\\n")
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
        "--results-csv", "/kaggle/working/results.csv",
        "--export-dir", f"/kaggle/working/runs/{run_id}"
    ]

    with open(log_file, "a") as f:
        f.write(f"[GPU {gpu_id}] Launching: {' '.join(cmd)}\\n")
        f.flush()
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            f.write(f"[GPU {gpu_id}] Run {run_id} FAILED with return code {ret.returncode}\\n")
            sys.exit(ret.returncode)
        else:
            f.write(f"[GPU {gpu_id}] Run {run_id} COMPLETED successfully.\\n")

with open(log_file, "a") as f:
    f.write(f"=== GPU {gpu_id} All Configs Done ===\\n")
'''
runner_script.write_text(runner_code)

log_gpu0 = "/kaggle/working/log_gpu0.txt"
log_gpu1 = "/kaggle/working/log_gpu1.txt"
if os.path.exists(log_gpu0): os.remove(log_gpu0)
if os.path.exists(log_gpu1): os.remove(log_gpu1)

# Launch workers concurrently
env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

p0 = subprocess.Popen([sys.executable, str(runner_script), "0", json.dumps(configs_gpu0), str(EPOCHS), DATA_ROOT], env=env0)
p1 = subprocess.Popen([sys.executable, str(runner_script), "1", json.dumps(configs_gpu1), str(EPOCHS), DATA_ROOT], env=env1)

def print_tail(file_path, num_lines=15):
    if not os.path.exists(file_path):
        print(f"[{file_path}] Waiting for output...")
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for l in lines[-num_lines:]:
                print(l.rstrip())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print("Dual GPU workers launched. Polling progress every 30s...")
while p0.poll() is None or p1.poll() is None:
    time.sleep(30)
    current_time = time.strftime("%H:%M:%S")
    print(f"\\n==================== [GPU 0 | {current_time}] Last 15 Lines ====================")
    print_tail(log_gpu0, 15)
    print(f"==================== [GPU 1 | {current_time}] Last 15 Lines ====================")
    print_tail(log_gpu1, 15)
    sys.stdout.flush()

p0.wait()
p1.wait()
print("\\nBoth GPU workers have finished execution.")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import json
import pandas as pd
from pathlib import Path

results_file = Path("/kaggle/working/results.csv")
if not results_file.exists():
    raise RuntimeError("results.csv does not exist. No training runs recorded.")

df = pd.read_csv(results_file)
print("=================== COMPLETE RESULTS LEDGER (results.csv) ===================")
print(df.to_string(index=False))

summary_dict = {
    "status": "completed",
    "epochs": EPOCHS,
    "cifar10": {},
    "cifar100": {}
}

final_choice = {}

for dset in ["cifar10", "cifar100"]:
    sub_df = df[df["dataset"] == dset]
    if sub_df.empty:
        continue
    sub_df = sub_df.copy()
    sub_df["max_val_top1"] = sub_df[["best_val_top1_raw", "best_val_top1_ema"]].max(axis=1)
    best_row = sub_df.sort_values(by="max_val_top1", ascending=False).iloc[0]
    
    weight_type = "ema" if best_row["best_val_top1_ema"] >= best_row["best_val_top1_raw"] else "raw"
    run_id = f"{dset}_{best_row['optimizer']}_{best_row['lr']}"
    
    summary_dict[dset] = {
        "best_run_id": run_id,
        "optimizer": best_row["optimizer"],
        "lr": float(best_row["lr"]),
        "best_val_top1_raw": float(best_row["best_val_top1_raw"]),
        "best_val_top1_ema": float(best_row["best_val_top1_ema"]),
        "best_weight_type": weight_type,
        "best_epoch": int(best_row["best_epoch"]),
        "wall_time": float(best_row["wall_time"])
    }
    
    final_choice[dset] = {
        "run_id": run_id,
        "weight_type": weight_type
    }

print("\\n=================== BEST BY VALIDATION TABLE (Never picked by test) ===================")
for dset, info in summary_dict.items():
    if dset in ["cifar10", "cifar100"]:
        print(f"[{dset.upper()}] Best Run: {info['best_run_id']} | Opt: {info['optimizer']} | LR: {info['lr']}")
        print(f"  Val Top-1 Raw: {info['best_val_top1_raw']:.2f}% | Val Top-1 EMA: {info['best_val_top1_ema']:.2f}% (Selected: {info['best_weight_type'].upper()})")
        print(f"  Best Epoch: {info['best_epoch']} | Wall Time: {info['wall_time']:.1f}s\\n")

with open("/kaggle/working/summary.json", "w") as f:
    json.dump(summary_dict, f, indent=4)
print("Saved /kaggle/working/summary.json")

with open("/kaggle/working/final_choice.json", "w") as f:
    json.dump(final_choice, f, indent=4)
print("Saved /kaggle/working/final_choice.json for 04_test.ipynb evaluation.")
"""))

with open("kaggle/03_main.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb_main, f)

# ==========================================
# 2. kaggle/04_test.ipynb
# ==========================================
nb_test = nbf.v4.new_notebook()

nb_test.cells.append(nbf.v4.new_markdown_cell("""# 04_test.ipynb — Final Test Set Evaluation under FREEZE Gate
This notebook evaluates the models on the official CIFAR test set:
1. **Auto-detects** attached read-only output from `03_main` under `/kaggle/input/*/final_choice.json`.
2. **Copies** run folders into `/kaggle/working/runs/` treating the input as read-only.
3. **DRY_RUN protection**: Default `DRY_RUN = True` lists all runs and stops before touching the test set.
4. **FREEZE Protection**: Enforces `experiments/FREEZE.md`.
5. **Write-once**: `eval_test.py` writes `test_results.json` once and refuses to overwrite.
6. Evaluates winner runs selected by validation, plus other finished runs labeled `"reported, not used for selection"`.
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import glob
import shutil
import subprocess
from pathlib import Path

# Setup working repository
REPO_SOURCE = "git"
REPO_URL = "https://github.com/PhoenixEvo/flower-lite.git"

if not os.path.exists("eval_test.py"):
    if os.path.exists("flower-lite"):
        os.chdir("flower-lite")
    elif os.path.exists("/kaggle/working/flower-lite"):
        os.chdir("/kaggle/working/flower-lite")
    else:
        os.system(f"git clone {REPO_URL} flower-lite")
        os.chdir("flower-lite")

os.system("pip install -e .")
print("Working Directory:", os.getcwd())
assert os.path.exists("eval_test.py"), "eval_test.py must exist!"
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""# 1. Auto-detect training output from attached dataset
search_patterns = [
    "/kaggle/input/*/final_choice.json",
    "/kaggle/input/*/*/final_choice.json",
]
candidates = []
for p in search_patterns:
    candidates.extend(glob.glob(p))

if not candidates and os.path.exists("experiments/final_choice.json"):
    candidates.append("experiments/final_choice.json")

print(f"Detected candidates for final_choice.json: {candidates}")
if len(candidates) == 0:
    raise FileNotFoundError(
        "Could not find any final_choice.json in /kaggle/input/*/ or /kaggle/input/*/*/. "
        "Please attach the output of 03_main as an input to this notebook!"
    )
elif len(candidates) > 1:
    raise ValueError(
        f"Ambiguous training output: found multiple final_choice.json files: {candidates}. "
        "Please ensure only one 03_main output dataset is attached."
    )

choice_file = Path(candidates[0])
print(f"Selected training output config: {choice_file}")
source_dir = choice_file.parent

with open(choice_file, "r") as f:
    final_choice = json.load(f)

print("Loaded final choices (selected strictly by validation):")
print(json.dumps(final_choice, indent=2))

# 2. Treat input folder as read-only. Copy run folders to /kaggle/working/runs/
working_dir = Path("/kaggle/working")
working_runs_dir = working_dir / "runs"
working_runs_dir.mkdir(parents=True, exist_ok=True)

source_runs_dir = source_dir / "runs" if (source_dir / "runs").exists() else source_dir

print(f"\\nCopying run folders from read-only {source_runs_dir} into {working_runs_dir}...")
copied_runs = []
for item in source_runs_dir.iterdir():
    if item.is_dir() and item.name != "runs":
        dest_run_dir = working_runs_dir / item.name
        dest_run_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["best.pt", "last.pt", "history.csv", "FINISHED"]:
            src_f = item / fname
            if src_f.exists():
                shutil.copy2(src_f, dest_run_dir / fname)
        copied_runs.append(item.name)

print(f"Successfully copied {len(copied_runs)} runs into {working_runs_dir}: {copied_runs}")
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""# 3. DRY-RUN Gate & FREEZE Marker
DRY_RUN = True  # Human: Set to False to perform actual test evaluation

print(f"DRY_RUN status: {DRY_RUN}")
if DRY_RUN:
    print("\\n============================== DRY RUN MODE ==============================")
    print("The following chosen models (selected strictly by validation) WOULD be evaluated:")
    for dataset, info in final_choice.items():
        print(f"  * [{dataset.upper()}] Run: '{info['run_id']}' (Weights: '{info.get('weight_type', 'ema').upper()}')")
    print("\\nThe following other finished runs WOULD be evaluated as 'reported, not used for selection':")
    for r_name in copied_runs:
        if not any(info.get("run_id") == r_name for info in final_choice.values()):
            d_name = "cifar10" if "c10" in r_name else "cifar100"
            print(f"  * [{d_name.upper()}] Run: '{r_name}'")
    print("\\n[STOPPING] DRY_RUN is True. Test set has NOT been touched.")
    print("Please set DRY_RUN = False in this cell and re-run to perform final official evaluation.")
    print("==========================================================================")
    raise RuntimeError("DRY_RUN is True. Halted before touching test set. Set DRY_RUN = False to proceed.")

# Write FREEZE marker to unlock evaluation
os.makedirs("experiments", exist_ok=True)
freeze_file = Path("experiments/FREEZE.md")
with open(freeze_file, "w") as f:
    f.write("# FREEZE\\nModel selection finalized. Test set unlocked for single-pass evaluation.\\n")
print(f"Wrote {freeze_file}. Test set evaluation gate is now unlocked.")
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""# 4. Evaluate chosen winner models on official test set
test_outputs = {}

for dataset, info in final_choice.items():
    run_id = info["run_id"]
    weight_type = info.get("weight_type", "ema")
    
    print(f"\\nEvaluating WINNER for {dataset.upper()} (Run ID: {run_id}, Weights: {weight_type.upper()})")
    
    ckpt_dir = f"/kaggle/working/runs/{run_id}"
    if not os.path.exists(ckpt_dir):
        ckpt_dir = f"experiments/{run_id}"
        
    cmd = [
        sys.executable, "eval_test.py",
        "--run-id", run_id,
        "--dataset", dataset,
        "--weight-type", weight_type,
        "--checkpoint-dir", ckpt_dir,
        "--output-dir", ckpt_dir,
        "--data-root", "/kaggle/working/data"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:\\n", res.stderr)
        raise RuntimeError(f"eval_test.py failed for {run_id}")
        
    res_json_path = os.path.join(ckpt_dir, "test_results.json")
    with open(res_json_path, "r") as f:
        test_outputs[dataset] = json.load(f)
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""# 5. Evaluate other finished runs ('reported, not used for selection') & print summary table
import pandas as pd

working_runs = Path("/kaggle/working/runs")
all_evaluations = []

# First, record the chosen winners
for dataset, res in test_outputs.items():
    all_evaluations.append({
        "dataset": dataset,
        "run_id": res["run_id"],
        "weight_type": res["weight_type"],
        "top1": res["top1"],
        "top5": res["top5"],
        "macro_f1": res["macro_f1"],
        "nll": res["nll"],
        "ece": res["ece"],
        "role": "SELECTED WINNER (chosen by validation)"
    })

# Next, record other finished runs
for run_dir in sorted(working_runs.iterdir()):
    if not run_dir.is_dir():
        continue
    run_id = run_dir.name
    if any(e["run_id"] == run_id for e in all_evaluations):
        continue
        
    dataset = "cifar10" if "c10" in run_id or "cifar10" in run_id else "cifar100"
    res_path = run_dir / "test_results.json"
    
    if not res_path.exists():
        cmd = [
            sys.executable, "eval_test.py",
            "--run-id", run_id,
            "--dataset", dataset,
            "--weight-type", "ema",
            "--checkpoint-dir", str(run_dir),
            "--output-dir", str(run_dir),
            "--data-root", "/kaggle/working/data"
        ]
        sub_res = subprocess.run(cmd, capture_output=True, text=True)
        if sub_res.returncode != 0:
            print(f"Skipping {run_id}: evaluation returned code {sub_res.returncode}")
            continue
            
    with open(res_path, "r") as f:
        metrics = json.load(f)
        all_evaluations.append({
            "dataset": dataset,
            "run_id": run_id,
            "weight_type": metrics.get("weight_type", "ema"),
            "top1": metrics.get("top1"),
            "top5": metrics.get("top5"),
            "macro_f1": metrics.get("macro_f1"),
            "nll": metrics.get("nll"),
            "ece": metrics.get("ece"),
            "role": "reported, not used for selection"
        })

df_all = pd.DataFrame(all_evaluations)
print("\\n============================== FINAL OFFICIAL TEST RESULTS ==============================")
print(df_all.to_string(index=False))

with open("/kaggle/working/all_test_evaluations.json", "w") as f:
    json.dump(all_evaluations, f, indent=4)
print("\\nWrote /kaggle/working/all_test_evaluations.json")
"""))

with open("kaggle/04_test.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb_test, f)

print("Successfully regenerated kaggle/04_test.ipynb!")

print("Successfully generated kaggle/03_main.ipynb and kaggle/04_test.ipynb!")
