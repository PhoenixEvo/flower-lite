import os

import nbformat as nbf

# ==========================================
# 1. kaggle/03_main.ipynb
# ==========================================
nb_main = nbf.v4.new_notebook()

nb_main.cells.append(nbf.v4.new_markdown_cell("""# 03_main.ipynb — Parameterized Training Pipeline (Sweep & Final Stages)
This notebook supports two execution stages via the `STAGE` variable:
- **`STAGE = "sweep"`**: Runs a 4-config hyperparameter sweep concurrently on GPU 0 (CIFAR-10) and GPU 1 (CIFAR-100).
  Writes `/kaggle/working/results_sweep.csv` and selects best configs into `/kaggle/working/final_choice_sweep.json`
  (including tie detection if difference < 0.5%).
- **`STAGE = "final"`**: Runs the long final training for the selected configurations (1 config per dataset).
  Auto-detects `final_choice_sweep.json` from attached inputs or uses the editable `FINAL_CONFIG` dict.
  Writes `/kaggle/working/results_final.csv` and `/kaggle/working/final_choice.json`.
- **Time Guard**: Calibrates seconds/epoch, projects total hours, and enforces `BUDGET_HOURS`.
- **Deterministic Resumption**: Exactly resumes from `last.pt` and skips runs with `FINISHED` markers.
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""!nvidia-smi
import torch
print(f"PyTorch Version: {torch.__version__}, Available GPUs: {torch.cuda.device_count()}")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import sys
import shutil

REPO_SOURCE = "git"  # Options: "git" or "kaggle"
REPO_URL = "https://github.com/PhoenixEvo/flower-lite.git"

if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")

if REPO_SOURCE == "git":
    if not os.path.exists("flower-lite"):
        os.system(f"git clone {REPO_URL} flower-lite")
        os.chdir("flower-lite")
    else:
        os.chdir("flower-lite")
        os.system("git pull")
else:
    if os.path.exists("/kaggle/input/flowerlite-repo"):
        os.system("cp -r /kaggle/input/flowerlite-repo /kaggle/working/flower-lite")
    os.chdir("/kaggle/working/flower-lite")

for p in [os.path.abspath("src"), "/kaggle/working/flower-lite/src"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

os.system("pip install -e .")
import flowerlite
print("FlowerLite package successfully imported from:", flowerlite.__file__)

DATA_ROOT = "/kaggle/working/data"
os.makedirs(DATA_ROOT, exist_ok=True)

import glob
import shutil
import torchvision

# Auto-detect offline dataset under /kaggle/input if attached
c10_batch1 = glob.glob("/kaggle/input/**/data_batch_1", recursive=True)
if c10_batch1:
    src_c10_dir = os.path.dirname(c10_batch1[0])
    target_c10_dir = os.path.join(DATA_ROOT, "cifar-10-batches-py")
    if not os.path.exists(target_c10_dir):
        print(f"Detected attached CIFAR-10 at {src_c10_dir}. Linking to {target_c10_dir}...")
        try:
            os.symlink(src_c10_dir, target_c10_dir)
        except Exception:
            shutil.copytree(src_c10_dir, target_c10_dir)
else:
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root=DATA_ROOT, download=True)

c100_train = glob.glob("/kaggle/input/**/train", recursive=True)
c100_found = False
for t_path in c100_train:
    dir_path = os.path.dirname(t_path)
    if os.path.exists(os.path.join(dir_path, "meta")) and os.path.exists(os.path.join(dir_path, "test")):
        target_c100_dir = os.path.join(DATA_ROOT, "cifar-100-python")
        if not os.path.exists(target_c100_dir):
            print(f"Detected attached CIFAR-100 at {dir_path}. Linking to {target_c100_dir}...")
            try:
                os.symlink(dir_path, target_c100_dir)
            except Exception:
                shutil.copytree(dir_path, target_c100_dir)
        c100_found = True
        break

if not c100_found:
    print("Downloading CIFAR-100...")
    torchvision.datasets.CIFAR100(root=DATA_ROOT, download=True)

c10_ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, download=False)
c100_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, download=False)
print(f"Dataset verification PASSED: CIFAR-10 ({len(c10_ds)} images), CIFAR-100 ({len(c100_ds)} images).")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""# ==========================================
# STAGE & RUNTIME HYPERPARAMETERS
# ==========================================
STAGE = "sweep"  # Set to "sweep" or "final"

EPOCHS = None    # Set to integer (e.g. 30 for sweep, 120 for final) or None to get Time Guard recommendation
BUDGET_HOURS = 8.0  # Maximum wall-clock budget for this stage in hours

# Human-editable configuration for STAGE == "final".
# If values are None, it will auto-load from final_choice_sweep.json in attached input!
FINAL_CONFIG = {
    "cifar10": None,   # Example: {"optimizer": "sgd", "lr": 0.15, "weight_type": "ema", "epochs": 120}
    "cifar100": None   # Example: {"optimizer": "adam", "lr": 0.001, "weight_type": "ema", "epochs": 120}
}

print(f"Active Stage: {STAGE.upper()}")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import subprocess
import time
import math

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

sec_per_epoch = measure_sec_per_epoch("0", data_root=DATA_ROOT)

# Number of configs running sequentially per GPU: 4 for sweep, 1 for final
num_runs_per_gpu = 4 if STAGE == "sweep" else 1
seconds_per_config_epoch = num_runs_per_gpu * sec_per_epoch
budget_hours_val = float(BUDGET_HOURS)
max_safe_N = math.floor((budget_hours_val * 3600.0) / seconds_per_config_epoch)

print(f"\\n================ TIME GUARD SUMMARY ({STAGE.upper()}) ================")
print(f"Measured sec/epoch: {sec_per_epoch:.2f}s")
print(f"Configs per GPU: {num_runs_per_gpu}")
print(f"Budget: {budget_hours_val:.2f} hours")
print(f"Largest safe N (epochs) that fits budget: {max_safe_N}")
print(f"===============================================================\\n")

if EPOCHS is None:
    raise ValueError(
        f"EPOCHS is currently None! Based on measured {sec_per_epoch:.2f} s/epoch and {num_runs_per_gpu} runs per GPU, "
        f"the largest safe epoch count fitting BUDGET_HOURS ({budget_hours_val:.2f}h) is {max_safe_N}. "
        f"Please set EPOCHS = {max_safe_N} (or any integer <= {max_safe_N}, e.g. 30 or 40 for a fast sweep) in Cell 3 and re-run."
    )

epochs_val = int(EPOCHS)
projected_hours = (seconds_per_config_epoch * epochs_val) / 3600.0
print(f"Configured EPOCHS: {epochs_val}")
print(f"Projected wall-clock time: {projected_hours:.2f} hours (Budget: {budget_hours_val:.2f} hours)")

if projected_hours > budget_hours_val:
    raise RuntimeError(
        f"Projected runtime ({projected_hours:.2f}h) EXCEEDS budget ({budget_hours_val:.2f}h)! "
        f"The maximum safe epoch count is {max_safe_N}. Please reduce EPOCHS <= {max_safe_N}."
    )
print("Time guard passed successfully! Proceeding to execution.")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import os
import sys
import glob
import json
from pathlib import Path

working_dir = Path("/kaggle/working")
runs_dir = working_dir / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)

if STAGE == "sweep":
    results_csv_name = "results_sweep.csv"
    prefix = "sweep_"
    configs_gpu0 = [
        {"dataset": "cifar10", "optimizer": "sgd", "lr": 0.10, "epochs": EPOCHS, "run_id": f"{prefix}c10_sgd_0.10"},
        {"dataset": "cifar10", "optimizer": "sgd", "lr": 0.15, "epochs": EPOCHS, "run_id": f"{prefix}c10_sgd_0.15"},
        {"dataset": "cifar10", "optimizer": "adam", "lr": 0.001, "epochs": EPOCHS, "run_id": f"{prefix}c10_adam_0.001"},
        {"dataset": "cifar10", "optimizer": "adam", "lr": 0.002, "epochs": EPOCHS, "run_id": f"{prefix}c10_adam_0.002"},
    ]
    configs_gpu1 = [
        {"dataset": "cifar100", "optimizer": "sgd", "lr": 0.10, "epochs": EPOCHS, "run_id": f"{prefix}c100_sgd_0.10"},
        {"dataset": "cifar100", "optimizer": "sgd", "lr": 0.15, "epochs": EPOCHS, "run_id": f"{prefix}c100_sgd_0.15"},
        {"dataset": "cifar100", "optimizer": "adam", "lr": 0.001, "epochs": EPOCHS, "run_id": f"{prefix}c100_adam_0.001"},
        {"dataset": "cifar100", "optimizer": "adam", "lr": 0.002, "epochs": EPOCHS, "run_id": f"{prefix}c100_adam_0.002"},
    ]
elif STAGE == "final":
    results_csv_name = "results_final.csv"
    prefix = "final_"
    
    # Auto-load final_choice_sweep.json if FINAL_CONFIG has None
    c10_cfg = FINAL_CONFIG.get("cifar10")
    c100_cfg = FINAL_CONFIG.get("cifar100")
    
    if c10_cfg is None or c100_cfg is None:
        sweep_choice_candidates = glob.glob("/kaggle/input/*/final_choice_sweep.json") + \
                                  glob.glob("/kaggle/input/*/*/final_choice_sweep.json") + \
                                  glob.glob("experiments/final_choice_sweep.json") + \
                                  glob.glob("/kaggle/working/final_choice_sweep.json")
        if not sweep_choice_candidates:
            raise FileNotFoundError(
                "STAGE is 'final' but final_choice_sweep.json was not found in /kaggle/input/*/, "
                "and FINAL_CONFIG entries are None. Please attach the output of the sweep run as input, "
                "or specify FINAL_CONFIG explicitly in Cell 3."
            )
        choice_sweep_file = sweep_choice_candidates[0]
        print(f"Loading sweep choices from: {choice_sweep_file}")
        with open(choice_sweep_file, "r") as f:
            sweep_data = json.load(f)
            
        if c10_cfg is None:
            c10_entry = sweep_data["cifar10"]["winner"] if "winner" in sweep_data["cifar10"] else sweep_data["cifar10"]
            c10_cfg = {
                "optimizer": c10_entry["optimizer"],
                "lr": c10_entry["lr"],
                "epochs": EPOCHS if EPOCHS is not None else c10_entry.get("epochs", 120),
                "weight_type": c10_entry.get("weight_type", "ema")
            }
        if c100_cfg is None:
            c100_entry = sweep_data["cifar100"]["winner"] if "winner" in sweep_data["cifar100"] else sweep_data["cifar100"]
            c100_cfg = {
                "optimizer": c100_entry["optimizer"],
                "lr": c100_entry["lr"],
                "epochs": EPOCHS if EPOCHS is not None else c100_entry.get("epochs", 120),
                "weight_type": c100_entry.get("weight_type", "ema")
            }

    configs_gpu0 = [{
        "dataset": "cifar10",
        "optimizer": c10_cfg["optimizer"],
        "lr": c10_cfg["lr"],
        "epochs": c10_cfg.get("epochs", EPOCHS),
        "run_id": f"{prefix}c10_{c10_cfg['optimizer']}_{c10_cfg['lr']}"
    }]
    configs_gpu1 = [{
        "dataset": "cifar100",
        "optimizer": c100_cfg["optimizer"],
        "lr": c100_cfg["lr"],
        "epochs": c100_cfg.get("epochs", EPOCHS),
        "run_id": f"{prefix}c100_{c100_cfg['optimizer']}_{c100_cfg['lr']}"
    }]
else:
    raise ValueError(f"Unknown STAGE '{STAGE}'. Must be 'sweep' or 'final'.")

print(f"GPU 0 (CIFAR-10) Configs: {configs_gpu0}")
print(f"GPU 1 (CIFAR-100) Configs: {configs_gpu1}")
"""))

nb_main.cells.append(nbf.v4.new_code_cell("""import subprocess
import time

results_csv = working_dir / results_csv_name

# Write worker runner script
runner_script = Path("/kaggle/working/worker_runner.py")
runner_code = f'''
import sys
import os
import json
import subprocess
from pathlib import Path

gpu_id = sys.argv[1]
configs_json = sys.argv[2]
data_root = sys.argv[3]
results_csv_path = sys.argv[4]

configs = json.loads(configs_json)
log_file = Path(f"/kaggle/working/log_gpu{{gpu_id}}.txt")

with open(log_file, "a") as f:
    f.write(f"=== Starting GPU {{gpu_id}} Worker ===\\n")

for cfg in configs:
    run_id = cfg["run_id"]
    dataset = cfg["dataset"]
    opt = cfg["optimizer"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    
    marker_local = Path("experiments") / run_id / "FINISHED"
    marker_work = Path("/kaggle/working/runs") / run_id / "FINISHED"
    if marker_local.exists() or marker_work.exists():
        with open(log_file, "a") as f:
            f.write(f"[GPU {{gpu_id}}] Run {{run_id}} is already FINISHED. Skipping.\\n")
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
        "--export-dir", f"/kaggle/working/runs/{{run_id}}"
    ]

    with open(log_file, "a") as f:
        f.write(f"[GPU {{gpu_id}}] Launching: {{' '.join(cmd)}}\\n")
        f.flush()
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            f.write(f"[GPU {{gpu_id}}] Run {{run_id}} FAILED with code {{ret.returncode}}\\n")
            sys.exit(ret.returncode)
        else:
            f.write(f"[GPU {{gpu_id}}] Run {{run_id}} COMPLETED successfully.\\n")

with open(log_file, "a") as f:
    f.write(f"=== GPU {{gpu_id}} All Configs Done ===\\n")
'''
runner_script.write_text(runner_code)

log_gpu0 = "/kaggle/working/log_gpu0.txt"
log_gpu1 = "/kaggle/working/log_gpu1.txt"
if os.path.exists(log_gpu0): os.remove(log_gpu0)
if os.path.exists(log_gpu1): os.remove(log_gpu1)

# Launch workers concurrently
env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

p0 = subprocess.Popen([sys.executable, str(runner_script), "0", json.dumps(configs_gpu0), DATA_ROOT, str(results_csv)], env=env0)
p1 = subprocess.Popen([sys.executable, str(runner_script), "1", json.dumps(configs_gpu1), DATA_ROOT, str(results_csv)], env=env1)

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

if not results_csv.exists():
    raise RuntimeError(f"{results_csv} does not exist. No training runs recorded.")

df = pd.read_csv(results_csv)
print(f"=================== RESULTS LEDGER ({results_csv_name}) ===================")
print(df.to_string(index=False))

if STAGE == "sweep":
    final_choice_sweep = {}
    for dset in ["cifar10", "cifar100"]:
        sub_df = df[df["dataset"] == dset].copy()
        if sub_df.empty:
            continue
        sub_df["max_val_top1"] = sub_df[["best_val_top1_raw", "best_val_top1_ema"]].max(axis=1)
        sorted_df = sub_df.sort_values(by="max_val_top1", ascending=False).reset_index(drop=True)
        
        top1_row = sorted_df.iloc[0]
        w_type_1 = "ema" if top1_row["best_val_top1_ema"] >= top1_row["best_val_top1_raw"] else "raw"
        score_1 = top1_row["best_val_top1_ema"] if w_type_1 == "ema" else top1_row["best_val_top1_raw"]
        reason_1 = f"Selected {w_type_1.upper()} because {w_type_1.upper()} Top-1 ({score_1:.2f}%) >= alternative."

        entry = {
            "winner": {
                "run_id": f"sweep_{dset}_{top1_row['optimizer']}_{top1_row['lr']}",
                "optimizer": top1_row["optimizer"],
                "lr": float(top1_row["lr"]),
                "weight_type": w_type_1,
                "best_val_top1": float(score_1),
                "reason": reason_1
            },
            "tie": False
        }

        # Check for tie with 2nd place (< 0.5% difference)
        if len(sorted_df) > 1:
            top2_row = sorted_df.iloc[1]
            w_type_2 = "ema" if top2_row["best_val_top1_ema"] >= top2_row["best_val_top1_raw"] else "raw"
            score_2 = top2_row["best_val_top1_ema"] if w_type_2 == "ema" else top2_row["best_val_top1_raw"]
            diff = abs(score_1 - score_2)
            if diff < 0.5:
                entry["tie"] = True
                entry["tied_runner_up"] = {
                    "run_id": f"sweep_{dset}_{top2_row['optimizer']}_{top2_row['lr']}",
                    "optimizer": top2_row["optimizer"],
                    "lr": float(top2_row["lr"]),
                    "weight_type": w_type_2,
                    "best_val_top1": float(score_2),
                    "score_difference": float(diff)
                }
                print(f"[{dset.upper()}] Note: TIE detected between 1st ({score_1:.2f}%) and 2nd ({score_2:.2f}%), difference = {diff:.3f}% (< 0.5%).")

        final_choice_sweep[dset] = entry

    out_file = Path("/kaggle/working/final_choice_sweep.json")
    with open(out_file, "w") as f:
        json.dump(final_choice_sweep, f, indent=4)
    print(f"\\nWrote sweep choice to {out_file}:")
    print(json.dumps(final_choice_sweep, indent=2))

elif STAGE == "final":
    final_choice = {}
    for dset in ["cifar10", "cifar100"]:
        sub_df = df[df["dataset"] == dset].copy()
        if sub_df.empty:
            continue
        sub_df["max_val_top1"] = sub_df[["best_val_top1_raw", "best_val_top1_ema"]].max(axis=1)
        best_row = sub_df.sort_values(by="max_val_top1", ascending=False).iloc[0]
        w_type = "ema" if best_row["best_val_top1_ema"] >= best_row["best_val_top1_raw"] else "raw"
        run_id = f"final_{dset}_{best_row['optimizer']}_{best_row['lr']}"
        final_choice[dset] = {
            "run_id": run_id,
            "optimizer": best_row["optimizer"],
            "lr": float(best_row["lr"]),
            "weight_type": w_type,
            "best_val_top1_raw": float(best_row["best_val_top1_raw"]),
            "best_val_top1_ema": float(best_row["best_val_top1_ema"])
        }

    out_file = Path("/kaggle/working/final_choice.json")
    with open(out_file, "w") as f:
        json.dump(final_choice, f, indent=4)
    print(f"\\nWrote final choice to {out_file}:")
    print(json.dumps(final_choice, indent=2))
"""))

with open("kaggle/03_main.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb_main, f)

# ==========================================
# 2. kaggle/04_test.ipynb
# ==========================================
nb_test = nbf.v4.new_notebook()

nb_test.cells.append(nbf.v4.new_markdown_cell("""# 04_test.ipynb — Final Test Set Evaluation under FREEZE Gate
This notebook evaluates the models on the official CIFAR test set:
1. **Auto-detects** attached output from `03_main` (Stage "final") containing `final_choice.json`.
2. **Sweep Reporting**: If `results_sweep.csv` is present, logs and reports sweep rows as validation numbers only.
3. **Copies** run folders into `/kaggle/working/runs/` treating input as read-only.
4. **DRY_RUN protection**: Default `DRY_RUN = True` lists what will be tested and stops before touching test data.
5. **FREEZE Protection**: Enforces `experiments/FREEZE.md`.
6. **Write-once**: `eval_test.py` writes `test_results.json` once per run and refuses to overwrite.
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

if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")

if not os.path.exists("eval_test.py"):
    if os.path.exists("flower-lite"):
        os.chdir("flower-lite")
        os.system("git pull")
    elif os.path.exists("/kaggle/working/flower-lite"):
        os.chdir("/kaggle/working/flower-lite")
        os.system("git pull")
    else:
        os.system(f"git clone {REPO_URL} flower-lite")
        os.chdir("flower-lite")

for p in [os.path.abspath("src"), "/kaggle/working/flower-lite/src"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

os.system("pip install -e .")
import flowerlite
print("Working Directory:", os.getcwd())
print("FlowerLite package successfully imported from:", flowerlite.__file__)
assert os.path.exists("eval_test.py"), "eval_test.py must exist!"
"""))

nb_test.cells.append(nbf.v4.new_code_cell("""# 1. Auto-detect training output (final_choice.json from stage final)
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
        "Please attach the output of 03_main (STAGE='final') as an input to this notebook!"
    )
elif len(candidates) > 1:
    raise ValueError(
        f"Ambiguous training output: found multiple final_choice.json files: {candidates}. "
        "Please ensure only one final stage output dataset is attached."
    )

choice_file = Path(candidates[0])
print(f"Selected final training output config: {choice_file}")
source_dir = choice_file.parent

with open(choice_file, "r") as f:
    final_choice = json.load(f)

print("Loaded final choices (selected strictly by validation):")
print(json.dumps(final_choice, indent=2))

# Check and report sweep results as validation numbers only
sweep_csv_candidates = glob.glob(str(source_dir / "*results_sweep.csv")) + \
                       glob.glob("/kaggle/input/*/results_sweep.csv") + \
                       glob.glob("/kaggle/input/*/*/results_sweep.csv")

if sweep_csv_candidates:
    import pandas as pd
    print(f"\\n[SWEEP VALIDATION REPORT ONLY] Found sweep results: {sweep_csv_candidates[0]}")
    df_sw = pd.read_csv(sweep_csv_candidates[0])
    print(df_sw.to_string(index=False))

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
    print("The following chosen winner models (selected strictly by validation) WOULD be evaluated:")
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

print("Successfully generated parameterized kaggle/03_main.ipynb and kaggle/04_test.ipynb!")
