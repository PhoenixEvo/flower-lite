
import nbformat as nbf

# --- 01_sweep.ipynb ---
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('# Kaggle Sweep Phase'))
nb.cells.append(nbf.v4.new_code_cell('!nvidia-smi\nimport torch\nprint("PyTorch:", torch.__version__, "GPUs:", torch.cuda.device_count())'))
nb.cells.append(nbf.v4.new_code_cell('''import os
import shutil

REPO_SOURCE = "git" # change to "kaggle" if using an attached dataset
REPO_URL = "<YOUR_REPO_URL>"

if REPO_SOURCE == "git":
    if not os.path.exists("flowerlite"):
        os.system(f"git clone {REPO_URL} flowerlite")
    os.chdir("flowerlite")
else:
    # Example dataset path
    os.system("cp -r /kaggle/input/flowerlite-repo /kaggle/working/flowerlite")
    os.chdir("/kaggle/working/flowerlite")

os.system("pip install -e .")
import flowerlite # verify import
print("Successfully imported flowerlite!")

import torchvision
torchvision.datasets.CIFAR10(root='/kaggle/working/data', download=True)
torchvision.datasets.CIFAR100(root='/kaggle/working/data', download=True)
'''))

nb.cells.append(nbf.v4.new_code_cell('''import subprocess, time, os

def measure_sec_per_epoch(gpu_id="0", batch_size=256):
    print(f"Measuring timing on GPU {gpu_id}...")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
    cmd = ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", 
           "--optimizer", "sgd", "--lr", "0.1", "--epochs", "1", "--run-id", "timing_run", 
           "--max-steps", "115", "--data-root", "/kaggle/working/data"]
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    elapsed_100_steps = None
    for line in res.stdout.split("\\n"):
        if "Timing (Steps 10-110)" in line:
            elapsed_100_steps = float(line.split(":")[1].replace("s", "").replace("-", "").strip())
            break
            
    if elapsed_100_steps is None:
        print(res.stdout)
        raise RuntimeError("Could not measure timing cleanly. See logs above.")
    
    steps_per_epoch = 45000 / batch_size
    sec_per_epoch = (elapsed_100_steps / 100) * steps_per_epoch
    
    print(f"100 steps took {elapsed_100_steps:.2f}s (excl. start-up & validation). Steps/s: {100 / elapsed_100_steps:.2f}")
    return sec_per_epoch

sec_per_epoch = measure_sec_per_epoch("0")
print(f"Measured sec/epoch: {sec_per_epoch:.2f}")

planned_epochs = 4 * 30 # 4 configs * 30 epochs
total_hours = (sec_per_epoch * planned_epochs) / 3600
print(f'Projected wall-clock: {total_hours:.2f} hours')

if total_hours > 6.0:
    raise Exception("Projection exceeds 6 hours. Consider lowering epoch count. Halting.")

print('Starting Sweep in background...')
cmds_0 = [
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", "--optimizer", "sgd", "--lr", "0.10", "--epochs", "30", "--run-id", "sweep_c10_sgd_10", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", "--optimizer", "sgd", "--lr", "0.15", "--epochs", "30", "--run-id", "sweep_c10_sgd_15", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", "--optimizer", "adam", "--lr", "0.001", "--epochs", "30", "--run-id", "sweep_c10_adam_001", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", "--optimizer", "adam", "--lr", "0.002", "--epochs", "30", "--run-id", "sweep_c10_adam_002", "--data-root", "/kaggle/working/data"]
]

cmds_1 = [
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar100", "--optimizer", "sgd", "--lr", "0.10", "--epochs", "30", "--run-id", "sweep_c100_sgd_10", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar100", "--optimizer", "sgd", "--lr", "0.15", "--epochs", "30", "--run-id", "sweep_c100_sgd_15", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar100", "--optimizer", "adam", "--lr", "0.001", "--epochs", "30", "--run-id", "sweep_c100_adam_001", "--data-root", "/kaggle/working/data"],
    ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar100", "--optimizer", "adam", "--lr", "0.002", "--epochs", "30", "--run-id", "sweep_c100_adam_002", "--data-root", "/kaggle/working/data"]
]

def run_sequence(cmds, gpu, log_file):
    with open(log_file, "w") as f:
        f.write(f"--- Starting GPU {gpu} sequence ---\\n")
    
    script_path = f"/kaggle/working/run_gpu{gpu}.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\\n")
        for cmd in cmds:
            f.write(" ".join(cmd) + f" >> {log_file} 2>&1\\n")
            f.write(f"cp -r experiments/{cmd[13]} /kaggle/working/\\n") # copy run dir out
    
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    return subprocess.Popen(["bash", script_path], env=env)

p0 = run_sequence(cmds_0, "0", "/kaggle/working/log_gpu0.txt")
p1 = run_sequence(cmds_1, "1", "/kaggle/working/log_gpu1.txt")

import sys
while p0.poll() is None or p1.poll() is None:
    time.sleep(30)
    print("--- GPU 0 Last 15 Lines ---")
    os.system("tail -n 15 /kaggle/working/log_gpu0.txt")
    print("--- GPU 1 Last 15 Lines ---")
    os.system("tail -n 15 /kaggle/working/log_gpu1.txt")
    sys.stdout.flush()

p0.wait()
p1.wait()

print("Sweep Complete. Output directories copied to /kaggle/working/")
# Compile summary json
import json
with open("/kaggle/working/summary.json", "w") as f:
    json.dump({"status": "sweep_completed"}, f)
'''))

with open('kaggle/01_sweep.ipynb', 'w') as f:
    nbf.write(nb, f)

# --- 02_final.ipynb ---
nb2 = nbf.v4.new_notebook()
nb2.cells.append(nbf.v4.new_markdown_cell('# Kaggle Final Phase'))
nb2.cells.append(nbf.v4.new_code_cell('!nvidia-smi\nimport torch\nprint("PyTorch:", torch.__version__, "GPUs:", torch.cuda.device_count())'))
nb2.cells.append(nbf.v4.new_code_cell('''import os
import shutil

REPO_SOURCE = "git" # change to "kaggle" if using an attached dataset
REPO_URL = "<YOUR_REPO_URL>"

if REPO_SOURCE == "git":
    if not os.path.exists("flowerlite"):
        os.system(f"git clone {REPO_URL} flowerlite")
    os.chdir("flowerlite")
else:
    os.system("cp -r /kaggle/input/flowerlite-repo /kaggle/working/flowerlite")
    os.chdir("/kaggle/working/flowerlite")

os.system("pip install -e .")
import flowerlite # verify import
print("Successfully imported flowerlite!")

import torchvision
torchvision.datasets.CIFAR10(root='/kaggle/working/data', download=True)
torchvision.datasets.CIFAR100(root='/kaggle/working/data', download=True)
'''))

nb2.cells.append(nbf.v4.new_code_cell('''import json
config = {
    'cifar10': {'optimizer': 'sgd', 'lr': 0.1, 'epochs': None},
    'cifar100': {'optimizer': 'sgd', 'lr': 0.1, 'epochs': None}
}
if not os.path.exists('/kaggle/working/final_config.json'):
    with open('/kaggle/working/final_config.json', 'w') as f:
        json.dump(config, f)
print('Edit /kaggle/working/final_config.json as needed. Fill in epochs!')
'''))

nb2.cells.append(nbf.v4.new_code_cell('''import subprocess, time, json, os, sys

with open('/kaggle/working/final_config.json') as f:
    cfg = json.load(f)

if cfg['cifar10']['epochs'] is None or cfg['cifar100']['epochs'] is None:
    raise ValueError("Epochs in final_config.json cannot be null. Please edit the JSON and provide epoch counts.")

FINAL_TRAIN_ON_FULL = False
if FINAL_TRAIN_ON_FULL:
    print("WARNING: Training on full 50,000 examples (mode=final). No validation split is available!")
    print("WARNING: You cannot choose between EMA vs RAW weights using validation metrics! (Will default to EMA).")
    mode_str = "final"
else:
    mode_str = "dev"
    
def measure_sec_per_epoch(gpu_id="0", batch_size=256):
    print(f"Measuring timing on GPU {gpu_id}...")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
    cmd = ["python", "train.py", "--config", "configs/base.yaml", "--dataset", "cifar10", 
           "--optimizer", "sgd", "--lr", "0.1", "--epochs", "1", "--run-id", "timing_run", 
           "--max-steps", "115", "--data-root", "/kaggle/working/data"]
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    elapsed_100_steps = None
    for line in res.stdout.split("\\n"):
        if "Timing (Steps 10-110)" in line:
            elapsed_100_steps = float(line.split(":")[1].replace("s", "").replace("-", "").strip())
            break
            
    if elapsed_100_steps is None:
        print(res.stdout)
        raise RuntimeError("Could not measure timing cleanly. See logs above.")
    
    steps_per_epoch = 50000 / batch_size if FINAL_TRAIN_ON_FULL else 45000 / batch_size
    sec_per_epoch = (elapsed_100_steps / 100) * steps_per_epoch
    
    print(f"100 steps took {elapsed_100_steps:.2f}s (excl. start-up & validation). Steps/s: {100 / elapsed_100_steps:.2f}")
    return sec_per_epoch

sec_per_epoch = measure_sec_per_epoch("0")
print(f"Measured sec/epoch: {sec_per_epoch:.2f}")

planned_epochs = max(cfg['cifar10']['epochs'], cfg['cifar100']['epochs'])
total_hours = (sec_per_epoch * planned_epochs) / 3600
print(f'Projected wall-clock: {total_hours:.2f} hours')

if total_hours > 5.0:
    raise Exception("Projection exceeds 5 hours. Consider lowering epoch count. Halting.")

print('Starting Final in background...')
os.makedirs("experiments", exist_ok=True)
with open("experiments/FREEZE.md", "w") as f:
    f.write("FREEZE")

# Write manifest choice
with open("experiments/manifest_choice.txt", "w") as f:
    f.write(f"FINAL_TRAIN_ON_FULL={FINAL_TRAIN_ON_FULL}")

run_id_c10 = f"final_c10_{int(time.time())}"
run_id_c100 = f"final_c100_{int(time.time())}"

c10_cmd = f"python train.py --config configs/base.yaml --dataset cifar10 --optimizer {cfg['cifar10']['optimizer']} --lr {cfg['cifar10']['lr']} --epochs {cfg['cifar10']['epochs']} --run-id {run_id_c10} --data-root /kaggle/working/data --resume --mode {mode_str} >> /kaggle/working/log_final_c10.txt 2>&1 && python eval_test.py --run-id {run_id_c10} --dataset cifar10 >> /kaggle/working/log_final_c10.txt 2>&1 && cp -r experiments/{run_id_c10} /kaggle/working/final_c10"

c100_cmd = f"python train.py --config configs/base.yaml --dataset cifar100 --optimizer {cfg['cifar100']['optimizer']} --lr {cfg['cifar100']['lr']} --epochs {cfg['cifar100']['epochs']} --run-id {run_id_c100} --data-root /kaggle/working/data --resume --mode {mode_str} >> /kaggle/working/log_final_c100.txt 2>&1 && python eval_test.py --run-id {run_id_c100} --dataset cifar100 >> /kaggle/working/log_final_c100.txt 2>&1 && cp -r experiments/{run_id_c100} /kaggle/working/final_c100"

with open("/kaggle/working/run_gpu0.sh", "w") as f:
    f.write("#!/bin/bash\\n")
    f.write(c10_cmd + "\\n")
    
with open("/kaggle/working/run_gpu1.sh", "w") as f:
    f.write("#!/bin/bash\\n")
    f.write(c100_cmd + "\\n")

env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

p0 = subprocess.Popen(["bash", "/kaggle/working/run_gpu0.sh"], env=env0)
p1 = subprocess.Popen(["bash", "/kaggle/working/run_gpu1.sh"], env=env1)

while p0.poll() is None or p1.poll() is None:
    time.sleep(30)
    print("--- GPU 0 Last 15 Lines ---")
    os.system("tail -n 15 /kaggle/working/log_final_c10.txt")
    print("--- GPU 1 Last 15 Lines ---")
    os.system("tail -n 15 /kaggle/working/log_final_c100.txt")
    sys.stdout.flush()

p0.wait()
p1.wait()

print("Final Complete. Output directories copied to /kaggle/working/")
with open("/kaggle/working/summary.json", "w") as f:
    json.dump({"status": "final_completed"}, f)
'''))

with open('kaggle/02_final.ipynb', 'w') as f:
    nbf.write(nb2, f)
