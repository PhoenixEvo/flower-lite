# Kaggle Preflight Package

Run these exact, copy-ready cells in your Kaggle notebook (using the PyTorch 2.x kernel with 2x T4 GPUs):

### Cell 1: Accelerator check
```bash
!nvidia-smi
!python -c "import torch; print('PyTorch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"
```

### Cell 2: Get repo and install
```bash
# Option A: Clone from git
!git clone <YOUR_REPO_URL> flowerlite
%cd flowerlite

# Option B: Use uploaded dataset path (uncomment if using Kaggle dataset upload instead)
# %cd /kaggle/input/your-flowerlite-repo

!pip install -e .
```

### Cell 3: CPU Info
```bash
!python -c "import os; print('CPU Count:', os.cpu_count())"
```

### Cell 4: Verify Gloo/NCCL correctly skips or passes locally
```bash
!pytest tests/test_metrics.py::test_all_reduce_correctness -v -rs
```

### Cell 5: Run Benchmark and Worker Sweep
```bash
!python tools/benchmark.py --mode kaggle --steps 200 --workers-sweep 2,4
```

### Cell 6: Inspect Hardware Output
```bash
!cat hardware.json
```
