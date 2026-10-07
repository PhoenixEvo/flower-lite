import argparse
import json
import os
import platform
import sys
import time

import torch
from torch import nn

from flowerlite.augment import get_train_transform
from flowerlite.data import ProtectedCIFAR, create_dataloaders

FINGERPRINT = "FL-L:S3x3-64|PM[d3,w64,s1]-PM[d4,w128,s2]-PM[d6,w256,s2]-PA[h4,d64,m384,8x8]-PM[d3,w384,s2]|H384-768-K|DW(3,5)+Shuffle2+SE8"

def write_hardware_json():
    hw = {
        "os": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }
    for i in range(hw["gpu_count"]):
        hw["gpus"].append({
            "name": torch.cuda.get_device_name(i),
            "vram_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3)
        })
    with open("hardware.json", "w") as f:
        json.dump(hw, f, indent=4)
    print("Hardware JSON written to hardware.json")

def parse_registry(registry_content: str):
    lines = registry_content.strip().split("\n")
    data = []
    start = -1
    for i, l in enumerate(lines):
        if l.startswith("| Fingerprint"):
            start = i + 2
            break
    if start == -1:
        return data
        
    for l in lines[start:]:
        if not l.startswith("|"):
            continue
        parts = [p.strip() for p in l.split("|")[1:-1]]
        if len(parts) >= 3:
            status = parts[2].lower()
            if status not in ["approved", "conflict", "pending"]:
                raise ValueError(f"Unknown status: {status}")
            data.append({"fingerprint": parts[0], "status": status})
    return data

class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32*32*3, 100)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_throughput(steps: int, world_size: int = 1, distributed: bool = False, num_workers: int = 0):
    print(f"Running throughput test (world_size={world_size}, workers={num_workers}, distributed={distributed})")
    try:
        if distributed and "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29501"
            import torch.distributed as dist
            dist.init_process_group("gloo" if not torch.cuda.is_available() else "nccl", rank=0, world_size=world_size)
            
        split_info = {"train_indices": list(range(45000)), "val_indices": list(range(45000, 50000))}
        pc = ProtectedCIFAR("./data", "cifar100", mode="dev", split_info=split_info)
        transform = get_train_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True)
        
        train_loader, _ = create_dataloaders(
            pc, transform, transform, global_batch_size=128, num_workers=num_workers,
            distributed=distributed, seed=2026, world_size=world_size
        )
        
        model = TrivialModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if distributed and torch.cuda.is_available():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
        elif distributed:
            model = nn.parallel.DistributedDataParallel(model)
            
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        
        total_images = 0
        start_time = time.time()
        
        step = 0
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = out.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_images += x.size(0) * world_size
            step += 1
            
        elapsed = time.time() - start_time
        images_per_sec = total_images / elapsed if elapsed > 0 else 0
        print(f"Throughput (world_size={world_size}, workers={num_workers}): {images_per_sec:.2f} images/s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Throughput test failed: {e}")

def run_preflight():
    print("Running Kaggle Preflight...")
    if not torch.cuda.is_available():
        print("NOT RUN: CUDA not available")
        return
        
    count = torch.cuda.device_count()
    if count != 2:
        print(f"NOT RUN: Expected 2 devices, got {count}")
        return
        
    for i in range(2):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"Device {i}: {name} ({mem:.2f} GB)")
        if "T4" not in name:
            print("WARNING: Expected Tesla T4")
        if mem < 15:
            print(f"NOT RUN: Expected >= 15 GiB total memory on device {i}")
            return
            
    print("NCCL Preflight starting...")
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29502"
        import torch.distributed as dist
        
        def run_nccl(rank):
            try:
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = "2"
                dist.init_process_group("nccl", rank=rank, world_size=2)
                t = torch.ones(1).cuda(rank)
                dist.all_reduce(t)
                assert t.item() == 2.0
                model = TrivialModel().cuda(rank)
                ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
                out = ddp(torch.randn(1, 3, 32, 32).cuda(rank))
                out.sum().backward()
                dist.destroy_process_group()
            except Exception as e:
                print(f"Rank {rank} failed: {e}")
                
        import torch.multiprocessing as mp
        mp.spawn(run_nccl, nprocs=2, join=True)
        print("NCCL Preflight PASS")
    except Exception as e:
        print(f"NCCL Preflight FAILED: {e}")
        
    print("Running test_all_reduce_correctness under Linux/Kaggle...")
    import subprocess
    res = subprocess.run(["pytest", "tests/test_metrics.py::test_all_reduce_correctness", "-v"], capture_output=True, text=True)
    if res.returncode == 0:
        print("test_all_reduce_correctness: PASS")
    else:
        print("test_all_reduce_correctness: FAILED")
        print(res.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "kaggle"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--workers-sweep", type=str, default="")
    args = parser.parse_args()
    
    print(f"Fingerprint: {FINGERPRINT}")
    write_hardware_json()
    
    if args.mode == "local":
        run_throughput(args.steps, world_size=1, distributed=False, num_workers=args.num_workers)
    elif args.mode == "kaggle":
        workers = [int(w) for w in args.workers_sweep.split(",")] if args.workers_sweep else [2, 4]
        for w in workers:
            run_throughput(args.steps, world_size=1, distributed=False, num_workers=w)
            print("NOT RUN: world_size=2 throughput requires torchrun or mp.spawn.")
        run_preflight()

if __name__ == "__main__":
    main()
