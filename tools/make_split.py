import argparse
import hashlib
import json
import os

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = "./data"
    os.makedirs(root, exist_ok=True)
    
    if args.dataset == "cifar10":
        ds = CIFAR10(root=root, train=True, download=True)
        num_classes = 10
        train_per_class = 4500
        val_per_class = 500
    else:
        ds = CIFAR100(root=root, train=True, download=True)
        num_classes = 100
        train_per_class = 450
        val_per_class = 50

    targets = np.array(ds.targets)
    
    rng = np.random.RandomState(args.seed)
    
    train_indices = []
    val_indices = []
    
    for c in range(num_classes):
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)
        train_indices.extend(c_idx[:train_per_class].tolist())
        val_indices.extend(c_idx[train_per_class:train_per_class+val_per_class].tolist())
        
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    
    split_info = {
        "train_indices": train_indices,
        "val_indices": val_indices
    }
    
    split_json = json.dumps(split_info, sort_keys=True)
    split_hash = hashlib.sha256(split_json.encode('utf-8')).hexdigest()
    
    if args.dry_run:
        print(f"Dataset: {args.dataset}")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        print(f"Split SHA-256: {split_hash}")
    else:
        out_path = f"configs/{args.dataset}_split_{args.seed}.json"
        with open(out_path, "w") as f:
            f.write(split_json)
        print(f"Wrote {out_path} with SHA-256 {split_hash}")

if __name__ == "__main__":
    main()
