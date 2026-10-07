import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from flowerlite.augment import get_train_transform
from flowerlite.config import load_config
from flowerlite.data import ProtectedCIFAR, create_dataloaders
from flowerlite.models.flowerlite import FlowerLiteL


class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.model = model
        self.updates = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = d * self.shadow[name] + (1.0 - d) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

def get_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    parser.add_argument("--force-mix", choices=["none", "mixup", "cutmix", "auto"], default="auto")
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=None)
    args = parser.parse_args()

    total_run_start = time.time()
    checkpoint_dir = Path("experiments") / args.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    finished_marker = checkpoint_dir / "FINISHED"

    if args.resume and finished_marker.exists():
        print(f"--- Run {args.run_id} is already FINISHED. Skipping. ---")
        return

    print(f"--- Training {args.run_id} ---")
    print(f"Dataset: {args.dataset}, Optimizer: {args.optimizer}, LR: {args.lr}, Epochs: {args.epochs}, Mode: {args.mode}")

    cfg = load_config(args.config)
    num_classes = 10 if args.dataset == "cifar10" else 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowerLiteL(num_classes=num_classes).to(device)
    
    # Optimizer
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        # Adam with decoupled weight decay (AdamW style, so weight_decay=5e-2 scaled by lr)
        # Note: The spec says "Adam: AdamW-style decoupled wd 5e-2 scaled as documented; betas (0.9, 0.999). Use torch.optim.Adam"
        # PyTorch Adam doesn't natively do AdamW decoupled weight decay correctly unless we do it manually or use AdamW.
        # But the spec explicitly says "Use torch.optim.Adam, not AdamW".
        # We must implement decoupled weight decay manually inside the loop, so set wd=0 in optimizer!
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    ema = EMA(model)

    # Data
    if args.force_mix == "auto":
        mix_status = "auto (MixUp/CutMix 50/50)" if args.epochs >= 100 else "disabled (epochs < 100)"
    else:
        mix_status = f"forced ({args.force_mix})"
    print(f"MixUp/CutMix configuration: {mix_status}")
    
    # We need a manual split info for the development 45k/5k
    split_info = None
    if os.path.exists("experiments/splits"):
        pass # In a real Kaggle we'd make the split. We will make it dynamically if missing.
        
    train_transform = get_train_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True)
    eval_transform = get_train_transform([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False)
    
    # Loaders
    if args.mode == "dev":
        split_info = {"train_indices": list(range(45000)), "val_indices": list(range(45000, 50000))}
        dataset_obj = ProtectedCIFAR(args.data_root, args.dataset, mode="dev", split_info=split_info)
    else:
        dataset_obj = ProtectedCIFAR(args.data_root, args.dataset, mode="final", split_info=None)
        
    train_loader, val_loader = create_dataloaders(
        dataset_obj, train_transform, eval_transform, 
        global_batch_size=256, num_workers=args.num_workers,
        distributed=False, seed=2026, world_size=1
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 3

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    start_epoch = 0
    checkpoint_dir = Path("experiments") / args.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        chk_path = checkpoint_dir / "last.pt"
        if chk_path.exists():
            print(f"Resuming from {chk_path}")
            ckpt = torch.load(chk_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1

    csv_path = checkpoint_dir / "history.csv"
    if not args.resume or not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_top1_raw,val_top1_ema,lr,time\n")

    step_10_time = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        
        for step, (x, y) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step
            if global_step == 10:
                step_10_time = time.time()
            elif global_step == 110 and step_10_time is not None:
                step_110_time = time.time()
                print(f"--- Timing (Steps 10-110): {step_110_time - step_10_time:.4f}s ---")
                
            x, y = x.to(device), y.to(device)
            
            lr = get_cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
                
            if args.optimizer == "adam":
                # Decoupled weight decay for Adam
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad:
                            p.data.mul_(1.0 - lr * 5e-2)

            if args.force_mix == "auto":
                if args.epochs >= 100:
                    active_mix = "mixup" if torch.rand(1).item() < 0.5 else "cutmix"
                else:
                    active_mix = "none"
            else:
                active_mix = args.force_mix

            if step == 0 and epoch == 0 and active_mix != "none":
                print(f"Batch augmentation active: {active_mix}")

            if active_mix != "none":
                import numpy as np
                alpha = 1.0
                lam = np.random.beta(alpha, alpha)
                idx = torch.randperm(x.size(0)).to(device)
                y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
                y_idx_onehot = torch.nn.functional.one_hot(y[idx], num_classes=num_classes).float()
                
                if active_mix == "mixup":
                    x = lam * x + (1.0 - lam) * x[idx]
                    y = lam * y_onehot + (1.0 - lam) * y_idx_onehot
                elif active_mix == "cutmix":
                    B, _, H, W = x.shape
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = int(W * cut_rat)
                    cut_h = int(H * cut_rat)
                    cx = np.random.randint(W)
                    cy = np.random.randint(H)
                    bbx1 = np.clip(cx - cut_w // 2, 0, W)
                    bby1 = np.clip(cy - cut_h // 2, 0, H)
                    bbx2 = np.clip(cx + cut_w // 2, 0, W)
                    bby2 = np.clip(cy + cut_h // 2, 0, H)
                    x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
                    lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                    y = lam * y_onehot + (1.0 - lam) * y_idx_onehot

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(x)
                if active_mix != "none":
                    # Soft targets: need to handle label smoothing manually if mixed, or cross_entropy does it?
                    # PyTorch CrossEntropyLoss accepts soft targets. 
                    loss = torch.nn.functional.cross_entropy(out, y, label_smoothing=0.05)
                else:
                    loss = loss_fn(out, y)
                    
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            
            ema.update()
            train_loss += loss.item()
            
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Max steps ({args.max_steps}) reached. Exiting epoch.")
                break
                
        train_loss /= max(1, step + 1)
        
        # Validation
        val_t0 = time.time()
        model.eval()
        val_loss = 0.0
        correct_raw = 0
        correct_ema = 0
        total = 0
        
        if args.mode == "dev":
            ema_model = FlowerLiteL(num_classes=num_classes).to(device)
            ema_model.load_state_dict(model.state_dict())
            for name, param in ema_model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(ema.shadow[name])
                    
            ema_model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out_raw = model(x)
                    out_ema = ema_model(x)
                    val_loss += loss_fn(out_ema, y).item()
                    correct_raw += (out_raw.argmax(1) == y).sum().item()
                    correct_ema += (out_ema.argmax(1) == y).sum().item()
                    total += y.size(0)
                    
            val_loss /= len(val_loader)
            val_top1_raw = correct_raw / total * 100.0
            val_top1_ema = correct_ema / total * 100.0
        else:
            val_loss = 0.0
            val_top1_raw = 0.0
            val_top1_ema = 0.0
            
        t1 = time.time()
        val_time = t1 - val_t0
        print(f"--- Val Time: {val_time:.4f}s ---")
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Top1 Raw: {val_top1_raw:.2f}% | Top1 EMA: {val_top1_ema:.2f}% | LR: {lr:.5f} | Time: {t1-t0:.1f}s")
        
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_top1_raw:.4f},{val_top1_ema:.4f},{lr:.6f},{t1-t0:.2f}\n")
            
        # Check and save best checkpoint
        best_pt_path = checkpoint_dir / "best.pt"
        if not hasattr(main, "best_val_top1"):
            main.best_val_top1 = -1.0
            main.best_epoch = 0
        
        if val_top1_ema > main.best_val_top1 or not best_pt_path.exists():
            main.best_val_top1 = val_top1_ema
            main.best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.shadow,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "val_top1_raw": val_top1_raw,
                "val_top1_ema": val_top1_ema
            }, best_pt_path)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs or (args.max_steps > 0 and global_step >= args.max_steps):
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.shadow,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "val_top1_raw": val_top1_raw,
                "val_top1_ema": val_top1_ema
            }, checkpoint_dir / "last.pt")
            
        if args.max_steps > 0 and global_step >= args.max_steps:
            print("Max steps reached overall. Exiting.")
            break

    # If run reached epoch completion or max-steps, handle completion actions
    is_finished = (epoch + 1 == args.epochs)
    if is_finished:
        finished_marker.write_text("FINISHED\n")

    # Read history to summarize
    best_raw = 0.0
    best_ema = 0.0
    final_raw = val_top1_raw
    final_ema = val_top1_ema
    best_ep = 0
    if csv_path.exists():
        with open(csv_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split(",")
                ep_idx = int(parts[0])
                r_val = float(parts[3])
                e_val = float(parts[4])
                if e_val > best_ema:
                    best_ema = e_val
                    best_raw = r_val
                    best_ep = ep_idx
            final_parts = lines[-1].split(",")
            final_raw = float(final_parts[3])
            final_ema = float(final_parts[4])

    wall_time = time.time() - total_run_start
    peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    # Export artifacts if requested
    if args.export_dir:
        import shutil
        exp_path = Path(args.export_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        for item in ["history.csv", "best.pt", "last.pt", "FINISHED"]:
            src_file = checkpoint_dir / item
            if src_file.exists():
                shutil.copy(src_file, exp_path / item)

    # Append to results CSV if requested
    if args.results_csv and (is_finished or args.max_steps > 0):
        res_file = Path(args.results_csv)
        res_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not res_file.exists()
        with open(res_file, "a") as f:
            if write_header:
                f.write("dataset,optimizer,lr,epochs,best_val_top1_raw,best_val_top1_ema,final_val_top1_raw,final_val_top1_ema,best_epoch,wall_time,peak_vram_mb\n")
            f.write(f"{args.dataset},{args.optimizer},{args.lr},{args.epochs},{best_raw:.4f},{best_ema:.4f},{final_raw:.4f},{final_ema:.4f},{best_ep},{wall_time:.2f},{peak_vram:.2f}\n")

if __name__ == "__main__":
    main()
