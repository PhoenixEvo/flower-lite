import csv
import json
import os

if os.name == 'posix':
    import fcntl
import datetime
from pathlib import Path


class TrackingError(Exception):
    pass

class ExperimentTracker:
    def __init__(self, exp_root: str, run_id: str, rank: int = 0):
        self.exp_root = Path(exp_root)
        self.run_id = run_id
        self.rank = rank
        self.run_dir = self.exp_root / run_id
        
        if self.rank == 0:
            if self.run_dir.exists():
                raise TrackingError(f"Run directory {self.run_dir} already exists. Cannot overwrite.")
            self.run_dir.mkdir(parents=True, exist_ok=False)
            
            # Write initial manifest placeholder
            self.manifest_path = self.run_dir / "manifest.json"
            
    def _assert_rank_zero(self):
        if self.rank != 0:
            raise TrackingError("Only rank 0 can write to tracking files.")

    def write_manifest(self, manifest_data: dict):
        self._assert_rank_zero()
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
            
    def save_config(self, config_str: str):
        self._assert_rank_zero()
        with open(self.run_dir / "config.yaml", 'w') as f:
            f.write(config_str)
            
    def append_ledger(self, split: str, metrics: dict, status: str, seed: int, config_slug: str):
        self._assert_rank_zero()
        if split not in ["val", "test"]:
            raise ValueError(f"Split must be 'val' or 'test', got {split}")
            
        ledger_path = self.exp_root / "results.csv"
        
        file_exists = ledger_path.exists()
        
        if file_exists:
            # Check for existing run_id overwrite attempt. We consider appending multiple rows for same run ok
            # but wait, user says "attempting to rewrite an existing row raises". 
            # "append-only results ledger... exactly one row per attempted run"
            with open(ledger_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("run_id") == self.run_id and row.get("split") == split:
                        raise TrackingError(f"Row for run {self.run_id} and split {split} already exists in ledger.")
        
        row_dict = {
            "run_id": self.run_id,
            "split": split,
            "status": status,
            "seed": seed,
            "config_slug": config_slug,
            "top1": metrics.get("top1", ""),
            "top5": metrics.get("top5", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "nll": metrics.get("nll", ""),
            "ece": metrics.get("ece", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        fieldnames = list(row_dict.keys())
        
        # Write to temp file then atomic append if posix, or just write
        # "Write to a temp line then append in one call"
        # Since this is append, we can format a CSV line and open in 'a' mode
        import io
        sio = io.StringIO()
        writer = csv.DictWriter(sio, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        line = sio.getvalue()
        
        if os.name == 'posix':
            with open(ledger_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line)
                fcntl.flock(f, fcntl.LOCK_UN)
        else:
            with open(ledger_path, 'a') as f:
                f.write(line)

    def save_checkpoint(self, state_dict: dict, name: str):
        self._assert_rank_zero()
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / name
        import torch
        if checkpoint_path.exists():
            raise TrackingError(f"Checkpoint {name} already exists. Overwriting is not allowed.")
        torch.save(state_dict, checkpoint_path)
