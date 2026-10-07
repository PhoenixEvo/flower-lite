import glob
import json
import shutil
from pathlib import Path

import pytest


def test_kaggle_test_notebook_path_logic(tmp_path):
    base_tmp = tmp_path / "kaggle_test"
    fake_input = base_tmp / "input" / "flowerlite_03_main"
    fake_input_runs = fake_input / "runs"
    fake_input_runs.mkdir(parents=True)

    fake_choice = {
        "cifar10": {"run_id": "c10_sgd_0.10", "weight_type": "ema"},
        "cifar100": {"run_id": "c100_sgd_0.10", "weight_type": "ema"}
    }
    with open(fake_input / "final_choice.json", "w") as f:
        json.dump(fake_choice, f, indent=2)

    for r_name in ["c10_sgd_0.10", "c10_adam_0.001", "c100_sgd_0.10"]:
        r_dir = fake_input_runs / r_name
        r_dir.mkdir()
        (r_dir / "best.pt").write_text("dummy_weights")
        (r_dir / "last.pt").write_text("dummy_weights")
        (r_dir / "history.csv").write_text("epoch,train_loss,val_loss,val_top1_raw,val_top1_ema,lr,time\n0,2.3,2.3,10.0,10.0,0.1,10.0\n")
        (r_dir / "FINISHED").write_text("FINISHED\n")

    search_patterns = [
        str(base_tmp / "input" / "*" / "final_choice.json"),
        str(base_tmp / "input" / "*" / "*" / "final_choice.json")
    ]
    candidates = []
    for p in search_patterns:
        candidates.extend(glob.glob(p))

    assert len(candidates) == 1
    choice_file = Path(candidates[0])

    fake_working = base_tmp / "working"
    fake_working_runs = fake_working / "runs"
    fake_working_runs.mkdir(parents=True)

    source_runs_dir = choice_file.parent / "runs"
    copied = []
    for item in source_runs_dir.iterdir():
        if item.is_dir():
            dest = fake_working_runs / item.name
            dest.mkdir(parents=True, exist_ok=True)
            for fname in ["best.pt", "last.pt", "history.csv", "FINISHED"]:
                src_f = item / fname
                if src_f.exists():
                    shutil.copy2(src_f, dest / fname)
            copied.append(item.name)

    assert len(copied) == 3
    for r in copied:
        assert (fake_working_runs / r / "best.pt").exists()
        assert (fake_working_runs / r / "history.csv").exists()

    # Ambiguity test
    fake_dup = base_tmp / "input" / "duplicate_output"
    fake_dup.mkdir()
    (fake_dup / "final_choice.json").write_text("{}")
    dup_candidates = []
    for p in search_patterns:
        dup_candidates.extend(glob.glob(p))

    assert len(dup_candidates) == 2
