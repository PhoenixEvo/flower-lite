import pytest

from flowerlite.config import ConfigError, load_config, save_config


def test_load_cifar10_config():
    cfg = load_config("configs/cifar10/final.yaml")
    assert cfg.data.dataset == "cifar10"
    assert cfg.data.num_classes == 10
    assert cfg.optim.optimizer == "sgd"
    
def test_load_cifar100_config():
    cfg = load_config("configs/cifar100/final.yaml")
    assert cfg.data.dataset == "cifar100"
    assert cfg.data.num_classes == 100

def test_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("optim:\n  fake_key: 123")
    with pytest.raises(ConfigError):
        load_config(str(p))

def test_num_classes_mismatch(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  dataset: cifar10\n  num_classes: 100")
    with pytest.raises(ConfigError):
        load_config(str(p))

def test_save_and_round_trip(tmp_path):
    cfg1 = load_config("configs/cifar10/final.yaml")
    p = tmp_path / "saved.yaml"
    save_config(cfg1, str(p))
    cfg2 = load_config(str(p))
    
    assert cfg1.data.dataset == cfg2.data.dataset
    assert cfg1.optim.lr == cfg2.optim.lr
    assert cfg1.data.root == cfg2.data.root
    assert cfg1.distributed.enabled == cfg2.distributed.enabled

def test_distributed_config_validation(monkeypatch, tmp_path):
    # Test divisibility assertion
    monkeypatch.setenv("WORLD_SIZE", "2")
    p = tmp_path / "dist.yaml"
    p.write_text("distributed:\n  enabled: true\ndata:\n  dataset: cifar10\n  global_batch_size: 257")
    with pytest.raises(ConfigError, match="divisible"):
        load_config(str(p))

def test_relative_paths_preserved(tmp_path):
    # Base contains data.root: "./data"
    cfg1 = load_config("configs/cifar10/final.yaml")
    assert cfg1.data.root == "./data"
    
    p = tmp_path / "saved.yaml"
    save_config(cfg1, str(p))
    
    cfg2 = load_config(str(p))
    assert cfg2.data.root == "./data"
    assert "c:" not in cfg2.data.root.lower()
    assert "/" in cfg2.data.root or "." in cfg2.data.root
