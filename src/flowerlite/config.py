import os
from dataclasses import dataclass, field

import yaml


class ConfigError(Exception):
    pass

@dataclass
class ArchitectureConfig:
    name: str = "flowerlite_l"
    stem_width: int = 64
    stage_widths: list[int] = field(default_factory=lambda: [64, 128, 256, 384])
    stage_depths: list[int] = field(default_factory=lambda: [3, 4, 6, 3])
    expansion: float = 2.0
    se_ratio: float = 0.125
    activation: str = "silu"
    drop_path_rate: float = 0.10
    attention_after_stage: int = 3
    attention_heads: int = 4
    attention_head_dim: int = 64
    attention_mlp_dim: int = 384
    head_hidden_dim: int = 768
    head_dropout: float = 0.15

@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    sync_bn: bool = False

@dataclass
class DataConfig:
    dataset: str = "cifar10"
    root: str = "./data"
    num_classes: int = 10
    global_batch_size: int = 256
    num_workers_per_gpu: int = 4
    normalization_mean: list[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalization_std: list[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])
    trivial_augment: bool = True
    mixup_prob: float = 0.0
    cutmix_prob: float = 0.0
    mix_alpha: float = 1.0
    label_smoothing: float = 0.0

@dataclass
class OptimConfig:
    optimizer: str = "sgd" # "sgd" or "adam"
    lr: float = 0.1
    momentum_beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4

@dataclass
class ScheduleConfig:
    epochs: int = 120
    warmup_epochs: int = 5
    eta_min: float = 0.0

@dataclass
class RuntimeConfig:
    seed: int = 2026
    clip_grad_norm: float = 5.0
    amp: bool = True
    ema: bool = True
    ema_decay: float = 0.9998
    print_freq: int = 20

@dataclass
class TrackingConfig:
    experiment_root: str = "experiments"
    run_name: str | None = None

@dataclass
class FlowerLiteConfig:
    model: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

def _merge_dict_into_dataclass(dc_instance, d):
    import dataclasses
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if not hasattr(dc_instance, k):
            raise ConfigError(f"Unknown key: {k}")
        
        attr_val = getattr(dc_instance, k)
        if dataclasses.is_dataclass(attr_val):
            _merge_dict_into_dataclass(attr_val, v)
        else:
            # Type casting could be added here if needed, but for now just set
            setattr(dc_instance, k, v)

def load_config(yaml_path: str) -> FlowerLiteConfig:
    if not os.path.exists(yaml_path):
        raise ConfigError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}

    cfg = FlowerLiteConfig()
    
    # Handle base config inheritance if 'base' is in data (though not strictly required, helpful for hierarchical)
    if 'base' in data:
        base_path = os.path.join(os.path.dirname(yaml_path), data.pop('base'))
        base_cfg = load_config(base_path)
        # simplistic merge by dicts
        cfg = base_cfg
    
    try:
        _merge_dict_into_dataclass(cfg, data)
    except ConfigError as e:
        raise ConfigError(f"Error parsing {yaml_path}: {e}")

    # Validate
    _validate_config(cfg)
    return cfg

def _validate_config(cfg: FlowerLiteConfig):
    if cfg.data.dataset not in ["cifar10", "cifar100"]:
        raise ConfigError(f"Invalid dataset: {cfg.data.dataset}")
    if cfg.data.dataset == "cifar10" and cfg.data.num_classes != 10:
        raise ConfigError("num_classes mismatch for cifar10")
    if cfg.data.dataset == "cifar100" and cfg.data.num_classes != 100:
        raise ConfigError("num_classes mismatch for cifar100")
    if cfg.optim.optimizer not in ["sgd", "adam"]:
        raise ConfigError(f"Unknown optimizer: {cfg.optim.optimizer}")
    
    if cfg.distributed.enabled:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if cfg.data.global_batch_size % world_size != 0:
            raise ConfigError(
                f"global_batch_size {cfg.data.global_batch_size} "
                f"must be divisible by world_size {world_size}"
            )
    
    # Relative paths remain relative.

def save_config(cfg: FlowerLiteConfig, path: str):
    import dataclasses
    d = dataclasses.asdict(cfg)
    with open(path, 'w') as f:
        yaml.safe_dump(d, f, sort_keys=False)
