"""Training configuration for WavMamba.

Ships a single protocol used in the paper:
    optimizer   : AdamW, lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-3
    scheduler   : warmup_cosine, warmup_epochs=5, floor_lr=1e-6
    epochs      : 30, batch_size=32, grad_clip=1.0
    criterion   : CrossEntropy (fixed — not configurable)
    wd_exclude  : norm/bias/A_log/D excluded from weight decay
    seeds       : (42,) by default; pass seeds=(0, 4, 8, 17, 42) for the
                  5-seed protocol reported in the paper
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainCfg:
    # Hyperparameters (the paper protocol — see module docstring).
    lr:               float          = 5e-4
    batch_size:       int            = 32
    num_epochs:       int            = 30
    grad_clip:        Optional[float]= 1.0
    weight_decay:     float          = 1e-3
    wd_exclude_norm_bias: bool       = True

    optimizer:        str            = 'adamw'
    betas:            tuple          = (0.9, 0.95)
    eps:              float          = 1e-8

    scheduler:        Optional[str]  = 'warmup_cosine'
    warmup_epochs:    int            = 5
    floor_lr:         float          = 1e-6

    seeds:            tuple          = (42,)


def default_cfg(seeds=(42,), **overrides) -> TrainCfg:
    """Return the paper protocol TrainCfg, optionally overridden."""
    cfg = TrainCfg(seeds=tuple(seeds))
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown TrainCfg field {k!r}")
        setattr(cfg, k, v)
    return cfg


def cfg_asdict(cfg: TrainCfg) -> dict:
    """Serialize TrainCfg to a JSON-friendly dict (tuples -> lists)."""
    return {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in asdict(cfg).items()
    }
