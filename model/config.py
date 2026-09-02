"""Model configuration for Debbi."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # Arch
    vocab_size: int = 4096
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_weights: bool = True
    use_sdp_attn: bool = True   # False forces a plain XLA-safe matmul attention

    # Training
    batch_size: int = 8          # per step (after gradient accumulation)
    grad_accum: int = 1          # effective batch = batch_size * grad_accum
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 20000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 1337
    save_every: int = 1000
    eval_every: int = 500
    eval_steps: int = 40
    log_every: int = 10
    dtype: str = "bfloat16"
    grad_checkpointing: bool = True
    compile_model: bool = False

    # Paths
    data_dir: str = "data"               # corpus.bin + id_map.json here
    out_dir: str = "checkpoints"
    run_name: str = "debbi-150m"

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return Config(**json.load(fh))