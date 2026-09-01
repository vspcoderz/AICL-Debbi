"""
train.py — train Debbi on a pre-tokenized AICL corpus bin file.

Data layout (produced by data/prepare_data.py):
  data/corpus.bin     uint16 little-endian token ids, one sample per line
                      (BOS ... EOS markers included in the stream)
  data/id_map.json    {"id_to_chars": {...}, "chars_to_id": {...}, ...}

Resumable: regularly writes model+optimizer+RNG+step to checkpoints/<run>/last.pt.

Run (Colab T4 fits comfortably):
  python model/train.py
  python model/train.py --data-dir data --max-steps 5000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import Config
from transformer import Debbi


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    prog = max(0.0, min(1.0, prog))
    coeff = 0.5 * (1 + math.cos(math.pi * prog))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


class MemmapDataset:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def sample_batch(self, seq_len: int, bs: int, device: str) -> torch.Tensor:
        n = len(self.data)
        starts = np.random.randint(0, max(1, n - seq_len - 1), size=bs, dtype=np.int64)
        rows = [self.data[s:s + seq_len + 1].astype(np.int64) for s in starts]
        return torch.from_numpy(np.stack(rows)).to(device)


@torch.no_grad()
def evaluate(model: Debbi, ds: MemmapDataset, cfg: Config, device: str) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for _ in range(cfg.eval_steps):
        b = ds.sample_batch(cfg.max_seq_len, cfg.batch_size, device)
        _, loss = model(b[:, :-1], b[:, 1:])
        total_loss += loss.item() * b[:, :-1].numel()
        total_tokens += b[:, :-1].numel()
    model.train()
    return total_loss / total_tokens


@torch.no_grad()
def init_rope_param():
    pass  # rope is a plain buffer; nothing to do


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--nano", action="store_true", help="tiny run for smoke tests")
    args, _ = ap.parse_known_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.nano:
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn_dim, cfg.max_seq_len = 128, 2, 4, 512, 256
        cfg.grad_checkpointing, cfg.compile_model = False, False

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | " + (torch.cuda.get_device_name(0) if device == "cuda" else ""))

    ds = MemmapDataset(os.path.join(cfg.data_dir, "corpus.bin"))
    print(f"corpus tokens: {len(ds.data):,}")
    with open(os.path.join(cfg.data_dir, "id_map.json"), encoding="utf-8") as fh:
        id_map = json.load(fh)
    cfg.vocab_size = len(id_map["id_to_chars"])
    print(f"vocab_size: {cfg.vocab_size}")

    model = Debbi(cfg).to(device)
    print(f"params: {model.n_params / 1e6:.1f}M | dtype: {cfg.dtype}")

    if cfg.dtype == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        if device != "cuda":
            cfg.dtype = "float32"
        else:
            cfg.dtype = "float16"

    dtype_t = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and cfg.dtype == "float16") else None

    step, best_ppl = 0, float("inf")
    ckpt_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    resume = os.path.join(ckpt_dir, "last.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        step = ck["step"]
        print(f"resumed from {resume} at step {step}")

    use_amp = device == "cuda" and cfg.dtype in ("float16", "bfloat16")
    gc = cfg.grad_checkpointing and device == "cuda"
    eff_bs = cfg.batch_size * cfg.grad_accum
    print(f"effective batch: {eff_bs} | amp: {use_amp} | grad_checkpointing: {gc}")

    def forward_batch(x, y):
        if gc:
            h = model.tok_emb(x)
            for blk in model.layers:
                h = checkpoint(blk, h, use_reentrant=False)
            h = model.norm(h)
            logits = model.lm_head(h)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype_t):
                logits, _ = model(x, None)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / cfg.grad_accum

    t0 = time.time()
    while step < cfg.max_steps:
        b = ds.sample_batch(cfg.max_seq_len, cfg.batch_size, device)
        x, y = b[:, :-1], b[:, 1:]

        if scaler is not None:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype_t):
                loss = forward_batch(x, y)
            scaler.scale(loss).backward()
        else:
            loss = forward_batch(x, y)
            loss.backward()

        if (step + 1) % cfg.grad_accum == 0 or step + 1 == cfg.max_steps:
            lr = get_lr(step, cfg)
            for g in optim.param_groups:
                g["lr"] = lr
            if scaler is not None:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)

        if (step + 1) % cfg.log_every == 0:
            dt = time.time() - t0
            tok_s = cfg.batch_size * cfg.max_seq_len / max(dt, 1e-9)
            shown = loss.item() * cfg.grad_accum
            print(f"step {step+1:>7}/{cfg.max_steps} loss {shown:.4f} ppl {math.exp(shown):.2f} "
                  f"lr {lr:.2e} {tok_s*cfg.grad_accum*1e-3:.0f}k tok/s")
            t0 = time.time()

        if (step + 1) % cfg.eval_every == 0:
            ppl = math.exp(evaluate(model, ds, cfg, device))
            best_ppl = min(best_ppl, ppl)
            print(f"  eval ppl: {ppl:.3f} (best {best_ppl:.3f})")

        if (step + 1) % cfg.save_every == 0:
            ck = {"model": model.state_dict(), "optim": optim.state_dict(),
                  "step": step + 1, "cfg": dataclasses.asdict(cfg)}
            torch.save(ck, resume)
            torch.save(ck, os.path.join(ckpt_dir, f"step{step + 1}.pt"))
            print(f"  saved {ckpt_dir}/step{step + 1}.pt")
        step += 1

    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "step": step, "cfg": dataclasses.asdict(cfg)}, resume)
    print(f"done. final weights at {resume}")


if __name__ == "__main__":
    main()