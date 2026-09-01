"""Debbi — a decoder-only transformer that natively consumes AICL tokens.

Standard GPT-style causal LM (pre-norm, RoPE, RMSNorm, SwiGLU) with one twist:
the embedding table indexes the AICL symbol vocabulary, so each encoded symbol
is exactly ONE token — the compression is built into the input representation.

Pure torch, no HF dependencies. Tested on torch >= 2.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (T, dim/2)
    return torch.cat((freqs, freqs), dim=-1)  # (T, dim)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim)  freqs: (T, head_dim)."""
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs[:T].cos().unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = freqs[:T].sin().unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_theta: float, max_seq_len: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        freq_dim = self.head_dim
        self.register_buffer("rope_freqs", precompute_rope_freqs(freq_dim, max_seq_len, rope_theta))

    def forward(self, x: torch.Tensor, seq_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # RoPE with seq_pos offset (for KV-cache generation)
        qk = q.size(-2)
        if seq_pos == 0:
            q = apply_rope(q, self.rope_freqs)
            k = apply_rope(k, self.rope_freqs)
        else:
            q = apply_rope(q, self.rope_freqs[seq_pos:seq_pos + qk])
            k = apply_rope(k, self.rope_freqs[seq_pos:seq_pos + qk])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_pos == 0))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.rope_theta, cfg.max_seq_len)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Debbi(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self._reset()

    def _reset(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        self.tok_emb.weight.data.uniform_(-math.sqrt(1 / self.cfg.vocab_size),
                                          math.sqrt(1 / self.cfg.vocab_size))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # -- KV-cache generation ----------------------------------------------
    def generate(self, ctx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_p: float = 0.9, eos_id: int | None = None) -> torch.Tensor:
        self.eval()
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                logits, cache = self._forward_cached(ctx, {})
            else:
                last = ctx[:, -1:]
                logits, cache = self._forward_cached(last, cache)
            logits = logits[:, -1, :] / temperature
            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                keep = cum - sorted_p <= top_p
                logits = torch.where(keep, sorted_p, torch.full_like(sorted_p, float("-inf")))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            ctx = torch.cat((ctx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).any():
                break
        return ctx

    @torch.no_grad()
    def _forward_cached(self, idx, cache):
        x = self.tok_emb(idx)
        B, T = idx.shape
        new_cache = {}
        for i, layer in enumerate(self.layers):
            cq = cache.get((i, "q"), None)
            ck = cache.get((i, "k"), None)
            cv = cache.get((i, "v"), None)
            seq_pos = ck.size(-2) if ck is not None else 0
            h = layer.attn_norm(x)
            q = layer.attn.wq(h).view(B, T, layer.attn.n_heads, layer.attn.head_dim).transpose(1, 2)
            k = layer.attn.wk(h).view(B, T, layer.attn.n_heads, layer.attn.head_dim).transpose(1, 2)
            v = layer.attn.wv(h).view(B, T, layer.attn.n_heads, layer.attn.head_dim).transpose(1, 2)
            q = apply_rope(q, layer.attn.rope_freqs[seq_pos:seq_pos + T])
            k = apply_rope(k, layer.attn.rope_freqs[seq_pos:seq_pos + T])
            if ck is not None:
                k = torch.cat((ck, k), dim=-2)
                v = torch.cat((cv, v), dim=-2)
                # q is already (B,H,1,D) — a single fresh query over the cache
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, self.cfg.dim)
            x = x + layer.attn.wo(y)
            x = x + layer.ffn(layer.ffn_norm(x))
            new_cache[(i, "q")] = q
            new_cache[(i, "k")] = k
            new_cache[(i, "v")] = v
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_cache


def build_model(cfg) -> Debbi:
    return Debbi(cfg)