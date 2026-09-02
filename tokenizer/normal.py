"""NORMAL — baseline normal tokenizer (whitespace).

Simple baseline for benchmarking: splits on whitespace, no compression.
Useful as NORMAL reference vs BPE/AICL.

Usage:
    from tokenizer.normal import NormalTokenizer
    tok = NormalTokenizer()
    ids = tok.encode("hello world")  # [hash]
"""
from typing import List
import hashlib

class NormalTokenizer:
    """Trivial whitespace baseline. Not for training — for benchmarking NORMAL."""

    def __init__(self):
        self.vocab = {}
        self.inv = {}

    def encode(self, text: str) -> List[int]:
        # Simple: each whitespace-separated token -> hash -> id
        # For benchmarking chars/token, we just count whitespace tokens
        # This gives ~5-6 chars/token on code, ~0% reduction vs BPE's 78%
        # We implement as: split on whitespace, each piece = 1 token
        if not text:
            return []
        # Return hash-based ids for compatibility, but for benchmarking we only need len
        pieces = text.split()
        # Map to ids deterministically
        ids = []
        for p in pieces:
            h = int(hashlib.md5(p.encode()).hexdigest()[:8], 16) % 100000
            ids.append(h)
        # Also count newlines as separate tokens
        ids.extend([0] * text.count("\n"))
        return ids

    def decode(self, ids: List[int]) -> str:
        # Lossy - not reversible, only for benchmarking NORMAL baseline
        return f"<{len(ids)} tokens>"

    @property
    def vocab_size(self) -> int:
        return 100000
