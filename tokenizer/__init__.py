"""AICL + SP + Phrase tokenizer package."""

from aicl_tokenizer import (
    AICLTokenizer,
    generate_symbol_pool,
    ALPHABET_SOURCE,
    MARKERS,
    RESERVED,
    SYMBOL_RANGES,
)
from train_vocab import build_vocabulary

try:
    from sp_tokenizer import SPTokenizer, train_sp
except ImportError:
    SPTokenizer = None  # type: ignore
    train_sp = None  # type: ignore

try:
    from bpe_phrase_tokenizer import BPhraseTokenizer, train_phrase_vocab
except ImportError:
    BPhraseTokenizer = None  # type: ignore
    train_phrase_vocab = None  # type: ignore

__all__ = [
    "AICLTokenizer",
    "SPTokenizer",
    "BPhraseTokenizer",
    "train_sp",
    "train_phrase_vocab",
    "generate_symbol_pool",
    "build_vocabulary",
    "ALPHABET_SOURCE",
    "MARKERS",
    "RESERVED",
    "SYMBOL_RANGES",
]