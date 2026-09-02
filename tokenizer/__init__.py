"""AICL / BPE / NORMAL tokenizer package — consistent naming.

- aicl_tokenizer / aicl      : AICL normal
- sp_tokenizer / bpe         : BPE normal (SP)
- bpe_phrase_tokenizer / aicl_fork : BPE fork with AICL phrase (85-90%)
- normal                     : whitespace baseline
"""

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
    from bpe import BPETokenizer  # alias
except ImportError:
    SPTokenizer = None  # type: ignore
    BPETokenizer = None  # type: ignore
    train_sp = None  # type: ignore

try:
    from bpe_phrase_tokenizer import BPhraseTokenizer, train_phrase_vocab
    from aicl_fork import AICLForkTokenizer  # alias
except ImportError:
    BPhraseTokenizer = None  # type: ignore
    AICLForkTokenizer = None  # type: ignore
    train_phrase_vocab = None  # type: ignore

try:
    from normal import NormalTokenizer
except ImportError:
    NormalTokenizer = None  # type: ignore

__all__ = [
    "AICLTokenizer",
    "SPTokenizer",
    "BPETokenizer",
    "BPhraseTokenizer",
    "AICLForkTokenizer",
    "NormalTokenizer",
    "train_sp",
    "train_phrase_vocab",
    "generate_symbol_pool",
    "build_vocabulary",
    "ALPHABET_SOURCE",
    "MARKERS",
    "RESERVED",
    "SYMBOL_RANGES",
]