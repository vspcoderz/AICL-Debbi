"""AICL — normal AICL tokenizer.

Re-export of aicl_tokenizer.AICLTokenizer.

Usage:
    from tokenizer.aicl import AICLTokenizer
    tok = AICLTokenizer.from_file("tokenizer/vocabularies/code-vocab.json")
"""
try:
    from .aicl_tokenizer import AICLTokenizer, generate_symbol_pool, ALPHABET_SOURCE, MARKERS, RESERVED, SYMBOL_RANGES
except ImportError:
    from aicl_tokenizer import AICLTokenizer, generate_symbol_pool, ALPHABET_SOURCE, MARKERS, RESERVED, SYMBOL_RANGES

__all__ = ["AICLTokenizer", "generate_symbol_pool", "ALPHABET_SOURCE", "MARKERS", "RESERVED", "SYMBOL_RANGES"]
