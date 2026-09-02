"""BPE — pure SentencePiece BPE (NORMAL).

Re-export of sp_tokenizer.SPTokenizer with clearer name.
78.2% on bpe_corpus (6000 vocab, 4.59 c/t).

Usage:
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer("tokenizer/vocabularies/sp_bpe_6k.model")
"""
from sp_tokenizer import SPTokenizer as BPETokenizer
from sp_tokenizer import train_sp as train_bpe

__all__ = ["BPETokenizer", "train_bpe"]
