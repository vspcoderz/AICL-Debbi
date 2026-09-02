"""AICL Fork — BPE+AICL phrase hierarchical (BPE fork with AICL features).

Re-export of bpe_phrase_tokenizer.BPhraseTokenizer with AICL naming.
85.2% with 4k phrases (10k vocab), 90.3% with 60k phrases (66k vocab) on bpe_corpus.

This is the BPE fork that integrates AICL phrase/indent ideas:
- BPE base (sp_bpe_6k, 78.2%) + phrase merge over BPE token n-grams
- Gives AICL-style phrase compression with BPE merge-order

Usage:
    from tokenizer.aicl_fork import AICLForkTokenizer
    tok = AICLForkTokenizer("tokenizer/vocabularies/sp_bpe_6k.model",
                            "tokenizer/vocabularies/bpe_phrase_4k.json")
    # 4k = 85.2% (10k vocab, uint16), 60k = 90.3% (66k vocab, int32)
"""
try:
    from .bpe_phrase_tokenizer import BPhraseTokenizer as AICLForkTokenizer
    from .bpe_phrase_tokenizer import train_phrase_vocab
except ImportError:
    from bpe_phrase_tokenizer import BPhraseTokenizer as AICLForkTokenizer
    from bpe_phrase_tokenizer import train_phrase_vocab

__all__ = ["AICLForkTokenizer", "train_phrase_vocab"]
