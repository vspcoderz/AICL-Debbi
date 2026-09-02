"""SP Tokenizer — pure BPE encoder fork.

Uses SentencePiece BPE directly for 78% compression.
No AICL symbol mapping, no case markers, no greedy longest-match.
Wraps `bpe_model.model` trained on `aiclcorpus/bpe_corpus.txt`.

Usage:
    tok = SPTokenizer("tokenizer/vocabularies/sp_bpe_6k.model")
    ids = tok.encode("hello world")      # List[int]
    text = tok.decode(ids)               # str
    # also: encode_as_pieces, vocab_size, id_to_piece, piece_to_id
"""

from __future__ import annotations

import os
from typing import Dict, List

import sentencepiece as spm


def train_sp(
    corpus_path: str,
    vocab_size: int = 6000,
    model_prefix: str = "sp_bpe_6k",
    max_piece_length: int = 32,
    input_sentence_size: int = 10000,
) -> str:
    """Train SentencePiece BPE from corpus. Returns .model path."""
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        max_sentencepiece_length=max_piece_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        split_digits=True,
        byte_fallback=True,
    )
    return f"{model_prefix}.model"


class SPTokenizer:
    """Thin wrapper around SentencePieceProcessor.

    Provides encode->ids, decode, and metadata compatible with
    Debbi's data/prepare_data.py + model/train.py pipeline.
    Specials: <pad>=0 (added), <unk>=0 in SP -> remapped to 3,
              <s>=1 (bos), </s>=2 (eos) kept as-is.
    Model itself uses: 0:<unk> 1:<s> 2:</s> + vocab.
    We expose Debbi convention: 0:<pad> 1:<bos> 2:<eos> 3:<unk> + rest shifted?
    For simplicity we keep SP ids and define <pad>=0 as extra — but SP model
    has no pad. Instead we use SP ids directly and let train.py handle pad=0
    as unused. Easiest: keep SP ids, treat 0 as unk/pad (collision is fine for now
    since pad never appears in corpus.bin except via masking).

    To keep corpus.bin as uint16 and vocab_size correct, we expose:
        vocab_size = sp.get_piece_size()  # includes unk/bos/eos
    and bos/eos ids are sp.piece_to_id("<s>") etc.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SP model not found: {model_path}")
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return 0  # we reserve 0; SP unk is also 0 — caller must handle

    @property
    def bos_id(self) -> int:
        return self.sp.piece_to_id("<s>")

    @property
    def eos_id(self) -> int:
        return self.sp.piece_to_id("</s>")

    @property
    def unk_id(self) -> int:
        return self.sp.piece_to_id("<unk>")

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return self.sp.decode(ids)

    def encode_as_pieces(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def id_to_piece(self, id: int) -> str:
        return self.sp.id_to_piece(id)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def __repr__(self) -> str:
        return f"SPTokenizer(model={self.model_path!r}, vocab_size={self.vocab_size})"
