"""BPE + Phrase hierarchical tokenizer — 90% target.

Level 1: SentencePiece BPE (4.59 c/t, 78.2%)
Level 2: Phrase merge over BPE token n-grams (2-5 pieces) -> 6.5-7.7 c/t, 84-87%

Training: corpus -> SP pieces -> count piece n-grams -> top N phrases
Encoding: text -> SP pieces -> greedy longest phrase merge -> ids
Decoding: ids -> phrase -> pieces -> SP decode -> text

Vocab: SP 6000 + phrases 4000 = 10000 total
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import sentencepiece as spm
try:
    from .sp_tokenizer import SPTokenizer
except ImportError:
    from sp_tokenizer import SPTokenizer


DELIM = "\x1f"


class BPhraseTokenizer:
    """BPE + phrase hierarchical tokenizer."""

    def __init__(self, sp_model_path: str, phrase_vocab_path: str = None, phrase_vocab: Dict = None):
        self.sp = SPTokenizer(sp_model_path)
        self.phrase_to_id: Dict[str, int] = {}
        self.id_to_phrase: Dict[int, str] = {}
        self.phrase_trie = {}
        self.num_phrases = 0

        if phrase_vocab_path and os.path.exists(phrase_vocab_path):
            with open(phrase_vocab_path) as f:
                data = json.load(f)
            self._load_phrases(data)
        elif phrase_vocab:
            self._load_phrases(phrase_vocab)

        # Vocab size = SP + phrases
        self._vocab_size = self.sp.vocab_size + self.num_phrases

    def _load_phrases(self, data: Dict):
        phrases = data.get("phrases", [])
        # phrases is list of {"phrase": "▁self\x1f.", "pieces": ["▁self","."], "freq": 634}
        for idx, entry in enumerate(phrases):
            phrase_key = entry["phrase"]  # delim-joined
            pid = self.sp.vocab_size + idx
            self.phrase_to_id[phrase_key] = pid
            self.id_to_phrase[pid] = phrase_key
            # insert into trie
            pieces = phrase_key.split(DELIM)
            node = self.phrase_trie
            for p in pieces:
                if p not in node:
                    node[p] = {}
                node = node[p]
            node["__id__"] = pid
            node["__len__"] = len(pieces)
        self.num_phrases = len(phrases)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        pieces = self.sp.encode_as_pieces(text)
        if not self.phrase_to_id:
            return self.sp.encode(text)

        # Greedy longest phrase merge over pieces
        ids = []
        i = 0
        n = len(pieces)
        while i < n:
            # try longest 5..2
            best_len = 0
            best_id = None
            node = self.phrase_trie
            # walk up to 5
            for j in range(i, min(i + 5, n)):
                p = pieces[j]
                if p not in node:
                    break
                node = node[p]
                if "__id__" in node:
                    best_len = j - i + 1
                    best_id = node["__id__"]
            if best_id is not None:
                ids.append(best_id)
                i += best_len
            else:
                # single BPE piece -> its id
                ids.append(self.sp.piece_to_id(pieces[i]))
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        pieces = []
        for pid in ids:
            if pid in self.id_to_phrase:
                phrase_key = self.id_to_phrase[pid]
                pieces.extend(phrase_key.split(DELIM))
            else:
                # SP id
                pieces.append(self.sp.id_to_piece(pid))
        return self.sp.sp.decode(pieces)

    def encode_as_pieces(self, text: str) -> List[str]:
        # for debugging
        pieces = self.sp.encode_as_pieces(text)
        # show merged
        ids = self.encode(text)
        merged = []
        for pid in ids:
            if pid in self.id_to_phrase:
                merged.append(f"[{self.id_to_phrase[pid].replace(DELIM,'+')}]")
            else:
                merged.append(self.sp.id_to_piece(pid))
        return merged


def train_phrase_vocab(
    corpus_path: str,
    sp_model_path: str,
    num_phrases: int = 4000,
    min_freq: int = 5,
    max_n: int = 5,
) -> Dict:
    """Learn phrase vocab over BPE pieces."""
    sp = SPTokenizer(sp_model_path)
    text = open(corpus_path).read()
    samples = [l for l in text.split("\n") if l.strip()]
    tokenized = [sp.encode_as_pieces(s) for s in samples]

    cnt = Counter()
    for pieces in tokenized:
        for n in range(2, max_n + 1):
            for i in range(len(pieces) - n + 1):
                key = DELIM.join(pieces[i : i + n])
                cnt[key] += 1

    # Filter by min_freq and sort by freq * (n-1) * avg_len
    # Use freq * (total_chars_saved) where total_chars = sum len(piece.replace("▁"," "))
    candidates = []
    for phrase, freq in cnt.items():
        if freq < min_freq:
            continue
        pieces = phrase.split(DELIM)
        # total chars in decoded form
        decoded = "".join(pieces).replace("▁", " ")
        # if starts with space, strip one (SP dummy)
        if decoded.startswith(" "):
            decoded = decoded[1:]
        chars_saved = len(decoded) - 1  # 1 token vs len chars
        # also consider n pieces saved: n-1 tokens saved
        score = freq * chars_saved
        candidates.append((phrase, freq, chars_saved, score))

    candidates.sort(key=lambda x: (-x[3], -x[1], x[0]))
    top = candidates[:num_phrases]

    phrases = []
    for phrase, freq, chars_saved, score in top:
        pieces = phrase.split(DELIM)
        phrases.append({"phrase": phrase, "pieces": pieces, "freq": freq, "chars_saved": chars_saved, "score": score})

    return {
        "sp_model": os.path.basename(sp_model_path),
        "num_phrases": len(phrases),
        "min_freq": min_freq,
        "max_n": max_n,
        "phrases": phrases,
    }


def train_and_save(corpus_path: str, sp_model_path: str, out_path: str, num_phrases: int = 4000):
    data = train_phrase_vocab(corpus_path, sp_model_path, num_phrases=num_phrases)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"saved {len(data['phrases'])} phrases to {out_path}")
    return data
