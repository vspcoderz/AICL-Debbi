"""BPE-learned vocabulary + AICL symbol representation.

Uses SentencePiece BPE to learn vocabulary, AICL's greedy longest-match
for encoding with full whitespace control.

Architecture:
  Training:  corpus → SP BPE → vocabulary tokens → AICL symbol mapping
  Encoding:  text → case-aware greedy longest-match using BPE vocabulary
  Decoding:  AICL symbol stream → token lookup → concatenate → text
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import sentencepiece as spm

SPACE_MARKER = "\u00b7"
NEWLINE_MARKER = "\u240a"
CASE_TITLE = "\u2191"
CASE_UPPER = "\u21e7"
CASE_LOWER = "\u21e9"
NUM_PREFIX = "\u2317"
ESC_OPEN = "\u00ab"
ESC_CLOSE = "\u00bb"
TAB_MARKER = "\u2b1e"

RESERVED = {SPACE_MARKER, NEWLINE_MARKER, CASE_TITLE, CASE_UPPER, CASE_LOWER, NUM_PREFIX, ESC_OPEN, ESC_CLOSE, TAB_MARKER}

SYMBOL_RANGES = [
    (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF),
    (0x2400, 0x243F), (0x2440, 0x245F), (0x2500, 0x257F),
    (0x2580, 0x259F), (0x25A0, 0x25FF), (0x2600, 0x26FF),
    (0x2700, 0x27BF), (0x27C0, 0x27EF), (0x27F0, 0x27FF),
    (0x2800, 0x28FF), (0x2900, 0x29FF), (0x2A00, 0x2AFF),
    (0x2B00, 0x2BFF), (0x2C00, 0x2CFF), (0x2D00, 0x2DFF),
    (0x2E00, 0x2E1F), (0x2E80, 0x2EFF), (0x2F00, 0x2FDF),
    (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF),
    (0x3100, 0x312F), (0x3130, 0x318F), (0x3190, 0x319F),
    (0x31A0, 0x31BF), (0x31C0, 0x31EF), (0x31F0, 0x31FF),
    (0x3200, 0x32FF), (0x3300, 0x33FF), (0x4DC0, 0x4DFF),
    (0xA490, 0xA4CF), (0xFE30, 0xFE4F), (0xFE50, 0xFE6F),
    (0x1D400, 0x1D7FF), (0x0370, 0x04FF), (0x0530, 0x058F),
    (0x10A0, 0x10FF),
]

ALPHABET_SOURCE = "abcdefghijklmnopqrstuvwxyz"


def generate_symbol_pool(size: int, skip: Iterable[str] = RESERVED) -> List[str]:
    skipset = set(skip)
    pool = []
    for start, end in SYMBOL_RANGES:
        for cp in range(start, end + 1):
            ch = chr(cp)
            if ch not in skipset:
                pool.append(ch)
                if len(pool) >= size:
                    return pool
    return pool


def train_bpe(corpus_path, vocab_size=6000, model_prefix="bpe_model",
              max_piece_length=32, input_sentence_size=10000):
    spm.SentencePieceTrainer.train(
        input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0,
        max_sentencepiece_length=max_piece_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True, split_digits=True, byte_fallback=True,
    )
    return f"{model_prefix}.model"


def build_bpe_aicl_vocab(sp_model_path, vocab_size=6000, seed=0):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    sp_tokens = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece in ("<unk>", "<s>", "</s>"):
            continue
        sp_tokens.append(piece)

    sp_tokens.sort(key=lambda t: (-len(t), t))

    pool = generate_symbol_pool(max(vocab_size + 100, len(sp_tokens) + 100))
    if len(pool) < vocab_size:
        raise RuntimeError(f"Symbol pool exhausted: {len(pool)} < {vocab_size}")

    alphabet = {ch: pool[k] for k, ch in enumerate(ALPHABET_SOURCE)}

    entries = []
    symbol_idx = len(ALPHABET_SOURCE)

    for piece in sp_tokens:
        if symbol_idx >= len(pool):
            break
        sym = pool[symbol_idx]
        symbol_idx += 1
        level = "bpe_space" if piece.startswith("▁") else ("bpe_char" if len(piece) == 1 else "bpe")
        entries.append({"text": piece, "symbol": sym, "level": level, "freq": 0, "score": len(piece)})

    return {
        "version": "0.2-bpe-aicl",
        "meta": {"size": len(entries), "vocab_size": vocab_size, "sp_model": os.path.basename(sp_model_path), "learner": "bpe"},
        "alphabet": alphabet,
        "entries": entries,
        "markers": {"space": SPACE_MARKER, "newline": NEWLINE_MARKER, "title": CASE_TITLE,
                    "upper": CASE_UPPER, "num": NUM_PREFIX, "esc_open": ESC_OPEN, "esc_close": ESC_CLOSE, "tab": TAB_MARKER},
    }


class TrieNode:
    __slots__ = ('children', 'token', 'depth')

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.token: Optional[str] = None
        self.depth: int = 0


class BPEAICLEncoder:
    """BPE vocabulary with case-aware greedy longest-match encoding."""

    def __init__(self, sp_model_path: str, vocab_data: Dict):
        self.token_to_symbol: Dict[str, str] = {}
        self.symbol_to_token: Dict[str, str] = {}
        for entry in vocab_data["entries"]:
            self.token_to_symbol[entry["text"]] = entry["symbol"]
            self.symbol_to_token[entry["symbol"]] = entry["text"]

        self.alpha_to_symbol: Dict[str, str] = vocab_data.get("alphabet", {})
        self.symbol_to_alpha: Dict[str, str] = {v: k for k, v in self.alpha_to_symbol.items()}

        self.trie = TrieNode()
        for token in self.token_to_symbol:
            self._trie_insert(token)

        self.assigned_symbols = set(self.token_to_symbol.values()) | set(self.alpha_to_symbol.values()) | RESERVED

    def _trie_insert(self, token: str):
        node = self.trie
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.depth += 1
        node.token = token

    def _match_longest(self, text: str, pos: int) -> Optional[str]:
        """Exact-case longest match at position."""
        node = self.trie
        best = None
        i = pos
        while i < len(text):
            ch = text[i]
            if ch not in node.children:
                break
            node = node.children[ch]
            i += 1
            if node.token is not None:
                best = node.token
        return best

    def _match_longest_lower(self, lower_text: str, pos: int) -> Optional[str]:
        """Case-insensitive longest match — returns original token."""
        node = self.trie
        best = None
        i = pos
        while i < len(lower_text):
            ch = lower_text[i]
            if ch not in node.children:
                break
            node = node.children[ch]
            i += 1
            if node.token is not None:
                best = node.token
        return best

    def encode(self, text: str) -> str:
        if not text:
            return ""

        result = []
        i = 0
        n = len(text)
        lower_text = text.lower()

        while i < n:
            ch = text[i]

            if ch == "\n":
                result.append(NEWLINE_MARKER)
                i += 1
                continue

            if ch == "\t":
                result.append(TAB_MARKER)
                i += 1
                continue

            if ch == " ":
                count = 0
                while i < n and text[i] == " ":
                    count += 1
                    i += 1
                result.append(SPACE_MARKER * count)
                continue

            # Try exact-case match first
            token = self._match_longest(text, i)
            if token and len(token) > 1:
                result.append(self.token_to_symbol[token])
                i += len(token)
                continue

            # Try case-insensitive match
            token = self._match_longest_lower(lower_text, i)
            if token and len(token) > 1:
                segment = text[i:i+len(token)]

                # If token exactly matches text (including case), use directly
                if token == segment:
                    result.append(self.token_to_symbol[token])
                    i += len(token)
                    continue

                # Determine case type from text segment
                upper_count = sum(1 for c in segment if c.isupper())
                alpha_count = sum(1 for c in segment if c.isalpha())

                if upper_count == 0:
                    # Text is all lowercase
                    token_upper = sum(1 for c in token if c.isupper())
                    if token_upper == 0:
                        # Token is also lowercase — use directly
                        result.append(self.token_to_symbol[token])
                        i += len(token)
                        continue
                    elif token_upper == len(token) and alpha_count > 1:
                        # Token is all uppercase — use CASE_LOWER
                        result.append(CASE_LOWER)
                        result.append(self.token_to_symbol[token])
                        i += len(token)
                        continue
                    # Mixed-case token — skip, fall through to char encoding
                elif upper_count == alpha_count:
                    # Text is all uppercase
                    result.append(CASE_UPPER)
                    result.append(self.token_to_symbol[token])
                    i += len(token)
                    continue
                elif upper_count == 1 and segment[0].isupper() and alpha_count > 1:
                    # Title case — token must be all lowercase for CASE_TITLE to work
                    if sum(1 for c in token if c.isupper()) == 0:
                        result.append(CASE_TITLE)
                        result.append(self.token_to_symbol[token])
                        i += len(token)
                        continue
                # Mixed case — fall through to char encoding

            # Single character handling
            if ch.isalpha():
                lower_ch = ch.lower()
                if ch.isupper():
                    result.append(CASE_TITLE)
                if lower_ch in self.alpha_to_symbol:
                    result.append(self.alpha_to_symbol[lower_ch])
                else:
                    result.append(ch)
            elif ch.isdigit():
                result.append(ch)
            elif ch in self.assigned_symbols:
                result.append(ESC_OPEN)
                result.append(ch)
                result.append(ESC_CLOSE)
            else:
                result.append(ch)

            i += 1

        return "".join(result)

    def decode(self, symbols: str) -> str:
        if not symbols:
            return ""

        result = []
        i = 0
        n = len(symbols)

        while i < n:
            ch = symbols[i]

            if ch == CASE_TITLE:
                i += 1
                if i < n:
                    decoded = self._decode_symbol(symbols[i])
                    if decoded:
                        for j, c in enumerate(decoded):
                            if c.isalpha():
                                decoded = decoded[:j] + c.upper() + decoded[j+1:]
                                break
                        result.append(decoded)
                i += 1
                continue

            if ch == CASE_UPPER:
                i += 1
                if i < n:
                    decoded = self._decode_symbol(symbols[i])
                    if decoded:
                        result.append(decoded.upper())
                i += 1
                continue

            if ch == CASE_LOWER:
                i += 1
                if i < n:
                    decoded = self._decode_symbol(symbols[i])
                    if decoded:
                        result.append(decoded.lower())
                i += 1
                continue

            if ch == ESC_OPEN:
                j = symbols.index(ESC_CLOSE, i + 1) if ESC_CLOSE in symbols[i+1:] else n
                result.append(symbols[i+1:j])
                i = j + 1
                continue

            if ch == SPACE_MARKER:
                result.append(" ")
                i += 1
                continue

            if ch == NEWLINE_MARKER:
                result.append("\n")
                i += 1
                continue

            if ch == TAB_MARKER:
                result.append("\t")
                i += 1
                continue

            if ch == NUM_PREFIX:
                i += 1
                continue

            decoded = self._decode_symbol(ch)
            if decoded is not None:
                result.append(decoded)
            else:
                result.append(ch)

            i += 1

        return "".join(result)

    def _decode_symbol(self, sym: str) -> Optional[str]:
        if sym in self.symbol_to_token:
            return self.symbol_to_token[sym].lstrip("▁")
        if sym in self.symbol_to_alpha:
            return self.symbol_to_alpha[sym]
        return None

    def token_count(self, text: str) -> int:
        return len(self.encode(text))


def train_and_build(corpus_path, output_path, vocab_size=6000, seed=0):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "bpe")
        sp_model = train_bpe(corpus_path, vocab_size=vocab_size, model_prefix=model_prefix)
        vocab_data = build_bpe_aicl_vocab(sp_model, vocab_size=vocab_size, seed=seed)

        import shutil
        sp_dest = os.path.join(os.path.dirname(output_path), "bpe_model.model")
        shutil.copy2(sp_model, sp_dest)
        vocab_data["meta"]["sp_model"] = os.path.basename(sp_dest)

        with open(output_path, "w") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        return sp_dest, vocab_data
