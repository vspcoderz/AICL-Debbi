"""BPE + AICL Hybrid Tokenizer.

Fork BPE for compression, add AICL for reversible encoding.
- BPE handles tokenization (78% compression)
- AICL handles symbol mapping (reversible, Unicode-safe)
- Case markers: ↑ title, ⇧ upper, ⇩ lower
- Whitespace: · space, ␊ newline, ⬞ tab
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm

# Markers
SPACE = "\u00b7"      # ·
NEWLINE = "\u240a"    # ␊
TAB = "\u2b1e"        # ⬞
CASE_TITLE = "\u2191" # ↑
CASE_UPPER = "\u21e7" # ⇧
CASE_LOWER = "\u21e9" # ⇩
ESC_OPEN = "\u00ab"   # «
ESC_CLOSE = "\u00bb"  # »

RESERVED = {SPACE, NEWLINE, TAB, CASE_TITLE, CASE_UPPER, CASE_LOWER, ESC_OPEN, ESC_CLOSE}

# Symbol ranges for AICL
SYMBOL_RANGES = [
    (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF),
    (0x2500, 0x257F), (0x2580, 0x259F), (0x25A0, 0x25FF),
    (0x2600, 0x26FF), (0x2700, 0x27BF), (0x27C0, 0x27EF),
    (0x2800, 0x28FF), (0x2900, 0x29FF), (0x2A00, 0x2AFF),
    (0x2B00, 0x2BFF), (0x2C00, 0x2CFF), (0x2D00, 0x2DFF),
    (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF),
    (0x3200, 0x32FF), (0x3300, 0x33FF), (0x4DC0, 0x4DFF),
    (0xA490, 0xA4CF), (0xFE30, 0xFE4F), (0xFE50, 0xFE6F),
]

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def generate_symbols(size: int) -> List[str]:
    pool = []
    skip = set(RESERVED)
    for start, end in SYMBOL_RANGES:
        for cp in range(start, end + 1):
            ch = chr(cp)
            if ch not in skip:
                pool.append(ch)
                if len(pool) >= size:
                    return pool
    return pool


def train_bpe(corpus_path: str, vocab_size: int = 6000, model_prefix: str = "bpe_model") -> str:
    spm.SentencePieceTrainer.train(
        input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0, max_sentencepiece_length=32,
        input_sentence_size=10000, shuffle_input_sentence=True,
        split_digits=True, byte_fallback=True,
    )
    return f"{model_prefix}.model"


def build_vocab(sp_model_path: str, vocab_size: int = 6000) -> Dict:
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    # Collect SP tokens
    tokens = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece not in ("<unk>", "<s>", "</s>"):
            tokens.append(piece)
    tokens.sort(key=lambda t: (-len(t), t))

    # Allocate symbols
    pool = generate_symbols(vocab_size + 100)
    alphabet = {ch: pool[k] for k, ch in enumerate(ALPHABET)}

    entries = []
    idx = len(ALPHABET)
    for piece in tokens:
        if idx >= len(pool):
            break
        entries.append({"text": piece, "symbol": pool[idx], "freq": 0})
        idx += 1

    return {
        "version": "1.0-bpe-aicl",
        "meta": {"size": len(entries), "vocab_size": vocab_size, "sp_model": os.path.basename(sp_model_path)},
        "alphabet": alphabet,
        "entries": entries,
        "markers": {"space": SPACE, "newline": NEWLINE, "tab": TAB,
                    "title": CASE_TITLE, "upper": CASE_UPPER, "lower": CASE_LOWER,
                    "esc_open": ESC_OPEN, "esc_close": ESC_CLOSE},
    }


class BPEAICLTokenizer:
    """BPE compression + AICL reversible encoding.

    Uses SP BPE for tokenization (78% compression).
    Maps tokens to AICL symbols (reversible).
    Adds case/whitespace markers.
    """

    def __init__(self, sp_model_path: str, vocab_data: Dict):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        # Build mappings
        self.token_to_symbol: Dict[str, str] = {}
        self.symbol_to_token: Dict[str, str] = {}
        for e in vocab_data["entries"]:
            self.token_to_symbol[e["text"]] = e["symbol"]
            self.symbol_to_token[e["symbol"]] = e["text"]

        self.alpha_to_symbol: Dict[str, str] = vocab_data.get("alphabet", {})
        self.symbol_to_alpha: Dict[str, str] = {v: k for k, v in self.alpha_to_symbol.items()}
        self.assigned = set(self.token_to_symbol.values()) | set(self.alpha_to_symbol.values()) | RESERVED

    def encode(self, text: str) -> str:
        """Encode text to AICL symbol stream using BPE."""
        if not text:
            return ""

        result = []
        lines = text.split("\n")

        for line_idx, line in enumerate(lines):
            if line_idx > 0:
                result.append(NEWLINE)

            if not line:
                continue

            # Split on tabs, handle each segment
            segments = line.split("\t")
            for seg_idx, segment in enumerate(segments):
                if seg_idx > 0:
                    result.append(TAB)

                if not segment:
                    continue

                # Count leading/trailing spaces
                leading = len(segment) - len(segment.lstrip(" "))
                trailing = len(segment) - len(segment.rstrip(" "))
                stripped = segment.strip(" ")

                if not stripped:
                    result.append(SPACE * len(segment))
                    continue

                # BPE encode
                lower = stripped.lower()
                tokens = self.sp.encode(lower, out_type=str)

                # Leading spaces
                if leading > 0:
                    result.append(SPACE * leading)

                # Encode each token
                real_tok_count = 0
                for tok_idx, tok in enumerate(tokens):
                    # ▁ prefix means "preceded by a space" (inter-word boundary)
                    has_space_prefix = tok.startswith("▁")
                    clean_tok = tok.lstrip("▁")

                    # Skip leading ▁ (SP dummy prefix, not a real space)
                    if has_space_prefix and tok_idx == 0 and not clean_tok:
                        continue

                    # Emit space marker for inter-word boundaries
                    if has_space_prefix and real_tok_count > 0:
                        result.append(SPACE)

                    start, end = self._get_char_range(tokens, tok_idx)

                    segment_text = stripped[start:end] if end <= len(stripped) else stripped[start:]
                    case = self._detect_case(segment_text)

                    tok_upper = sum(1 for c in clean_tok if c.isupper())
                    tok_alpha = sum(1 for c in clean_tok if c.isalpha())

                    if case == "lower" and tok_upper == 0:
                        result.append(self.token_to_symbol.get(tok, clean_tok))
                    elif case == "lower" and tok_upper == tok_alpha and tok_alpha > 1:
                        result.append(CASE_LOWER)
                        result.append(self.token_to_symbol.get(tok, clean_tok))
                    elif case == "upper":
                        result.append(CASE_UPPER)
                        result.append(self.token_to_symbol.get(tok, clean_tok))
                    elif case == "title" and tok_upper == 0:
                        result.append(CASE_TITLE)
                        result.append(self.token_to_symbol.get(tok, clean_tok))
                    elif clean_tok == segment_text:
                        result.append(self.token_to_symbol.get(tok, clean_tok))
                    else:
                        for ch in segment_text:
                            if ch == " ":
                                result.append(SPACE)
                            elif ch.isalpha():
                                low = ch.lower()
                                if ch.isupper():
                                    result.append(CASE_TITLE)
                                result.append(self.alpha_to_symbol.get(low, ch))
                            elif ch.isdigit():
                                result.append(ch)
                            elif ch in self.assigned:
                                result.append(ESC_OPEN + ch + ESC_CLOSE)
                            else:
                                result.append(ch)
                        continue

                    real_tok_count += 1

                # Trailing spaces
                if trailing > 0:
                    result.append(SPACE * trailing)

        return "".join(result)

    def decode(self, symbols: str) -> str:
        """Decode AICL symbol stream back to original text."""
        if not symbols:
            return ""

        # First pass: decode symbols to tokens
        tokens = []
        i = 0
        n = len(symbols)

        while i < n:
            ch = symbols[i]

            if ch == CASE_TITLE:
                i += 1
                if i < n:
                    decoded = self._decode_sym(symbols[i])
                    if decoded:
                        for j, c in enumerate(decoded):
                            if c.isalpha():
                                decoded = decoded[:j] + c.upper() + decoded[j+1:]
                                break
                        tokens.append(decoded)
                i += 1
                continue

            if ch == CASE_UPPER:
                i += 1
                if i < n:
                    decoded = self._decode_sym(symbols[i])
                    if decoded:
                        tokens.append(decoded.upper())
                i += 1
                continue

            if ch == CASE_LOWER:
                i += 1
                if i < n:
                    decoded = self._decode_sym(symbols[i])
                    if decoded:
                        tokens.append(decoded.lower())
                i += 1
                continue

            if ch == ESC_OPEN:
                j = symbols.index(ESC_CLOSE, i + 1) if ESC_CLOSE in symbols[i+1:] else n
                tokens.append(symbols[i+1:j])
                i = j + 1
                continue

            if ch == SPACE:
                tokens.append(" ")
                i += 1
                continue

            if ch == NEWLINE:
                tokens.append("\n")
                i += 1
                continue

            if ch == TAB:
                tokens.append("\t")
                i += 1
                continue

            decoded = self._decode_sym(ch)
            if decoded is not None:
                tokens.append(decoded)
            else:
                tokens.append(ch)
            i += 1

        # Second pass: join tokens — strip ▁ (spaces handled by our markers)
        raw = "".join(tokens)
        raw = raw.replace("▁", "")
        return raw

    def _decode_sym(self, sym: str) -> Optional[str]:
        if sym in self.symbol_to_token:
            # Preserve ▁ as space (SP convention)
            return self.symbol_to_token[sym]
        if sym in self.symbol_to_alpha:
            return self.symbol_to_alpha[sym]
        return None

    def _get_char_range(self, tokens: List[str], idx: int) -> Tuple[int, int]:
        start = 0
        for i in range(idx):
            t = tokens[i].lstrip("▁")
            start += len(t)
            if tokens[i].startswith("▁") and i > 0:
                start += 1
        tok = tokens[idx]
        clean = tok.lstrip("▁")
        end = start + len(clean)
        if tok.startswith("▁") and idx > 0:
            start += 1
            end += 1
        return start, end

    def _detect_case(self, text: str) -> str:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return "none"
        upper = sum(1 for c in letters if c.isupper())
        if upper == 0:
            return "lower"
        if upper == len(letters):
            return "upper"
        if upper == 1 and text[0].isupper():
            return "title"
        return "mixed"

    def token_count(self, text: str) -> int:
        return len(self.encode(text))


def train_and_build(corpus_path: str, output_path: str, vocab_size: int = 6000) -> Tuple[str, Dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "bpe")
        sp_model = train_bpe(corpus_path, vocab_size=vocab_size, model_prefix=prefix)
        vocab_data = build_vocab(sp_model, vocab_size=vocab_size)

        import shutil
        sp_dest = os.path.join(os.path.dirname(output_path), "bpe_model.model")
        shutil.copy2(sp_model, sp_dest)
        vocab_data["meta"]["sp_model"] = os.path.basename(sp_dest)

        with open(output_path, "w") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        return sp_dest, vocab_data
