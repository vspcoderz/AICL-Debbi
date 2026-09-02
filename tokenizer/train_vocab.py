"""
train_vocab.py — learn an AICL vocabulary from a text corpus.

Candidate levels (per spec):
  1. phrase n-grams   (2-6 word sequences)   bonus 1.35
  2. word n-grams     (2-5 words)            treated as phrase level
  3. single words                            bonus 1.0
  4. character n-grams (2-6 letters, e.g. "tion") bonus 1.0
  5. individual letters via the alphabet map (1:1, no savings)

Scoring (per spec):  score = frequency x chars_saved x bonus
                      chars_saved = len(text) - 1   (symbol is 1 codepoint)

Take the top-N by score and assign each a unique symbol from the pool.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    from .aicl_tokenizer import (
        ALPHABET_SOURCE,
        AICLTokenizer,
        generate_symbol_pool,
        RESERVED,
    )
except ImportError:
    from aicl_tokenizer import (
        ALPHABET_SOURCE,
        AICLTokenizer,
        generate_symbol_pool,
        RESERVED,
    )

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LETTER_RUN_RE = re.compile(r"[a-z]+")
DIGIT_RUN_RE = re.compile(r"[0-9]+")
# Extended: also capture runs of common punctuation + digits
PUNCT_RUN_RE = re.compile(r"[._,\"'()`=:;\[\]{}<>|/+!@#&*?~\-]+")
IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]+")

BONUS = {"word": 1.0, "char": 1.0, "phrase": 1.35}
MAX_PHRASE_WORDS = 6
MAX_CHAR_GRAM = 6
MIN_CHAR_GRAM = 1


def _tokenize_words(text: str) -> List[Tuple[int, str]]:
    """Return (start_index, lowercase_word) pairs over the folded text."""
    return [(m.start(), m.group(0).lower()) for m in WORD_RE.finditer(text)]


def count_candidates(text: str, min_freq: int) -> Dict[str, Tuple[str, int, float]]:
    """Return {candidate_text: (level, freq, score)} for all candidate levels."""
    folded = text.lower()

    word_counts: Counter = Counter()
    ngram_counts: Counter = Counter()
    tokens = _tokenize_words(folded)
    for _, w in tokens:
        word_counts[w] += 1
    # word + phrase n-grams (2..6 consecutive words joined by a space)
    for size in range(2, MAX_PHRASE_WORDS + 1):
        for start in range(len(tokens) - size + 1):
            span = tokens[start:start + size]
            ngram_counts[" ".join(t for _, t in span)] += 1

    char_counts: Counter = Counter()
    seqs = LETTER_RUN_RE.findall(folded)
    for seq in seqs:
        n = len(seq)
        for L in range(MIN_CHAR_GRAM, min(MAX_CHAR_GRAM, n) + 1):
            for s in range(n - L + 1):
                char_counts[seq[s:s + L]] += 1

    # Also count punctuation/digit runs and identifiers for char-level entries
    for seq in PUNCT_RUN_RE.findall(folded):
        n = len(seq)
        for L in range(MIN_CHAR_GRAM, min(MAX_CHAR_GRAM, n) + 1):
            for s in range(n - L + 1):
                char_counts[seq[s:s + L]] += 1

    # Count single punctuation characters directly from raw text
    PUNCT_CHARS = set('_.\",:;()[]{}=+-*/<>!@#&|?~`\'')
    for ch in folded:
        if ch in PUNCT_CHARS:
            char_counts[ch] += 1

    # PUNCTUATION+SPACE and PUNCTUATION+PUNCTUATION pairs
    # These compress 2 chars → 1 symbol (saves 1 char each)
    PUNCT_SPACE = Counter()
    PUNCT_PUNCT = Counter()
    all_chars = list(folded)
    for i in range(len(all_chars) - 1):
        c1, c2 = all_chars[i], all_chars[i+1]
        if c1 in PUNCT_CHARS and c2 == ' ':
            PUNCT_SPACE[c1+c2] += 1
        elif c1 in PUNCT_CHARS and c2 in PUNCT_CHARS:
            PUNCT_PUNCT[c1+c2] += 1
    # Also " " + punct (space before punctuation)
    for i in range(len(all_chars) - 1):
        c1, c2 = all_chars[i], all_chars[i+1]
        if c1 == ' ' and c2 in PUNCT_CHARS:
            PUNCT_SPACE[c1+c2] += 1

    cands: Dict[str, Tuple[str, int, float]] = {}
    for g, f in PUNCT_SPACE.items():
        if f >= min_freq:
            cands[g] = ("punct_pair", f, float(f * 1 * BONUS["char"]))
    for g, f in PUNCT_PUNCT.items():
        if f >= min_freq:
            cands[g] = ("punct_pair", f, float(f * 1 * BONUS["char"]))
    for w, f in word_counts.items():
        if f >= min_freq and len(w) >= 2:
            cands[w] = ("word", f, float(f * (len(w) - 1) * BONUS["word"]))
    for g, f in ngram_counts.items():
        if f >= min_freq:
            cands[g] = ("phrase", f, float(f * (len(g) - 1) * BONUS["phrase"]))
    for g, f in char_counts.items():
        if f >= min_freq and g not in cands and len(g) > 1:
            cands[g] = ("char", f, float(f * (len(g) - 1) * BONUS["char"]))

    # Space-bound variants: ' word' and 'word ' absorb the neighbouring space,
    # so ONE symbol covers space+word (BPE's leading '\u2581' trick). Same
    # frequency, but len+1 raw chars saved -> usually outranks the bare form.
    for src, (level, f, _score) in list(cands.items()):
        if level in ("word", "phrase") and not src.startswith(" ") and not src.endswith(" "):
            lead = " " + src
            trail = src + " "
            cands[lead] = (level, f, float(f * (len(lead) - 1) * BONUS[level]))
            cands[trail] = (level, f, float(f * (len(trail) - 1) * BONUS[level]))
    return cands


def build_vocabulary(
    text: str,
    size: int = 2000,
    min_freq: int = 2,
    seed: int = 0,
    max_pool: int = 20000,
) -> Dict:
    """Learn a size-N vocabulary from `text` and return its JSON data dict."""
    import random

    cands = count_candidates(text, min_freq)
    ranked = sorted(cands.items(), key=lambda kv: (-kv[1][2], kv[0]))
    ranked = ranked[:size]

    pool = generate_symbol_pool(max(max_pool, size), skip=RESERVED)
    if len(pool) < size:
        raise RuntimeError(f"symbol pool exhausted (wanted {size}, have {len(pool)})")

    rng = random.Random(seed)
    symbols = pool[size:]  # keep the first `size` for... reserve: alphabet block
    # simplest, deterministic: alphabet gets its own slice from the pool head
    alphabet = {}
    for k, ch in enumerate(ALPHABET_SOURCE):
        alphabet[ch] = pool[k]

    entries = []
    used = set(pool[k] for k in range(len(ALPHABET_SOURCE)))
    next_sym = len(ALPHABET_SOURCE)
    for i, (src, (level, freq, score)) in enumerate(ranked):
        sym = pool[next_sym]
        next_sym += 1
        if next_sym > len(pool):
            raise RuntimeError("symbol pool exhausted")
        entries.append({"text": src, "symbol": sym, "level": level, "freq": int(freq), "score": round(score, 1)})

    return {
        "version": "0.1-debbi",
        "meta": {
            "size": len(entries),
            "min_freq": min_freq,
            "corpus_chars": len(text),
            "seed": seed,
        },
        "alphabet": alphabet,
        "entries": entries,
        "markers": {
            "space": "\u00b7",
            "newline": "\u240a",
            "title": "\u2191",
            "upper": "\u21e7",
            "num": "\u2317",
            "esc_open": "\u00ab",
            "esc_close": "\u00bb",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an AICL vocabulary")
    ap.add_argument("--input", required=True, help="corpus text file(s), comma separated")
    ap.add_argument("--output", required=True, help="output vocab JSON path")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-limit", type=int, default=0,
                    help="cap chars of the encode-report (0 = whole corpus)")
    args = ap.parse_args()

    chunks = []
    for path in args.input.split(","):
        path = path.strip()
        with open(path, "r", encoding="utf-8") as fh:
            chunks.append(fh.read())
    text = "\n".join(chunks)

    data = build_vocabulary(text, size=args.size, min_freq=args.min_freq, seed=args.seed)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    tok = AICLTokenizer(data)
    rep = text[: args.report_limit] if args.report_limit else text
    encoded = tok.encode(rep)
    print(f"vocab size: {len(tok.entries)}")
    print(f"corpus (report on {len(rep)} chars): {len(rep)} chars -> "
          f"AICL {len(encoded)} codepoints "
          f"({(1 - len(encoded) / max(len(rep), 1)) * 100:.2f}% character reduction)")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()