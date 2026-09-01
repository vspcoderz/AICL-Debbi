"""
vs_bpe.py — the honest benchmark: how many tokens does AICL save vs BPE?

The research claim ("AICL beats BPE") is ONLY meaningful as a measured ratio
of token counts over the same text. This script:

  1. trains an AICL vocabulary on the corpus,
  2. (optionally) trains a SentencePiece BPE tokenizer on the same corpus,
  3. counts tokens on a held-out slice for BOTH and prints the ratio.

  AICL tokens   = number of output codepoints of the AICL encoder
                  (each codepoint is exactly one model vocabulary ID)
  BPE tokens    = SentencePiece tokens / pieces

Run:
  python benchmark/vs_bpe.py --corpus data/corpus.txt --size 2000

If SentencePiece is not installed, only AICL statistics are shown.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenizer"))

from aicl_tokenizer import AICLTokenizer  # noqa: E402
from train_vocab import build_vocabulary  # noqa: E402


def load_samples(path: str, cap_chars: int = 2_000_000):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if len(text) > cap_chars:
        text = text[:cap_chars]
    lines = [ln for ln in text.splitlines(keepends=True) if ln.strip()]
    n = max(1, len(lines) // 5)
    train, held = lines[:-n], lines[-n:]
    return "".join(train), "".join(held), train, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--vocab-out", default=None)
    args = ap.parse_args()

    train_text, held_text, train_lines, held_lines = load_samples(args.corpus)
    vocab = build_vocabulary(train_text, size=args.size, min_freq=args.min_freq)
    tok = AICLTokenizer(vocab)
    aicl_train = tok.encode(train_text)
    aicl_held = tok.encode(held_text)
    print(f"corpus: {len(train_text)+len(held_text)} chars | AICL vocab: {len(tok.entries)}")

    sp = None
    try:
        import sentencepiece as spm  # type: ignore
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "raw.txt")
            with open(raw, "w", encoding="utf-8") as fh:
                fh.writelines(train_lines[:2000])
            model = os.path.join(td, "bpe.model")
            spm.SentencePieceTrainer.train(
                input=raw, model_prefix=os.path.join(td, "bpe"),
                vocab_size=args.size, model_type="bpe",
                character_coverage=1.0, input_sentence_size=2000,
                max_sentencepiece_length=24,
            )
            sp = spm.SentencePieceProcessor(model_file=model)
    except ImportError:
        print("sentencepiece not installed — BPE column omitted (pip install sentencepiece).")

    if sp is not None:
        bpe_train = sum(len(sp.encode(s)) for s in train_lines)
        bpe_held = sum(len(sp.encode(s)) for s in held_lines)
        print(f"\n{'':>14}{'chars':>12}{'AICL toks':>12}{'BPE toks':>12}{'AICL/BPE':>10}")
        print(f"{'train':>14}{len(train_text):>12}{len(aicl_train):>12}{bpe_train:>12}{len(aicl_train)/max(bpe_train,1):>10.3f}")
        print(f"{'held-out':>14}{len(held_text):>12}{len(aicl_held):>12}{bpe_held:>12}{len(aicl_held)/max(bpe_held,1):>10.3f}")
        red = (1 - len(aicl_held) / max(bpe_held, 1)) * 100
        print(f"\nheld-out token reduction vs BPE: {red:+.2f}%  (positive = AICL wins)")
    else:
        print(f"\nAICL token count on train: {len(aicl_train)} "
              f"(char reduction {(1 - len(aicl_train) / max(len(train_text),1)) * 100:.2f}%)")
        print(f"AICL token count on held-out: {len(aicl_held)}")

    if args.vocab_out:
        import json
        with open(args.vocab_out, "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        print(f"AICL vocab saved to {args.vocab_out}")


if __name__ == "__main__":
    main()