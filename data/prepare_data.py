"""
prepare_data.py — build data/corpus.bin + data/id_map.json from text files.

Steps:
  1. read text from files (or stdin)
  2. train an AICL vocabulary (or load an existing one)
  3. encode each sample with the AICL tokenizer
  4. map every output codepoint to a vocab ID (BOS/EOS/UNK at 0-3)
  5. write concatenated uint16 ids with <bos>...<eos> boundaries
  6. write id_map.json

Usage:
  python data/prepare_data.py --input file1.txt,file2.txt --out-dir data \
      --vocab tokenizer/vocabularies/code-vocab.json --vocab-size 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenizer"))

import numpy as np  # noqa: E402

from aicl_tokenizer import AICLTokenizer  # noqa: E402
from train_vocab import build_vocabulary  # noqa: E402

BOS, EOS, UNK = 1, 2, 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file(s), comma separated")
    ap.add_argument("--out-dir", default="data", help=f"writes corpus.bin + id_map.json ({BOS=} {EOS=})")
    ap.add_argument("--vocab", default=None, help="existing AICL vocab JSON (else train new)")
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    samples = []
    total = 0
    for path in args.input.split(","):
        path = path.strip()
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for ln in text.split("\n"):
            if ln.strip():
                samples.append(ln)
        total += len(text)
    print(f"read {len(samples)} samples, {total} chars")

    if args.vocab and os.path.exists(args.vocab):
        tok = AICLTokenizer.from_file(args.vocab)
        print(f"loaded AICL vocab from {args.vocab} ({len(tok.entries)} entries)")
    else:
        print("training a new AICL vocabulary...")
        vocab = build_vocabulary("\n".join(samples), size=args.vocab_size, min_freq=args.min_freq)
        tok = AICLTokenizer(vocab)
        os.makedirs(os.path.dirname(args.vocab) or ".", exist_ok=True)
        if args.vocab:
            tok.save(args.vocab)
            print(f"saved AICL vocab to {args.vocab}")

    # collect all output codepoints -> ids
    chars_seen = set(tok.markers.values())
    chars_seen.update("«" "»")  # guaranteed escape chars may appear
    streams = []
    for s in samples:
        enc = tok.encode(s)
        streams.append(enc)
        for c in enc:
            chars_seen.add(c)

    vocab_size = 4 + len(chars_seen)  # 0=pad,1=bos,2=eos,3=unk
    chars = sorted(chars_seen)
    chars_to_id = {"<pad>": 0, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}
    id_to_chars = {"0": "<pad>", "1": "<bos>", "2": "<eos>", "3": "<unk>"}
    for k, c in enumerate(chars, start=4):
        chars_to_id[c] = k
        id_to_chars[str(k)] = c
    unk_id = UNK

    # build the id stream
    ids = []
    for enc in streams:
        ids.append(BOS)
        for c in enc:
            ids.append(chars_to_id.get(c, unk_id))
        ids.append(EOS)

    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(args.out_dir, exist_ok=True)
    arr.tofile(os.path.join(args.out_dir, "corpus.bin"))
    with open(os.path.join(args.out_dir, "id_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"id_to_chars": id_to_chars, "chars_to_id": chars_to_id,
                   "vocab_size": vocab_size, "num_tokens": int(len(arr))}, fh, ensure_ascii=False, indent=2)

    print(f"wrote {args.out_dir}/corpus.bin ({len(arr):,} ids) + id_map.json (vocab_size {vocab_size})")
    print(f"AICL compression on this corpus: {(1 - len(''.join(streams)) / max(total,1)) * 100:.2f}% chars")


if __name__ == "__main__":
    main()