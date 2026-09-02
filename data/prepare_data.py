"""
prepare_data.py — build data/corpus.bin + data/id_map.json from text files.

Supports two tokenizer backends:
  --tokenizer aicl : AICL reversible Unicode symbols (default legacy)
  --tokenizer bpe  : Pure SentencePiece BPE encoder (78% compression, recommended)

Steps (aicl):
  1. read text from files
  2. train/load AICL vocabulary
  3. encode each sample with AICLTokenizer
  4. map every output codepoint to vocab ID (BOS/EOS/UNK at 0-3)

Steps (bpe):
  1. read text
  2. load SP model (or train if missing)
  3. encode each sample with SPTokenizer.encode -> List[int]
  4. write ids directly with BOS/EOS boundaries, build id_map from SP vocab

Usage:
  python data/prepare_data.py --input file1.txt,file2.txt --out-dir data \
      --tokenizer bpe --sp-model tokenizer/vocabularies/sp_bpe_6k.model
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

BOS, EOS, UNK = 1, 2, 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file(s), comma separated")
    ap.add_argument("--out-dir", default="data", help=f"writes corpus.bin + id_map.json ({BOS=} {EOS=})")
    ap.add_argument("--tokenizer", choices=["aicl", "bpe", "bpe_phrase"], default="bpe_phrase",
                    help="tokenizer backend: aicl (legacy), bpe (SP), bpe_phrase (BPE+AICL phrase 90%, recommended)")
    ap.add_argument("--vocab", default=None, help="existing AICL vocab JSON (else train new) [aicl only]")
    ap.add_argument("--sp-model", default="tokenizer/vocabularies/sp_bpe_6k.model",
                    help="SentencePiece .model path [bpe* only]")
    ap.add_argument("--phrase-vocab", default="tokenizer/vocabularies/bpe_phrase_60k.json",
                    help="phrase vocab JSON [bpe_phrase only]")
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
    print(f"tokenizer: {args.tokenizer}")

    if args.tokenizer in ("bpe", "bpe_phrase"):
        if args.tokenizer == "bpe_phrase":
            from bpe_phrase_tokenizer import BPhraseTokenizer
            phrase_vocab = args.phrase_vocab
            if not os.path.isabs(phrase_vocab):
                repo_root = os.path.join(os.path.dirname(__file__), "..")
                cand = os.path.join(repo_root, phrase_vocab)
                if os.path.exists(cand):
                    phrase_vocab = cand
            sp_model = args.sp_model
            if not os.path.isabs(sp_model):
                repo_root = os.path.join(os.path.dirname(__file__), "..")
                cand = os.path.join(repo_root, sp_model)
                if os.path.exists(cand):
                    sp_model = cand
            tok = BPhraseTokenizer(sp_model, phrase_vocab)
            print(f"loaded BPE+Phrase model from {sp_model} + {phrase_vocab} (vocab_size={tok.vocab_size} = {tok.sp.vocab_size} BPE + {tok.num_phrases} phrases)")
            # Build id_map: SP ids + phrase ids
            vocab_size = tok.vocab_size
            chars_to_id = {"<pad>": 0, "<bos>": tok.bos_id, "<eos>": tok.eos_id, "<unk>": tok.unk_id}
            id_to_chars = {"0": "<pad>", str(tok.bos_id): "<bos>", str(tok.eos_id): "<eos>"}
            for i in range(tok.sp.vocab_size):
                piece = tok.sp.id_to_piece(i)
                if piece in ("<unk>", "<s>", "</s>"):
                    continue
                chars_to_id[piece] = i
                id_to_chars[str(i)] = piece
            for pid, phrase_key in tok.id_to_phrase.items():
                # phrase key is delim-joined pieces, decode for display
                phrase_text = phrase_key.replace("\x1f", "")
                chars_to_id[phrase_key] = pid
                id_to_chars[str(pid)] = phrase_key

            ids = []
            for s in samples:
                enc = tok.encode(s)
                ids.append(tok.bos_id)
                ids.extend(enc)
                ids.append(tok.eos_id)

            arr = np.array(ids, dtype=np.uint16 if vocab_size < 65535 else np.int32)
            os.makedirs(args.out_dir, exist_ok=True)
            arr.tofile(os.path.join(args.out_dir, "corpus.bin"))
            with open(os.path.join(args.out_dir, "id_map.json"), "w", encoding="utf-8") as fh:
                json.dump({"id_to_chars": id_to_chars, "chars_to_id": chars_to_id,
                           "vocab_size": vocab_size, "num_tokens": int(len(arr)),
                           "tokenizer": "bpe_phrase", "sp_model": os.path.basename(sp_model),
                           "phrase_vocab": os.path.basename(phrase_vocab), "num_phrases": tok.num_phrases}, fh, ensure_ascii=False, indent=2)

            total_ids = sum(len(tok.encode(s)) + 2 for s in samples)
            avg_tok_len = total / max(total_ids, 1)
            print(f"wrote {args.out_dir}/corpus.bin ({len(arr):,} ids) + id_map.json (vocab_size {vocab_size})")
            print(f"BPE+Phrase compression: {total_ids} ids for {total} chars -> {avg_tok_len:.2f} chars/token, {(1 - total_ids*1.0/max(total,1))*100:.2f}% token reduction vs chars")
            return

        from sp_tokenizer import SPTokenizer

        # Resolve sp-model relative to repo root
        sp_model = args.sp_model
        if not os.path.isabs(sp_model):
            # data/prepare_data.py -> repo root is ..
            repo_root = os.path.join(os.path.dirname(__file__), "..")
            cand = os.path.join(repo_root, sp_model)
            if os.path.exists(cand):
                sp_model = cand
        tok = SPTokenizer(sp_model)
        print(f"loaded SP BPE model from {sp_model} (vocab_size={tok.vocab_size})")

        # Build id_map from SP vocab + Debbi specials
        # SP ids: 0:<unk> 1:<s> 2:</s> 3+ vocab. Debbi convention 0:<pad> 1:<bos> 2:<eos> 3:<unk>
        # We map <pad> and <unk> both to 0 (shared) to keep SP ids unchanged.
        vocab_size = tok.vocab_size  # 6000 includes unk/bos/eos
        chars_to_id = {"<pad>": 0, "<bos>": tok.bos_id, "<eos>": tok.eos_id, "<unk>": tok.unk_id}
        # id_to_chars: 0 is <pad> (also <unk> alias), 1 bos, 2 eos
        id_to_chars = {"0": "<pad>", str(tok.bos_id): "<bos>", str(tok.eos_id): "<eos>"}
        # keep <unk> alias in chars_to_id only, don't overwrite id_to_chars[0]
        for i in range(tok.vocab_size):
            piece = tok.id_to_piece(i)
            if piece in ("<unk>", "<s>", "</s>"):
                continue
            chars_to_id[piece] = i
            id_to_chars[str(i)] = piece
        # Ensure unk maps to 0 but id 0 stays <pad> for readability
        # (unk piece itself is not needed in id_to_chars)

        # Encode samples to ids with BOS/EOS
        ids = []
        for s in samples:
            enc = tok.encode(s)
            ids.append(tok.bos_id)
            ids.extend(enc)
            ids.append(tok.eos_id)

        arr = np.array(ids, dtype=np.uint16 if vocab_size < 65535 else np.int32)
        os.makedirs(args.out_dir, exist_ok=True)
        arr.tofile(os.path.join(args.out_dir, "corpus.bin"))
        with open(os.path.join(args.out_dir, "id_map.json"), "w", encoding="utf-8") as fh:
            json.dump({"id_to_chars": id_to_chars, "chars_to_id": chars_to_id,
                       "vocab_size": vocab_size, "num_tokens": int(len(arr)),
                       "tokenizer": "bpe", "sp_model": os.path.basename(sp_model)}, fh, ensure_ascii=False, indent=2)

        # Report BPE compression
        total_ids = sum(len(tok.encode(s)) + 2 for s in samples)
        avg_tok_len = total / max(total_ids, 1)
        print(f"wrote {args.out_dir}/corpus.bin ({len(arr):,} ids) + id_map.json (vocab_size {vocab_size})")
        print(f"BPE compression: {total_ids} ids for {total} chars -> {avg_tok_len:.2f} chars/token, {(1 - total_ids*1.0/max(total,1))*100:.2f}% token reduction vs chars")
        return

    # --- AICL path (legacy) ---
    from aicl_tokenizer import AICLTokenizer
    from train_vocab import build_vocabulary

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