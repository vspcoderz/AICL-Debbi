"""
generate.py — sample text from a trained Debbi checkpoint.

  python model/generate.py --ckpt checkpoints/debbi-150m/last.pt \
      --vocab tokenizer/vocabularies/code-vocab.json \
      --id-map data/id_map.json --prompt "def quicksort(arr):"

Token ids map to AICL output codepoints (each id = one codepoint), then the
AICL decoder turns the symbol stream back into readable text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenizer"))

from config import Config  # noqa: E402
from transformer import Debbi  # noqa: E402
from aicl_tokenizer import AICLTokenizer  # noqa: E402

BOS_ID, EOS_ID = 1, 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--id-map", required=True)
    ap.add_argument("--prompt", default="")

    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    tok = AICLTokenizer.from_file(args.vocab)
    with open(args.id_map, encoding="utf-8") as fh:
        id_map = json.load(fh)
    id_to_chars = id_map["id_to_chars"]
    cfg.vocab_size = len(id_to_chars)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Debbi(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    if "cfg" in ck:  # checkpoint carries its own architecture
        saved = ck["cfg"]
        for k, v in saved.items():
            setattr(cfg, k, v)
        cfg.vocab_size = len(id_to_chars)
        model = Debbi(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step', '?')}), {model.n_params/1e6:.1f}M params")

    # prompt -> ids
    chars_to_id = id_map["chars_to_id"]
    prompt_ids = [chars_to_id.get(c, 3) for c in tok.encode(args.prompt)]
    ctx = torch.tensor([[BOS_ID] + prompt_ids], dtype=torch.long, device=device)

    out = model.generate(ctx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p, eos_id=EOS_ID)

    ids_out = out[0].tolist()
    chars = "".join(id_to_chars.get(str(i), "�") for i in ids_out[len(ctx[0]) - 1:])
    text = tok.decode(chars)
    print("\n" + "=" * 60)
    print(args.prompt + text)
    print("=" * 60)


if __name__ == "__main__":
    main()