# PLAN — Debbi MVP (aicl/Debbi)

Goal: honestly answer whether the AICL tokenizer (rare-Unicode-symbol
compression) beats BPE for a from-scratch decoder-only LM on code/agentic
tasks, at matched parameter count.

## Milestones

### M0 — Tokenizer (DONE)
- [x] Reversible AICL encoder/decoder in pure Python
  `decode(encode(x)) == x`, tested (case, numbers, escapes, symbols, random)
- [x] Cost-aware vocab learning (`freq × chars_saved × bonus`)
- [x] camelCase-safe matching (mixed-case spans are never swallowed)

### M1 — Honest compression benchmark (NEXT)
- [ ] `benchmark/vs_bpe.py` on a real code corpus (The Stack python slice)
      AICL tokens vs SentencePiece BPE tokens, train + held-out
- [ ] Expected outcome that *either way* informs M2:
      AICL wins  -> build bigger; AICL loses -> iterate tokenizer
      (rare Unicode under stock BPE is the likely culprit; for Debbi's own
      vocab every symbol is exactly 1 token, so it cannot lose there)

### M2 — Debbi-150M trained & evaluated
- [ ] End-to-end training on a few million code tokens (free T4)
- [ ] Perplexity + HumanEval/MBPP pass@1 vs a BPE-tokenized baseline at
      matched config
- [ ] Generation sanity samples in the notebook

### M3 — Scale
- [ ] 500M on the full data mix; then 1B per original spec (weeks of free T4)
- [ ] GGUF export for llama.cpp

## Guardrails
- Every reported metric must come from a run (no guessed numbers).
- Tokenizer roundtrip tests must stay green before any release.
- Stop-scaling checkpoints every ~12h Colab session; resume from `last.pt`.