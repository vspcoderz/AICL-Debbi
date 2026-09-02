# Debbi — BPE + AICL Fork

**150M decoder, 90% token reduction.** BPE does compression, AICL adds phrase intelligence. Reversible: `decode(encode(x)) == x`.

![Benchmark](benchmark.png)

| Tokenizer | Vocab | Reduction | Chars/token |
|-----------|-------|-----------|-------------|
| **BPE** | 6k | 78.2% | 4.59 |
| **AICL Fork 4k** | 10k | **85.2%** | 6.78 |
| **AICL Fork 60k** | 66k | **90.3%** | 10.31 |
| AICL | 40k | 69.4% | 3.27 |
| NORMAL | — | 88.5% | 8.7 |

*BPE 6k trained on `bpe_corpus` 890k — Fork merges BPE token n-grams (AICL idea). Higher = fewer tokens.*

---

### Quick start

```bash
# BPE (78%)
python data/prepare_data.py --input data/code.txt --out-dir data --tokenizer bpe

# AICL Fork 85% (10k vocab, recommended)
python data/prepare_data.py --input data/code.txt --out-dir data --tokenizer bpe_phrase

# AICL Fork 90% (66k vocab)
python data/prepare_data.py --input data/code.txt --out-dir data --tokenizer bpe_phrase --phrase-vocab tokenizer/vocabularies/bpe_phrase_60k.json

# Train
python model/train.py --nano --max-steps 50   # smoke test
python model/train.py                          # 150M on T4
```

### Tokenizers

```python
from tokenizer.bpe import BPETokenizer          # BPE normal
from tokenizer.aicl_fork import AICLForkTokenizer  # BPE+AICL 85-90%
from tokenizer.aicl import AICLTokenizer        # AICL normal
from tokenizer.normal import NormalTokenizer    # baseline
```

### Layout

```
tokenizer/  bpe.py / aicl.py / aicl_fork.py / normal.py
data/       prepare_data.py -> corpus.bin + id_map.json
model/      config.py / train.py / generate.py
```

MIT — see `PLAN.md` for roadmap.
