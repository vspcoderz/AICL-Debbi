"""Roundtrip tests: decode(encode(x)) == x must hold for every input.

Run:  python -m unittest discover tokenizer/tests
   or: python tokenizer/tests/test_roundtrip.py
"""

import os
import random
import string
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aicl_tokenizer import AICLTokenizer  # noqa: E402
from train_vocab import build_vocabulary  # noqa: E402

SAMPLE_CORPUS = (
    "The server is not working because the database has an error. "
    "QuickSort is a fast sorting algorithm. Quicksort(left) + middle + Quicksort(right) "
    "return the sorted array. The model attends to every token in the context window.\n"
    "Indenting is essential in Python: def f(x):\n    return x * 2 + 1\n"
    "Numbers like 12345 and years like 2026 appear often. UPPERCASE WORDS and Title Case too.\n"
    "The quick brown fox jumps over the lazy dog 1234567890. Testing… αβγ emoji Ω≈ç√∫˜µ≤≥±\n"
)


def build_test_tokenizer(size=600):
    data = build_vocabulary(SAMPLE_CORPUS, size=size, min_freq=1)
    return AICLTokenizer(data)


ADVERSARIAL = [
    "",
    "plain ascii text with spaces and newlines\nand two lines",
    "The database has an error.",
    "  leading and trailing spaces  ",
    "TAB\tand other\twhitespace",
    "numbers: 1 22 333 4444 12345 and 2026-09-01",
    "UPPERCASE AND Mixed Case Words",
    "~!@#$%^&*()_+-=[]{}|;':\",./<>?`",
    "the the the the the",
    "a",
    "A",
    "1",
    "\n\n\n",
    "unicode: café Ω≈ç√∫˜µ≤≥± αβγ 中文 日本語",
    "escape-ish: « и «x» ⌗ ↑ ⇧ · ␊",
    "nested escapes: ««x»» 👨‍👩‍👧‍👦",
    "python def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n\tpivot = arr[len(arr) // 2]",
    "     indented\tcode\n    more",
]


class TestRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tok = build_test_tokenizer()

    def test_corpus_roundtrip(self):
        self.assertEqual(self.tok.decode(self.tok.encode(SAMPLE_CORPUS)), SAMPLE_CORPUS)

    def test_adversarial_roundtrip(self):
        for s in ADVERSARIAL:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, repr(s))

    def test_random_roundtrip(self):
        rng = random.Random(42)
        alphabet = string.ascii_letters + string.digits + " \n.,!?;:'\"-+*()[]{}_/\\@#$%^&"
        for _ in range(200):
            n = rng.randint(0, 120)
            s = "".join(rng.choice(alphabet) for _ in range(n))
            if rng.random() < 0.3:
                s = s + "".join(chr(rng.randint(0x2190, 0x27BF)) for _ in range(rng.randint(0, 4)))
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, repr(s[:40]))

    def test_encode_is_shorter_on_redundant_text(self):
        repeated = SAMPLE_CORPUS * 8
        enc = self.tok.encode(repeated)
        self.assertLess(len(enc), len(repeated))

    def test_token_count_equals_output_length(self):
        s = "the database has an error, says the server"
        self.assertEqual(self.tok.token_count(s), len(self.tok.encode(s)))

    def test_save_load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            import json

            json.dump(self.tok.to_dict(), fh, ensure_ascii=False)
            path = fh.name
        try:
            again = AICLTokenizer.from_file(path)
            s = SAMPLE_CORPUS + "\nMore: 2026 αβγ.."
            self.assertEqual(again.decode(again.encode(s)), s)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=1)