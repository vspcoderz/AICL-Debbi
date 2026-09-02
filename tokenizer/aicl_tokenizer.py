"""
AICL tokenizer — reversible, learned Unicode-symbol text compression.

AICL maps the most frequent words / phrases / character n-grams of a corpus to
single Unicode symbols, then emits text as a stream where each output
codepoint is exactly one vocabulary ID. The output is smaller than the input
(an n-char source becomes 1 symbol) while staying fully reversible.

Encoding rules
--------------
*  ' '  (space)   -> '·'
*  '\n' (newline) -> '␊'
*  longest vocabulary match, case-insensitive, with word-boundary checks:
       normal          -> <symbol>
       first letter up -> '↑' + <symbol>
       all up          -> '⇧' + <symbol>
*  a digit run at a word boundary -> '⌗' + digits (digits pass through)
*  a character that is itself a reserved marker or an assigned symbol
   is escaped as '«ch»'
*  letters covered by the alphabet map  -> their alphabet symbol
*  everything else passes through unchanged (reversible by identity)

Decoding reverses all of the above; in particular every output codepoint maps
back to exactly one input character.

The encoder/decoder are pure stdlib Python — no dependencies.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Symbols & reserved markers
# ----------------------------------------------------------------------------

# One codepoint per reserved role. These are excluded from the symbol pool so
# a literal occurrence in user text is unambiguous and reversible.
SPACE = "\u00b7"      # ·
NEWLINE = "\u240a"    # ␊  (control pictures)
CASE_TITLE = "\u2191" # ↑  apply title case to next unit
CASE_UPPER = "\u21e7" # ⇧  apply UPPER case to next unit
NUM = "\u2317"        # ⌗  number-run prefix
ESC_OPEN = "\u00ab"   # «
ESC_CLOSE = "\u00bb"  # »

MARKERS = {
    "space": SPACE,
    "newline": NEWLINE,
    "title": CASE_TITLE,
    "upper": CASE_UPPER,
    "num": NUM,
    "esc_open": ESC_OPEN,
    "esc_close": ESC_CLOSE,
}

RESERVED = set(MARKERS.values())

# Symbol pool ranges from the AICL spec (inclusive start/end).
# Original spec ranges + extended ranges for larger vocab experiments.
SYMBOL_RANGES: List[Tuple[int, int]] = [
    (0x2190, 0x21FF),  # Arrows
    (0x2200, 0x22FF),  # Mathematical Operators
    (0x2300, 0x23FF),  # Miscellaneous Technical
    (0x2400, 0x243F),  # Enclosed Alphanumerics
    (0x2440, 0x245F),  # Optical Character Recognition
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x27C0, 0x27EF),  # Misc Mathematical Symbols-A
    (0x27F0, 0x27FF),  # Supplemental Arrows-A
    (0x2800, 0x28FF),  # Braille Patterns
    (0x2900, 0x29FF),  # Supplemental Arrows-B
    (0x2A00, 0x2AFF),  # Supplemental Arrows-C
    (0x2B00, 0x2BFF),  # Misc Symbols and Arrows
    (0x2C00, 0x2CFF),  # Glagolitic
    (0x2D00, 0x2DFF),  # Ethiopic Extended
    (0x2E00, 0x2E1F),  # Supplemental Punctuation
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x2F00, 0x2FDF),  # Kangxi Radicals
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x3190, 0x319F),  # Kanbun
    (0x31A0, 0x31BF),  # Bopomofo Extended
    (0x31C0, 0x31EF),  # CJK Strokes
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3200, 0x32FF),  # Enclosed CJK Letters and Months
    (0x3300, 0x33FF),  # CJK Compatibility
    (0x4DC0, 0x4DFF),  # Yijing Hexagram Symbols
    (0xA490, 0xA4CF),  # Yi Radicals
    (0xFE30, 0xFE4F),  # CJK Compatibility Forms
    (0xFE50, 0xFE6F),  # Small Form Variants
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    (0x0370, 0x04FF),  # Greek and Coptic + Cyrillic
    (0x0530, 0x058F),  # Armenian
    (0x10A0, 0x10FF),  # Georgian
]


def generate_symbol_pool(size: int, skip: Iterable[str] = RESERVED) -> List[str]:
    """Yield up to `size` unique symbol codepoints from the spec ranges."""
    skipset = set(skip)
    pool: List[str] = []
    for start, end in SYMBOL_RANGES:
        for cp in range(start, end + 1):
            if pool.__len__() >= size:
                return pool
            cr = chr(cp)
            if cr in skipset or 0xD800 <= cp <= 0xDFFF:
                continue
            pool.append(cr)
    return pool


ALPHABET_SOURCE = "abcdefghijklmnopqrstuvwxyz"

# ----------------------------------------------------------------------------
# Vocabulary format
# ----------------------------------------------------------------------------
#
#   {
#     "version": "0.1-debbi",
#     "meta": {...},
#     "alphabet": {"a": "α", ...},                 letter -> symbol (1:1)
#     "entries": [                                  sorted by len(text) desc
#         {"text": "the server is", "symbol": "▸", "level": "phrase",
#          "freq": 50, "score": 1234.5},
#         ...
#     ],
#     "markers": {...}
#   }


class AICLTokenizer:
    """Pure-Python reversible AICL encoder/decoder."""

    def __init__(self, data: Dict):
        self.markers: Dict[str, str] = dict(data.get("markers", MARKERS))
        self.meta: Dict = data.get("meta", {})
        self.alphabet: Dict[str, str] = dict(data.get("alphabet", {}))
        raw_entries: List[Dict] = data.get("entries", [])
        # sorted longest-first so greedy longest-match is a simple scan
        self.entries: List[Dict] = sorted(raw_entries, key=lambda e: len(e["text"]), reverse=True)
        self.by_first: Dict[str, List[Dict]] = {}
        for e in self.entries:
            self.by_first.setdefault(e["text"][0], []).append(e)
        # symbol -> source for every assigned symbol
        self.symbol_to_text: Dict[str, str] = {}
        for e in self.entries:
            if e["symbol"] in self.symbol_to_text:
                raise ValueError(f"duplicate symbol {e['symbol']!r}")
            self.symbol_to_text[e["symbol"]] = e["text"]
        for ch, sym in self.alphabet.items():
            if sym in self.symbol_to_text:
                raise ValueError(f"duplicate symbol {sym!r}")
            self.symbol_to_text[sym] = ch
        self.symbol_values: set = set(self.symbol_to_text.keys())
        # reference-cheap lookups
        self._markers_values = set(self.markers.values())
        # fast longest-match index: per first char -> (lengths desc, text->entry)
        self._fast: Dict[str, Tuple[Tuple[int, ...], Dict[str, Dict]]] = {}
        for e in self.entries:
            tmap = self._fast.setdefault(e["text"][0], ((), {}))[1]
            tmap[e["text"]] = e
        for fc, (_lens, tmap) in self._fast.items():
            self._fast[fc] = (tuple(sorted({len(k) for k in tmap}, reverse=True)), tmap)

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "AICLTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    # -- units ----------------------------------------------------------

    def encode(self, text: str) -> str:
        """Encode `text` into the AICL symbol stream (all output chars are IDs)."""
        out: List[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]

            # Vocabulary match FIRST so multi-word/space-inclusive phrases can
            # actually be replaced by one symbol (a leading space is a legal
            # match start: ' the' -> <symbol>).
            matched = self._match_longest(text, i)
            if matched is not None:
                entry, length, case = matched
                if case == "upper":
                    out.append(self.markers["upper"])
                elif case == "title":
                    out.append(self.markers["title"])
                out.append(entry["symbol"])
                i += length
                continue

            if ch == " ":
                out.append(self.markers["space"]); i += 1; continue
            if ch == "\n":
                out.append(self.markers["newline"]); i += 1; continue

            if ch.isdigit() and (i == 0 or not text[i - 1].isalnum()):
                out.append(self.markers["num"])

            if ch in self._markers_values or ch in self.symbol_values:
                out.append(self.markers["esc_open"]); out.append(ch); out.append(self.markers["esc_close"])
                i += 1
                continue

            alph = self.alphabet.get(ch)
            if alph is not None:
                out.append(alph)
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _span_case(part: str) -> Optional[str]:
        """Casing of a matched span.

        Leading spaces (legal match starts: ' the' -> <symbol>) are ignored for
        the casing check; the actual word determines lower/title/upper.

        Return 'lower' | 'title' | 'upper', or None when the casing is mixed
        (e.g. 'QuickSort', 'getURL'). Mixed-case spans are never matched so
        identifiers round-trip exactly instead of being mangled by ↑/⇧.
        """
        core = part.lstrip(" ")
        if not core:
            return "lower"
        if core.islower():
            return "lower"
        if core.isupper():
            return "upper"
        if core[0].isupper() and core[1:].islower():
            return "title"
        return None

    def _match_longest(self, text: str, i: int) -> Optional[Tuple[Dict, int, str]]:
        """Longest vocabulary match at i, case-insensitive.

        Word/phrase entries require word boundaries on both sides — except the
        side that is *already* the entry's own leading/trailing space. This is
        what lets a single symbol absorb ' the' or 'the ' instead of wasting a
        token on the space. Char-level entries (e.g. 'tion', 'ing') may match
        anywhere inside a word. A match is only accepted when the span's casing
        is uniform (title/all-upper/all-lower), keeping identifiers exact.
        """
        rest = text[i:]
        if not rest:
            return None
        fast = self._fast.get(rest[0].lower())
        if not fast:
            return None
        lens, tmap = fast
        n = len(text)
        for L in lens:  # longest-first
            if L > len(rest):
                continue
            e = tmap.get(rest[:L].lower())
            if e is None:
                continue
            src = e["text"]
            if e["level"] != "char" and not src.startswith(" ") and i > 0 and text[i - 1].isalnum():
                continue  # never swallow a word/phrase inside a word
            after = i + L
            if e["level"] != "char" and not src.endswith(" ") and after < n and text[after].isalnum():
                continue  # next char must not be alnum (word boundary)
            case = self._span_case(text[i:after])
            if case is None:
                continue  # mixed casing ('QuickSort') — keep exact
            return e, L, case
        return None

    # -- decode ---------------------------------------------------------

    def decode(self, text: str) -> str:
        out: List[str] = []
        pending_case: Optional[str] = None
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == self.markers["space"]:
                out.append(" "); i += 1; continue
            if ch == self.markers["newline"]:
                out.append("\n"); i += 1; continue
            if ch == self.markers["num"]:
                i += 1; continue  # prefix marker; digits passed through
            if ch == self.markers["title"] or ch == self.markers["upper"]:
                pending_case = ch; i += 1; continue
            if ch == self.markers["esc_open"]:
                j = text.find(self.markers["esc_close"], i + 1)
                if j == -1:
                    # malformed: emit the raw open marker
                    out.append(ch); i += 1; continue
                lit = text[i + 1:j]
                out.append(self._apply_case(lit, pending_case))
                pending_case = None
                i = j + 1
                continue

            src = self.symbol_to_text.get(ch)
            if src is not None:
                out.append(self._apply_case(src, pending_case))
                pending_case = None
                i += 1
                continue
            # passthrough
            out.append(self._apply_case(ch, pending_case))
            pending_case = None
            i += 1
        return "".join(out)

    @staticmethod
    def _apply_case(unit: str, case: Optional[str]) -> str:
        if case is None or not unit:
            return unit
        if case == CASE_TITLE:
            lead = unit[: len(unit) - len(unit.lstrip(" "))]
            core = unit[len(lead):]
            if core:
                core = core[0].upper() + core[1:]
            return lead + core
        if case == CASE_UPPER:
            return unit.upper()
        return unit

    # -- counting -------------------------------------------------------

    def token_count(self, text: str) -> int:
        """Each output codepoint is exactly one vocabulary ID."""
        return len(self.encode(text))

    def encode_units(self, text: str) -> List[str]:
        """Output codepoints (each maps to one ID in the model vocabulary)."""
        return list(self.encode(text))

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "version": "0.1-debbi",
            "meta": self.meta,
            "alphabet": dict(sorted(self.alphabet.items())),
            "entries": sorted(
                self.entries,
                key=lambda e: (len(e["text"]), e["text"]),
                reverse=True,
            ),
            "markers": dict(self.markers),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)