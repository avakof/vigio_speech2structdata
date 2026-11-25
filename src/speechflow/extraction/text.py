"""Text folding and exact span location inside a segment's normalized text."""

from __future__ import annotations

import re
import unicodedata

_SPECIAL = {"…": "...", "’": "'", "‘": "'", "«": '"', "»": '"',
            "–": "-", "—": "-", "œ": "oe", "æ": "ae", " ": " ", " ": " "}


def _fold_char(ch: str) -> str:
    ch = _SPECIAL.get(ch, ch)
    out = []
    for c in unicodedata.normalize("NFKD", ch):
        if unicodedata.combining(c):
            continue
        out.append(c.casefold())
    return "".join(out)


def fold(text: str) -> str:
    """Accent-, case-, ellipsis- and whitespace-insensitive form for comparisons only."""
    return " ".join("".join(_fold_char(c) for c in text).split())


def _folded_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        for c in _fold_char(ch):
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            chars.append(c)
            index.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def term_spans(text: str, term: str):
    """Yield original character ranges of every whole occurrence of a spoken field."""
    folded, index = _folded_with_map(text)
    needle = fold(term)
    if needle:
        for match in re.finditer(r"(?<![\w-])" + re.escape(needle) + r"(?![\w-])", folded):
            yield index[match.start()], index[match.end() - 1] + 1


def locate(text: str, quote: str, taken: list[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    """Half-open code-point span of `quote` in `text`, tolerant to case/accents/ellipsis.

    Among several occurrences the first one not overlapping `taken` spans is chosen.
    The returned span always indexes the ORIGINAL text, so text[start:end] is an exact quote.
    """
    folded, index = _folded_with_map(text)
    needle = fold(quote)
    if not needle:
        return None
    taken = taken or []
    pos, first = folded.find(needle), None
    while pos != -1:
        span = (index[pos], index[pos + len(needle) - 1] + 1)
        if first is None:
            first = span
        if all(span[1] <= a or span[0] >= b for a, b in taken):
            return span
        pos = folded.find(needle, pos + 1)
    return first
