"""Deterministic French quantity parsing (digits, decimal commas, number words, ranges).

Returns None whenever the text is not confidently understood: the caller must then
route the observation to review. Unknown quantities are never converted to zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .text import fold

SMALL = {"zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
         "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13,
         "quatorze": 14, "quinze": 15, "seize": 16}
TENS = {"vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}
COLLECTIVES = {"dizaine": (10, True), "douzaine": (12, False), "quinzaine": (15, True),
               "vingtaine": (20, True), "trentaine": (30, True), "quarantaine": (40, True),
               "cinquantaine": (50, True), "centaine": (100, True)}
APPROX_WORDS = ("environ", "a peu pres", "presque", "autour de", "dans les", "une petite", "pas loin de")
UNKNOWN_WORDS = {"quelques", "plusieurs", "beaucoup", "pas mal", "des", "un peu"}
FRACTIONS = {"demi": Decimal("0.5"), "demie": Decimal("0.5"), "quart": Decimal("0.25"),
             "quarts": Decimal("0.25"), "tiers": None}


@dataclass(frozen=True)
class ParsedQuantity:
    value: Decimal | None
    maximum: Decimal | None
    qualifier: str  # exact | approximate | range | unknown


def dec_str(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", fold(text).replace("'", " ")) if t]


def _parse_int_words(tokens: list[str]) -> int | None:
    """French cardinal words below one million, strict about ordering."""
    if not tokens:
        return None
    total = current = 0
    prev = None  # small | ten | hundred | thousand
    last_small_tok = None
    last_ten = None
    i = 0
    seen = False
    while i < len(tokens):
        t = tokens[i]
        if t == "et":
            if prev not in ("small", "ten") or i == len(tokens) - 1:
                return None
            i += 1
            continue
        if t == "quatre" and i + 1 < len(tokens) and tokens[i + 1] in ("vingt", "vingts"):
            if prev in ("small", "ten"):
                return None
            current += 80
            prev, i, seen, last_ten = "ten80", i + 2, True, "quatrevingt"
            continue
        if t in SMALL:
            v = SMALL[t]
            if prev == "small" and not (last_small_tok == "dix" and v in (7, 8, 9)):
                return None
            if prev in ("ten", "ten80"):
                if v == 0 or (v >= 10 and last_ten not in ("soixante", "quatrevingt")):
                    return None
                if v >= 10 and v > 16:
                    return None
            current += v
            prev, last_small_tok, seen = "small", t, True
        elif t in TENS:
            if prev in ("small", "ten", "ten80"):
                return None
            current += TENS[t]
            prev, seen, last_ten = "ten", True, t
        elif t in ("cent", "cents"):
            if prev in ("hundred",):
                return None
            current = (current or 1) * 100 if current < 100 else None  # type: ignore[operator]
            if current is None:
                return None
            prev, seen = "hundred", True
        elif t == "mille":
            if prev == "thousand":
                return None
            total += (current or 1) * 1000
            current, prev, seen = 0, "thousand", True
        else:
            return None
        i += 1
    return total + current if seen else None


_DIGITS = re.compile(r"^\d+(?:[.,]\d+)?$")
_DIGITS_SPACED = re.compile(r"^\d{1,3}(?: \d{3})+$")


def _parse_number(tokens: list[str], raw: str) -> Decimal | None:
    """Plain number: digits with decimal comma/point, or words, optionally with 'virgule' and fractions."""
    if not tokens:
        return None
    joined = " ".join(tokens)
    if len(tokens) == 1 and _DIGITS.match(tokens[0]):
        return Decimal(tokens[0].replace(",", "."))
    if _DIGITS_SPACED.match(joined):
        return Decimal(joined.replace(" ", ""))
    # "X et demi(e)" / "X et quart" / "X trois quarts"
    frac = Decimal(0)
    if tokens[-1] in FRACTIONS and FRACTIONS[tokens[-1]] is not None:
        body = tokens[:-1]
        if tokens[-1] in ("quarts",):
            if len(body) >= 1 and body[-1] in ("trois",):
                frac, body = Decimal("0.75"), body[:-1]
            else:
                return None
        else:
            frac = FRACTIONS[tokens[-1]]
            if body and body[-1] == "et":
                body = body[:-1]
            elif body and body[-1] in ("un", "une"):
                body = body[:-1]
                if body:
                    return None
            elif body:
                return None
        base = _parse_whole(body) if body else Decimal(0)
        return None if base is None else base + frac
    if "virgule" in tokens:
        k = tokens.index("virgule")
        whole, dec = tokens[:k], tokens[k + 1:]
        w = _parse_whole(whole)
        if w is None or not dec:
            return None
        if all(d in SMALL and SMALL[d] < 10 for d in dec) and len(dec) > 1 or (len(dec) == 1 and dec[0] in SMALL and SMALL[dec[0]] < 10):
            digits = "".join(str(SMALL[d]) for d in dec)
        elif len(dec) == 1 and _DIGITS.match(dec[0]) and "," not in dec[0] and "." not in dec[0]:
            digits = dec[0]
        else:
            n = _parse_int_words(dec)
            if n is None:
                return None
            digits = str(n)
        return Decimal(f"{w}.{digits}")
    return _parse_whole(tokens)


def _parse_whole(tokens: list[str]) -> Decimal | None:
    if len(tokens) == 1 and re.fullmatch(r"\d+", tokens[0]):
        return Decimal(tokens[0])
    n = _parse_int_words(tokens)
    return None if n is None else Decimal(n)


def parse_quantity(text: str | None) -> ParsedQuantity | None:
    if text is None or not text.strip():
        return None
    folded = fold(text).replace("'", " ")
    qualifier = "exact"
    for w in APPROX_WORDS:
        if re.search(rf"(^| ){w}( |$)", folded):
            qualifier = "approximate"
            folded = re.sub(rf"(^| ){w}( |$)", " ", folded).strip()
    folded = re.sub(r"(^| )(de|d)( |$)", " ", folded).strip() if qualifier == "approximate" else folded
    if folded in UNKNOWN_WORDS:
        return ParsedQuantity(None, None, "unknown")
    # collectives: une dizaine / une douzaine
    m = re.fullmatch(r"(?:une?|la)? ?(\w+)(?: de)?", folded)
    if m and m.group(1) in COLLECTIVES:
        value, approx = COLLECTIVES[m.group(1)]
        return ParsedQuantity(Decimal(value), None, "approximate" if approx or qualifier == "approximate" else qualifier)
    # ranges
    for pat in (r"^(?:entre )?(.+?) (?:et|a|ou) (.+)$", r"^de (.+?) a (.+)$"):
        m = re.match(pat, folded)
        if m and (folded.startswith(("entre ", "de ")) or " ou " in f" {folded} " or " a " in f" {folded} "):
            lo, hi = (_parse_number(_tokens(g), g) for g in m.groups())
            if lo is not None and hi is not None and lo < hi:
                return ParsedQuantity(lo, hi, "range")
    value = _parse_number(_tokens(folded), text)
    if value is None:
        return None
    return ParsedQuantity(value, None, qualifier)
