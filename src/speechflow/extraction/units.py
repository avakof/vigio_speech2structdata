"""Spoken-unit normalization and deterministic conversion between compatible units only.

A counted package (carton, bouteille, sac...) is never a fixed mass or volume, so it has
no conversion entry. Unknown spoken units are kept (normalized) but never converted.
"""

from __future__ import annotations

from decimal import Decimal

from .text import fold

_ALIASES = {
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilogramme": "kg", "kilogrammes": "kg",
    "g": "g", "gramme": "g", "grammes": "g", "gr": "g",
    "l": "l", "litre": "l", "litres": "l",
    "cl": "cl", "centilitre": "cl", "centilitres": "cl",
    "ml": "ml", "millilitre": "ml", "millilitres": "ml",
    "bouteille": "bouteille", "bouteilles": "bouteille",
    "carton": "carton", "cartons": "carton",
    "boite": "boite", "boites": "boite", "boîte": "boite", "boîtes": "boite",
    "sac": "sac", "sacs": "sac", "sachet": "sachet", "sachets": "sachet",
    "paquet": "paquet", "paquets": "paquet",
    "piece": "piece", "pieces": "piece", "pièce": "piece", "pièces": "piece",
    "caisse": "caisse", "caisses": "caisse", "barquette": "barquette", "barquettes": "barquette",
    "pack": "pack", "packs": "pack", "pot": "pot", "pots": "pot", "bidon": "bidon", "bidons": "bidon",
    "canette": "canette", "canettes": "canette", "brique": "brique", "briques": "brique",
    "fut": "fut", "futs": "fut", "bocal": "bocal", "bocaux": "bocal", "plateau": "plateau", "plateaux": "plateau",
    "unite": "unite", "unites": "unite", "unité": "unite", "unités": "unite",
}
# Counted packaging containers that may carry an explicit per-package content.
PACKAGE_UNITS = {"carton", "caisse", "sac", "sachet", "paquet", "boite", "barquette", "pack", "pot",
                 "bidon", "plateau", "fut", "bocal", "brique"}
# dimension -> {unit: factor to the dimension's base unit}
_DIMENSIONS = {
    "mass": {"g": Decimal(1), "kg": Decimal(1000)},
    "volume": {"ml": Decimal(1), "cl": Decimal(10), "l": Decimal(1000)},
}


def normalize_unit(spoken: str | None) -> str | None:
    if spoken is None or not spoken.strip():
        return None
    words = fold(spoken).replace("'", " ")
    for prefix in ("de ", "d "):
        if words.startswith(prefix):
            words = words[len(prefix):]
    if words in _ALIASES:
        return _ALIASES[words]
    return words  # unknown unit stays as normalized text and is never converted


def dimension(unit: str | None) -> str | None:
    for name, table in _DIMENSIONS.items():
        if unit in table:
            return name
    return None


def compatible(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a == b or (dimension(a) is not None and dimension(a) == dimension(b))


def convert(value: Decimal, source: str, target: str) -> Decimal | None:
    """Convert only between units of the same physical dimension; otherwise None."""
    if source == target:
        return value
    dim = dimension(source)
    if dim is None or dim != dimension(target):
        return None
    table = _DIMENSIONS[dim]
    return value * table[source] / table[target]


def is_known_unit(unit: str | None) -> bool:
    """True when `unit` is one of the normalized units in the alias table (unknown ones stay reviewable)."""
    return unit is not None and unit in set(_ALIASES.values())
