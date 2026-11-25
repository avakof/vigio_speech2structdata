"""Reviewer reproducer: French quantity parsing battery (deterministic, no model)."""
from speechflow.extraction.numbers import parse_quantity
cases = ["deux virgule cinq", "2,5", "douze", "soixante-dix", "quatre-vingt-dix-sept", "soixante et onze",
         "mille deux cents", "1 000", "1 500", "une douzaine", "une dizaine", "un demi", "un et demi",
         "un kilo et demi", "deux kilos cinq", "deux ou trois", "entre deux et trois", "environ trois",
         "à peu près cinq", "zéro", "aucun", "une", "un", "trois quarts", "un quart", "0,75", "2.5",
         "vingt-et-un", "cent vingt", "deux cent cinquante", "quelques", "plusieurs", "pas mal"]
for c in cases:
    p = parse_quantity(c)
    print(f"{c!r:28} -> " + ("None" if p is None else f"value={p.value} max={p.maximum} qualifier={p.qualifier}"))
