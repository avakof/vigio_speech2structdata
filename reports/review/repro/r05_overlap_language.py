"""Reviewer reproducer: (a) audio-level re-emission flag fires on a single shared function word and
misses a partial-word re-emission; (b) extraction-level overlap-duplicate flag is skipped inside one window;
(c) language support relies solely on the model's self-reported `language`. No model is loaded."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import segs, obs, run, show
from speechflow.audio import _flag_reemissions

def flag(a, b, gap=0):
    s = segs(a, b, gap=gap)
    _flag_reemissions(s)
    return [(x.status, x.errors) for x in s]

print("(a1) single shared word 'de' across a 0-gap split:", flag("Il reste trois kilos de", "de farine en cave"))
print("(a2) unrelated words, shared 'lait' (legit two statements):", flag("Douze bouteilles de lait", "lait demi-écrémé six briques"))
print("(a3) partial-word re-emission 'bouteil' / 'bouteilles':", flag("Il reste douze bouteil", "bouteilles de lait"))
print("(a4) same text, gap 1 s (no shared input):", flag("Deux kilos de sucre", "Deux kilos de sucre", gap=16000))

s = segs("Deux kilos de sucre.", "Deux kilos de sucre.", gap=0)
show("(b) identical text in overlapping padded inputs, same window (segments not pre-flagged)",
     run(s, [obs("S1", "Deux kilos de sucre", product_spoken="sucre", quantity_text="Deux", quantity="2", unit_spoken="kilos"),
             obs("S2", "Deux kilos de sucre", ref="o2", product_spoken="sucre", quantity_text="Deux", quantity="2", unit_spoken="kilos")]))
first = run(s[:1], [obs("S1", "Deux kilos de sucre", product_spoken="sucre", quantity_text="Deux", quantity="2", unit_spoken="kilos")]).observations
show("(b') same pair, second segment in a LATER window",
     run(s, [obs("S2", "Deux kilos de sucre", product_spoken="sucre", quantity_text="Deux", quantity="2", unit_spoken="kilos")], already=first, context={"S1"}))

s = segs("There are 3 kilos of flour in the cellar.")
show("(c) English transcript, model self-reports fr",
     run(s, [obs("S1", "3 kilos of flour", product_spoken="flour", quantity_text="3", quantity="3", unit_spoken="kilos")], language="fr"))
s = segs("Ci sono 3 kili di farina.")
show("(c') Italian transcript, model self-reports fr",
     run(s, [obs("S1", "3 kili di farina", product_spoken="farina", quantity_text="3", quantity="3", unit_spoken="kili")], language="fr"))
