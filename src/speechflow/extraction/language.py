"""Independent, dependency-free language evidence for transcript text.

Not a language identifier: a conservative lexical *veto*. It counts function words (articles,
prepositions, auxiliaries, copulas) that are distinctive per language. Content words (products,
units) carry no weight, so short French product strings such as "farine" or "pommes de terre"
stay inconclusive (None) rather than receiving a hard label. A non-French label is returned only
when other-language function words clearly outweigh French ones; the caller treats that as a reason
for review, never as a translation or a rejection of the transcript.

Applicability limits: only fr/en/it/es/de/pt are modelled; texts shorter than three alphabetic
tokens, or made of content words only, are inconclusive; a French sentence quoting foreign phrases
can be mislabelled; the lexicons are small and curated, not trained.
"""

from __future__ import annotations

import re

from .text import fold

_LEX = {
    "fr": "le la les un une des du de d l et est sont il ils elle elles y au aux avec dans pour par sur "
          "ne n qu c j s ce cette ces mon ma mes ton son sa ses nous vous reste restent rester faut "
          "deux trois quatre cinq huit neuf onze douze treize quinze seize vingt trente quarante cinquante "
          "soixante cent mille virgule demi demie environ encore aussi mais ou plus pas non puis",
    "en": "the of there are is was were and in with we have has had to for from left remaining some any "
          "this that these those it its at by or not no about around approximately please",
    "it": "ci sono di della delle dei degli del nel nella nei che per abbiamo ho ha hanno sul alla allo "
          "c e è rimasti rimaste rimangono ne non con un una uno il lo gli le la da su tra fra e anche",
    "es": "hay quedan queda los las del con el una unos unas sobre para por no y en de la le se su sus "
          "tenemos tengo estan esta estamos",
    "de": "es gibt und der die das noch sind von im ist nicht mit ein eine einen keine kein auf für "
          "wir haben hat zu den dem des am aus bei nach",
    "pt": "tem temos restam os as do da no na uma um com para dos das não nao em de a o e se por sobre",
}
# Function words that French shares with a language are counted for both; ambiguity never favours "other".
_TOKEN_LANGS: dict[str, set[str]] = {}
for _lang, _words in _LEX.items():
    for _w in _words.split():
        _TOKEN_LANGS.setdefault(fold(_w), set()).add(_lang)
LANGS = tuple(_LEX)

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def scores(text: str) -> tuple[dict[str, int], int]:
    tokens = _TOKEN.findall(fold(text).replace("'", " "))
    out = {lang: 0 for lang in LANGS}
    for tok in tokens:
        for lang in _TOKEN_LANGS.get(tok, ()):
            out[lang] += 1
    return out, len(tokens)


def assess(text: str) -> str | None:
    """'fr' only with positive French evidence, another code only when it clearly outweighs French,
    None when inconclusive (short strings, product names, ties)."""
    sc, n = scores(text)
    fr = sc["fr"]
    other, best = max(((lang, sc[lang]) for lang in LANGS if lang != "fr"), key=lambda p: p[1])
    if n < 3 and best < 2:
        return "fr" if fr >= 2 else None
    if best >= 1 and fr == 0 and n >= 3:
        return other
    if best >= 2 and best > fr:
        return other
    if fr >= 2 and fr > best:
        return "fr"
    return None
