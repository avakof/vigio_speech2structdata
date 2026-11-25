"""Check the inventory phrase around a model-selected evidence span."""

import re

from ..contracts import EvidenceSpan, TranscriptSegment
from .numbers import parse_quantity
from .text import fold
from .units import is_known_unit, normalize_unit

_NEG = re.compile(
    r"(?<![\w-])(n y a (plus|pas|aucun|aucune|rien)|ne reste (plus|pas|rien)|n en (reste|avons|a) (plus|pas|aucun)|"
    r"n (avons|ai|a|est) (plus|pas)|plus (de|du|des|d|un|une|aucun|aucune|rien)|pas (de|d|du|des)|aucun|aucune|"
    r"rien|sans|epuise\w*|rupture)(?![\w-])")


def negation_cue(joined: str) -> str | None:
    """'negation' if a negation cue is present; 'comparative' for 'plus de <number>'; else None."""
    found = None
    for m in _NEG.finditer(joined):
        if m.group(0).startswith("plus d"):
            tail = joined[m.end():].split()
            if any((p := parse_quantity(" ".join(tail[:n]))) is not None and p.value is not None
                   for n in (3, 2, 1) if len(tail) >= n):
                found = found or "comparative"
                continue
        return "negation"
    return found

# A decimal separator belongs to its number. Coordinating "et" also occurs
# inside French numerals, ranges and product names, so check both clauses.
_BOUNDARY = re.compile(r"(?:[!?;…]|(?<!\d)[.,]|[.,](?!\d)|\b(?:mais|puis|ensuite|et)\b)", re.IGNORECASE)
_PREDICATE = re.compile(r"^(?:il (?:ne |en )?(?:reste|y a)|nous (?:avons|n avons)|on (?:a|compte)|j ai|je compte)\b")
_NEGATED_ITEM = re.compile(r"\b(?:(?:pas|plus) (?:de|du|des|d)|aucune?) +\w+")
_TRAILING_APPROX = r"(?:environ|a peu pres|approximativement)"


def _words(text: str) -> str:
    return fold(text).replace("'", " ")


def _term(text: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(_words(text)) + r"(?![\w-])")


def _new_item(left, right):
    left = _words(left)
    right = _words(right)
    if _PREDICATE.match(right):
        return True
    if not left or left.split()[-1] in {"de", "du", "des", "d", "et", "entre", "a"}:
        return False
    words = right.split()[:16]
    starts_with_measure = any(
        is_known_unit(normalize_unit(word.strip(".,!?"))) and parse_quantity(" ".join(words[:i])) is not None
        for i, word in enumerate(words) if i)
    return starts_with_measure and bool(
        _NEGATED_ITEM.search(left) or any(is_known_unit(normalize_unit(word)) for word in left.split()))


def _clause_ranges(text):
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        if boundary.group().casefold() == "et":
            if not _new_item(text[start:boundary.start()], text[boundary.end():boundary.end() + 180]):
                continue
        if start < boundary.start():
            yield start, boundary.start()
        start = boundary.end()
    if start < len(text):
        yield start, len(text)


def narrow_count_quote(span: EvidenceSpan, terms) -> EvidenceSpan:
    """Retain one independently supported clause when the model quotes several items."""
    parts = list(_clause_ranges(span.quote))
    if len(parts) <= 1:
        return span
    patterns = [_term(term) for term in terms if term]
    supported = [(a, b) for a, b in parts
                 if all(pattern.search(_words(span.quote[a:b])) for pattern in patterns)]
    if len(supported) != 1:
        return span
    a, b = supported[0]
    text = span.quote[a:b]
    left = span.start + a + len(text) - len(text.lstrip())
    quote = text.strip()
    return EvidenceSpan(segment_id=span.segment_id, start=left, end=left + len(quote), quote=quote)


def _quantities_at_unit(clause, unit):
    for match in _term(unit).finditer(clause):
        before = clause[:match.start()].split()[-16:]
        # The longest understood phrase preserves approximate and range markers.
        for i in range(len(before)):
            parsed = parse_quantity(" ".join(before[i:]))
            if parsed is not None and parsed.value is not None:
                yield parsed
                break


def check_quantity_scope(
    evidence: list[EvidenceSpan], segments: dict[str, TranscriptSegment], labels: dict[str, TranscriptSegment],
    product: str | None, quantity_text: str, unit: str, operation: str, location: str | None = None,
) -> str | None:
    """A quote can copy real words but omit their qualifier or attribute them to another item."""
    if not evidence:
        return None
    first, last = evidence[0], evidence[-1]
    first_segment = segments[first.segment_id]
    local = [segments[e.segment_id].normalized_text for e in evidence]
    # Include the unquoted text around the evidence. A VAD split does not end a sentence.
    by_ordinal = {s.ordinal: s for s in labels.values()}
    prior = by_ordinal.get(first_segment.ordinal - 1)
    following = by_ordinal.get(segments[last.segment_id].ordinal + 1)
    prefix = prior.normalized_text[-256:] if prior else ""
    prefix += ";" if prefix and _new_item(prefix, local[0]) else " "
    source = prefix + local[0]
    for left, right in zip(local, local[1:]):
        source += (";" if _new_item(left, right) else " ") + right
    start = len(prefix) + first.start
    end = len(source) - (len(local[-1]) - last.end)
    if following:
        after = _words(following.normalized_text)
        if re.match(rf"^{_TRAILING_APPROX}\s*(?:[.!?;]|$)", after):
            source += " " + following.normalized_text[:64]
    claimed = parse_quantity(quantity_text)
    fields = [_term(term) for term in (quantity_text, unit, product, location) if term]
    problems = []
    for a, b in _clause_ranges(source):
        if a >= end or b <= start:
            continue
        clause = _words(source[a:b])
        if not all(field.search(clause) for field in fields):
            continue
        if operation == "count" and negation_cue(clause):
            problems.append("count_conflicts_with_source_context")
            continue
        if claimed and claimed.qualifier == "exact" and re.search(rf"\b{_TRAILING_APPROX}$", clause):
            problems.append("quantity_or_qualifier_conflicts_with_source_context")
            continue
        quantities = list(_quantities_at_unit(clause, unit))
        if quantities and claimed not in quantities:
            problems.append("quantity_or_qualifier_conflicts_with_source_context")
            continue
        return None
    return problems[0] if problems else "quantity_unit_and_product_not_in_one_source_clause"
