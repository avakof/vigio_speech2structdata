"""Verify explicit outer/inner measures and correction cues against source coordinates."""
from dataclasses import dataclass
import re

from .numbers import dec_str, parse_quantity
from .text import fold
from .units import PACKAGE_UNITS, is_known_unit, normalize_unit

_UNIT_WORD = re.compile(r"\b[^\W\d_]+\b")
_CORRECTION = re.compile(r"(?:\b(?:pardon|je corrige|rectification|ou plutot|plutot)|\bnon(?=\s*[,;:]|\s*$))[\s,;:….-]*$")
_ABSENCE = re.compile(
    r"\b(?:il ne reste|il n['’]y a|nous n['’]avons) (?:pas|plus) (?:de\s+|du\s+|d['’])(?P<noun>\w+)", re.I)


@dataclass
class Measure:
    start: int
    unit_start: int
    end: int
    quantity: str
    unit: str


def _measures(text):
    for word in _UNIT_WORD.finditer(text):
        unit = normalize_unit(word.group())
        if not is_known_unit(unit):
            continue
        # Look back only over a bounded phrase. Preserve original character coordinates.
        prefix_start = max(0, word.start() - 128)
        prefix = text[prefix_start:word.start()]
        starts = list(re.finditer(r"(?<![\w-])[\w\d]", prefix))[-14:]
        for start in starts:
            parsed = parse_quantity(prefix[start.start():].strip())
            if parsed and parsed.value is not None and parsed.qualifier == "exact":
                yield Measure(prefix_start + start.start(), word.start(), word.end(), dec_str(parsed.value), unit)
                break


def ambiguous_decimal_continuations(segments):
    """Locate a measured suffix after a malformed, unfinished spoken decimal."""
    result = {}
    segments = list(segments)
    for left, right in zip(segments, segments[1:]):
        if left.ordinal + 1 != right.ordinal:
            continue
        tail = fold(left.normalized_text).split()
        if len(tail) < 2 or tail[-1] != "virgules" or parse_quantity(tail[-2]) is None:
            continue
        first = next(_measures(right.normalized_text), None)
        if first and not right.normalized_text[:first.start].strip(" ,:…"):
            result[right.segment_id] = first
    return result


def complete_split_measure(item, evidence, segments):
    """Complete a partially copied number only from uninterrupted, model-cited speech."""
    if len(evidence) < 2 or item.operation not in ("count", "correction"):
        return item
    for left, right in zip(evidence, evidence[1:]):
        a, b = segments[left.segment_id], segments[right.segment_id]
        if (a.ordinal + 1 != b.ordinal or a.normalized_text[left.end:].strip()
                or b.normalized_text[:right.start].strip()):
            return item
    quoted = "\n".join(span.quote for span in evidence)
    measures = [m for m in _measures(quoted) if "\n" in quoted[m.start:m.end]]
    if len(measures) != 1:
        return item
    measure = measures[0]
    spoken_number = " ".join(quoted[measure.start:measure.unit_start].split())
    spoken_unit = quoted[measure.unit_start:measure.end]
    if (item.unit_spoken and normalize_unit(item.unit_spoken) != measure.unit
            or item.quantity_text and not re.search(r"(?<![\w-])" + re.escape(fold(item.quantity_text))
                                                     + r"(?![\w-])", fold(spoken_number))):
        return item
    return item.model_copy(update={"quantity_text": spoken_number, "unit_spoken": spoken_unit})


def source_relation_problems(labels, context_labels, observations):
    """Find source-backed omissions even if a generated reply drops a whole event."""
    offsets = {}
    source = ""
    target_start = None
    for label, segment in labels.items():
        if source:
            source += "\n"
        offsets[segment.segment_id] = len(source)
        if target_start is None and label not in context_labels:
            target_start = len(source)
        source += segment.normalized_text

    def cites(observation, measure):
        for span in observation.evidence:
            base = offsets.get(span.segment_id)
            if base is not None and base + span.start < measure.end and measure.start < base + span.end:
                return True
        return False

    def measured(observation, measure):
        parsed = parse_quantity(observation.quantity_text)
        return bool(parsed and parsed.value is not None and dec_str(parsed.value) == measure.quantity
                    and normalize_unit(observation.unit_spoken) == measure.unit and cites(observation, measure))

    for absent in _ABSENCE.finditer(source):
        if target_start is None or absent.end() <= target_start:
            continue
        if fold(absent["noun"]) == "temps":
            continue
        involved = [o for o in observations if any(
            span.segment_id in offsets and offsets[span.segment_id] + span.start < absent.end()
            and absent.start() < offsets[span.segment_id] + span.end for span in o.evidence)]
        if not any(o.operation == "negation" or o.status == "needs_review" for o in involved):
            yield f"Record the explicit inventory absence {absent.group()!r} as negation, without inventing zero.", involved

    measures = list(_measures(source))
    for measure in measures:
        if target_start is None or measure.end <= target_start:
            continue
        boundaries = [base for base in offsets.values() if measure.start < base < measure.end]
        if not boundaries:
            continue
        involved = [o for o in observations if cites(o, measure)]
        complete = any(measured(o, measure) and all(
            any(offsets.get(e.segment_id, len(source)) + e.start < edge for e in o.evidence)
            and any(offsets.get(e.segment_id, -len(source)) + e.end > edge for e in o.evidence)
            for edge in boundaries) for o in involved)
        if not complete:
            phrase = source[measure.start:measure.end].splitlines()
            yield (f"One measured quantity spans consecutive segments: {phrase!r}. Return one observation "
                   "and cite each segment with segment/quote and more_evidence; do not return the fragments as counts."), involved
    for first, second in zip(measures, measures[1:]):
        if target_start is None or second.end <= target_start:
            continue
        bridge = fold(source[first.end:second.start])
        packages = first.unit in PACKAGE_UNITS and bridge in {"de", "d'", "contenant", "avec"}
        correction = bool(_CORRECTION.search(bridge))
        if not (packages or correction):
            continue
        involved = [o for o in observations if cites(o, first) or cites(o, second)]
        if packages:
            complete = any(measured(o, first) and o.packaging
                           and o.packaging.per_package_quantity == second.quantity
                           and o.packaging.per_package_unit == second.unit and cites(o, second)
                           for o in involved)
            message = "Use the outer count and unit as primary and retain the explicit contents per package."
        else:
            before = [o for o in involved if measured(o, first) and o.operation in ("count", "correction")]
            revisions = [o for o in involved if o.operation == "correction" and cites(o, second)]
            complete = any(old is not new and (new.status == "needs_review" or
                           (new.correction_of and new.correction_of != new.observation_id))
                           for old in before for new in revisions)
            involved = [o for o in involved if o not in before or cites(o, second)]
            message = "Keep the earlier amount as a separate event and the following correction with its own quote and prior reference."
        if not complete:
            phrase = source[first.start:second.end]
            yield f"{message} Source: {phrase!r}", involved
