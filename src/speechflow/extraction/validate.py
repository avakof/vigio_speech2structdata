"""Independent, deterministic validation of an LLM reply against the transcript.

Schema-valid JSON is not evidence of correct meaning. Everything here is recomputed from the
transcript text: evidence spans (one or several consecutive segments), numbers, units,
location, negation, packaging totals, correction targets.

Outcomes per reply:
* `repairable` errors: the reply cannot be trusted structurally (unknown segment, quote not in
  the segment, spans out of order, duplicate labels). The caller may spend its single repair on them.
* per-observation `errors`: the observation is kept but marked needs_review.
* context-only statements are dropped (counted): they were already processed by the previous window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from ..contracts import EvidenceSpan, Packaging, TranscriptSegment, ValidatedObservation, stable_id
from .language import assess as assess_language
from .numbers import APPROX_WORDS, dec_str, parse_quantity
from .relations import complete_split_measure, source_relation_problems
from .scope import check_quantity_scope, narrow_count_quote, negation_cue
from .schema import LLMObservation, LLMOutput
from .text import fold, locate, term_spans
from .units import PACKAGE_UNITS, compatible, convert, dimension, is_known_unit, normalize_unit


@dataclass
class ReplyCheck:
    observations: list[ValidatedObservation] = field(default_factory=list)
    repairable: list[str] = field(default_factory=list)
    semantic_repair: list[str] = field(default_factory=list)
    unresolved_mentions: list[list[EvidenceSpan]] = field(default_factory=list)
    reply_errors: list[str] = field(default_factory=list)  # reviewable window-level notes
    skipped_duplicates: int = 0
    context_only_dropped: int = 0
    replaced: list[str] = field(default_factory=list)  # ids of earlier partial observations completed here
    replaced_notes: list[str] = field(default_factory=list)
    conversions: list[str] = field(default_factory=list)  # deterministic unit normalizations applied to corrections


def _words(text: str) -> str:
    return fold(text).replace("'", " ").replace("’", " ")


def _in(needle: str | None, haystack_folded: str) -> bool:
    """Whole-word containment: 'deux' must not match inside 'vingt-deux' or 'douze'."""
    if not needle or not needle.strip():
        return False
    pattern = r"(?<![\w-])" + re.escape(_words(needle)) + r"(?![\w-])"
    return re.search(pattern, haystack_folded) is not None


def _label_no(label: str) -> int | None:
    m = re.fullmatch(r"S(\d+)", label)
    return int(m.group(1)) if m else None


def _spans(ev: list[EvidenceSpan]) -> list[tuple[str, int, int]]:
    return [(e.segment_id, e.start, e.end) for e in ev]


def _overlaps(a: ValidatedObservation, b: ValidatedObservation) -> bool:
    return any(x[0] == y[0] and x[1] < y[2] and y[1] < x[2] for x in _spans(a.evidence) for y in _spans(b.evidence))


def _contained(inner: ValidatedObservation, outer: ValidatedObservation) -> bool:
    return all(any(x[0] == y[0] and y[1] <= x[1] and x[2] <= y[2] for y in _spans(outer.evidence))
               for x in _spans(inner.evidence))


def _same_values(a: ValidatedObservation, b: ValidatedObservation) -> bool:
    return (a.operation, a.quantity, a.unit, a.quantity_max, fold(a.product_spoken or "")) == \
           (b.operation, b.quantity, b.unit, b.quantity_max, fold(b.product_spoken or ""))


# ------------------------------------------------------------------ main entry
def check_reply(
    reply: LLMOutput,
    *,
    run_id: str,
    labels: dict[str, TranscriptSegment],
    context_labels: set[str],
    known: dict[str, ValidatedObservation],
    already: list[ValidatedObservation],
    segment_index: dict[str, TranscriptSegment] | None = None,
) -> ReplyCheck:
    """Validate `reply`. `labels` maps every prompt label (context + extract) to its segment."""
    out = ReplyCheck()
    segment_index = segment_index or {s.segment_id: s for s in labels.values()}
    refs = [o.ref for o in reply.observations]
    if len(set(refs)) != len(refs):
        out.repairable.append(f"duplicate ref labels: {refs}")
    taken: dict[str, list[tuple[int, int]]] = {}
    created: dict[str, ValidatedObservation] = {}
    positions: dict[str, tuple[int, int, int]] = {}
    tie: dict[tuple, int] = {}
    if reply.language != "fr":
        out.reply_errors.append(f"unsupported_language:{reply.language}")
    # Independent lexical evidence per segment: the model's own `language` claim is never trusted alone.
    seg_lang = {seg.segment_id: assess_language(seg.normalized_text) for seg in segment_index.values()}
    for lab, seg in labels.items():
        if lab in context_labels:
            continue
        found = seg_lang.get(seg.segment_id)
        if found not in (None, "fr"):
            out.reply_errors.append(f"language_evidence:{seg.segment_id}:{found}")
            if reply.language == "fr":
                out.reply_errors.append(f"language_disagreement:{seg.segment_id}:model=fr,lexical={found}")
        elif found == "fr" and reply.language != "fr":
            out.reply_errors.append(f"language_disagreement:{seg.segment_id}:model={reply.language},lexical=fr")

    for index, item in enumerate(reply.observations):
        cited = [(item.segment, item.quote)] + [(m.segment, m.quote) for m in item.more_evidence]
        bad = [lab for lab, _ in cited if lab not in labels]
        if bad:
            out.repairable.append(f"{item.ref}: unknown segment label(s) {bad}")
            continue
        numbers = [_label_no(lab) for lab, _ in cited]
        if len(set(lab for lab, _ in cited)) != len(cited) or numbers != sorted(numbers):
            out.repairable.append(f"{item.ref}: evidence parts must cite distinct segments in transcript order")
            continue
        if all(lab in context_labels for lab, _ in cited):
            out.context_only_dropped += 1  # already processed by the previous window
            continue
        evidence: list[EvidenceSpan] = []
        failed = False
        for lab, quote in cited:
            seg = labels[lab]
            span = locate(seg.normalized_text, quote, taken.get(seg.segment_id, []))
            if span is None:
                out.repairable.append(f"{item.ref}: quote {quote!r} is not found in segment {lab}")
                # A fabricated quote may still refer to a real item. Keep a
                # source anchor for repair without creating an observation.
                if item.product_spoken and item.quantity_text and item.unit_spoken:
                    anchors = []
                    for start, end in term_spans(seg.normalized_text, item.product_spoken):
                        anchor = EvidenceSpan(segment_id=seg.segment_id, start=start, end=end,
                                              quote=seg.normalized_text[start:end])
                        if check_quantity_scope([anchor], segment_index, labels, item.product_spoken,
                                                item.quantity_text, item.unit_spoken, item.operation,
                                                item.location) is None:
                            anchors.append(anchor)
                    if anchors:
                        out.unresolved_mentions.append(anchors)
                failed = True
                break
            taken.setdefault(seg.segment_id, []).append(span)
            evidence.append(EvidenceSpan(segment_id=seg.segment_id, start=span[0], end=span[1],
                                         quote=seg.normalized_text[span[0]:span[1]]))
        if failed:
            continue
        item = complete_split_measure(item, evidence, segment_index)
        if len(evidence) == 1 and item.operation == "count" and item.quantity_text and item.product_spoken:
            terms = [item.quantity_text, item.unit_spoken, item.product_spoken, item.location]
            if item.package:
                terms += [item.package.per_package_text, item.package.per_package_unit_spoken]
            evidence[0] = narrow_count_quote(evidence[0], terms)
        contiguous = all(b - a == 1 for a, b in zip(numbers, numbers[1:]))
        first_seg = segment_index[evidence[0].segment_id]
        position = (first_seg.ordinal, evidence[0].start, index)
        item_conversions: list[str] = []
        built = _build(item, evidence, contiguous, reply.language, known, created, positions, position,
                       run_id, tie, already, segment_index, labels, seg_lang, item_conversions)
        # --- relation to observations extracted earlier (previous windows or this reply) ---
        skip = False
        for other in list(already) + list(created.values()):
            if not _overlaps(built, other):
                continue
            if other in created.values() or other.observation_id in out.replaced:
                built.errors.append(f"overlaps_observation:{other.observation_id}")
            elif _spans(built.evidence) == _spans(other.evidence) and built.operation == other.operation:
                out.skipped_duplicates += 1  # same evidence re-extracted: idempotent, not a new event
                skip = True
            elif (_contained(other, built) and other.status == "needs_review"
                  and not any(a.correction_of == other.observation_id for a in already)
                  and not any(c.correction_of == other.observation_id for c in created.values())):
                out.replaced.append(other.observation_id)
                out.replaced_notes.append(
                    f"partial observation {other.observation_id} (needs_review, evidence within a split statement) "
                    f"replaced by {built.observation_id}")
            else:
                built.errors.append(f"overlaps_earlier_observation:{other.observation_id}")
        if skip:
            continue
        # --- physically overlapping ASR inputs: keep both, flag the later one, same rule in and across windows ---
        for other in list(already) + list(created.values()):
            if other.operation == built.operation and _padding_overlap_duplicate(built, other, segment_index):
                built.errors.append(f"possible_overlap_duplicate_of:{other.observation_id}")
        if built.errors and built.status == "ok":
            built = built.model_copy(update={"status": "needs_review"})
        created[item.ref] = built
        positions[item.ref] = position
        out.conversions += item_conversions
        out.observations.append(built)
    for message, involved in source_relation_problems(labels, context_labels, [*already, *out.observations]):
        out.semantic_repair.append(message)
        out.reply_errors.append(message)
        for observation in involved:
            if observation in out.observations:
                observation.status = "needs_review"
                observation.errors.append(message)
    return out


def _physically_overlap(a: TranscriptSegment, b: TranscriptSegment) -> bool:
    return a.segment_id != b.segment_id and a.input_start < b.input_end and b.input_start < a.input_end


def _padding_overlap_duplicate(new: ValidatedObservation, old: ValidatedObservation,
                               seg_index: dict[str, TranscriptSegment]) -> bool:
    """Identical content decoded in two ASR inputs that share audio samples: may be one utterance decoded
    twice. Utterances in disjoint inputs are never touched (text equality alone is not duplication)."""
    segs_new = [seg_index.get(e.segment_id) for e in new.evidence]
    segs_old = [seg_index.get(e.segment_id) for e in old.evidence]
    if not any(a is not None and b is not None and _physically_overlap(a, b) for a in segs_new for b in segs_old):
        return False
    same_words = fold(" ".join(e.quote for e in new.evidence)) == fold(" ".join(e.quote for e in old.evidence))
    same_values = (new.quantity is not None and
                   (new.quantity, new.unit, new.quantity_max, fold(new.product_spoken or "")) ==
                   (old.quantity, old.unit, old.quantity_max, fold(old.product_spoken or "")))
    return same_words or same_values


def _build(item: LLMObservation, evidence: list[EvidenceSpan], contiguous: bool, language: str, known, created,
           positions, position, run_id, tie, already, segment_index, labels, seg_lang,
           conversions) -> ValidatedObservation:
    errors: list[str] = []
    joined = " ".join(_words(e.quote) for e in evidence)
    if language != "fr":
        errors.append(f"unsupported_language:{language}")
    foreign = {}  # lexical evidence from the segment text and from the quote itself (never from the model's claim alone)
    for e in evidence:
        for found in (seg_lang.get(e.segment_id), assess_language(e.quote)):
            if found not in (None, "fr"):
                foreign.setdefault(e.segment_id, found)
    for seg_id, found in foreign.items():
        errors.append(f"language_evidence:{seg_id}:{found}")
    if foreign and language == "fr":
        errors.append(f"language_disagreement:model=fr,lexical={sorted(set(foreign.values()))}")
    if not contiguous:
        errors.append("evidence_spans_not_adjacent")
    for e in evidence:  # ASR uncertainty (e.g. ambiguous padded overlap) qualifies every observation citing it
        seg = segment_index.get(e.segment_id)
        if seg is not None and seg.status == "needs_review":
            errors.append(f"asr_needs_review:{seg.segment_id}" + (f": {'; '.join(seg.errors)}" if seg.errors else ""))

    # --- numbers: derived from the spoken words, the model value is only cross-checked ---
    quantity = maximum = None
    qualifier = "unknown"
    quantity_text = item.quantity_text
    if quantity_text is not None:
        if not _in(quantity_text, joined):
            errors.append(f"quantity_text {quantity_text!r} not in evidence")
            quantity_text = None
        else:
            if item.qualifier == "approximate":
                cited = " ".join(e.quote for e in evidence)
                for modifier in APPROX_WORDS:
                    phrase = f"{modifier} {quantity_text}"
                    span = locate(cited, phrase)
                    if span is not None and _in(phrase, joined):
                        quantity_text = cited[span[0]:span[1]]
                        break
            parsed = parse_quantity(quantity_text)
            if parsed is None:
                errors.append(f"quantity_text {quantity_text!r} not understood")
            else:
                qualifier = parsed.qualifier
                if parsed.value is not None:
                    quantity = dec_str(parsed.value)
                if parsed.maximum is not None:
                    maximum = dec_str(parsed.maximum)
                if item.quantity is not None and parsed.value is not None:
                    try:
                        if Decimal(item.quantity) != parsed.value:
                            errors.append(f"model quantity {item.quantity} != derived {quantity}")
                    except Exception:
                        errors.append(f"model quantity {item.quantity!r} is not a decimal")
    elif item.quantity is not None:
        errors.append("model quantity without spoken quantity_text ignored")

    # --- unit ---
    unit_spoken, unit = item.unit_spoken, None
    if unit_spoken is not None:
        if not _in(unit_spoken, joined):
            # Recover inflection only when one explicit spelling in the cited text
            # denotes exactly the same unit. Preserve the actual spoken form.
            wanted = normalize_unit(unit_spoken)
            variants = {w for e in evidence for w in re.findall(r"\b[^\W\d_]+\b", e.quote)
                        if is_known_unit(wanted) and normalize_unit(w) == wanted}
            if len(variants) == 1:
                unit_spoken = next(iter(variants))
            else:
                errors.append(f"unit_spoken {unit_spoken!r} not in evidence")
                unit_spoken = None
        if unit_spoken is not None:
            unit = normalize_unit(unit_spoken)
            if not is_known_unit(unit):
                errors.append(f"unit_not_recognized:{unit}")  # kept as spoken-normalized, never converted

    # --- product / location must be words of the evidence ---
    product = item.product_spoken
    product_in_evidence = product is not None and _in(product, joined)
    if product is not None and not product_in_evidence and item.operation != "correction":
        errors.append(f"product_spoken {product!r} not in evidence")
    location = item.location
    if location is not None and not _in(location, joined):
        errors.append(f"location {location!r} not in evidence")
        location = None

    # --- operation vs. actual evidence (negation, comparatives) ---
    operation = item.operation
    cue = negation_cue(joined)
    if operation == "negation":
        if cue != "negation":
            errors.append("negation_not_in_evidence" if cue is None else "negation_cue_is_comparative")
            operation = "unknown"
        if quantity_text is not None:
            errors.append("negation_with_quantity")
        if product is None:
            errors.append("negation_product_missing")
    elif operation in ("count", "correction"):
        if cue == "negation" and item.operation == "count":
            errors.append("negation_cue_in_evidence_but_operation_count")
        elif cue == "comparative":
            errors.append("lower_bound_not_representable")
            if qualifier == "exact":
                qualifier = "approximate"
    if operation in ("count", "correction") and quantity_text and quantity is not None and unit_spoken:
        problem = check_quantity_scope(
            evidence, segment_index, labels, product if product_in_evidence else None,
            quantity_text, unit_spoken, operation)
        if problem:
            errors.append(problem)
    if operation == "unknown":
        errors.append("operation_unknown")
    if operation == "count":
        if quantity is None:
            errors.append("quantity_missing")
    if quantity is not None and unit is None and operation != "negation":
        errors.append("unit_missing")

    # --- packaging: only from explicit "<package> de <n> <unit>" syntax; totals derived deterministically ---
    packaging = _packaging(item, joined, quantity, qualifier, unit, errors)
    # A speaker can count containers and their contents without naming a product.
    if operation == "count" and product is None and not (packaging and packaging.total is not None):
        errors.append("product_missing")
    if packaging and packaging.per_package_unit:
        if not is_known_unit(packaging.per_package_unit):
            errors.append("unrecognized_package_contents_unit")
        if product and normalize_unit(product) == packaging.per_package_unit:
            tail = segment_index[evidence[-1].segment_id].normalized_text[evidence[-1].end:]
            last_unit = _words(item.package.per_package_unit_spoken or "")
            if (packaging.total is not None and joined.rstrip(" .!?,;:").endswith(last_unit)
                    and not tail.lstrip(" .!?,;:").strip()):
                product = None  # the sole terminal noun denotes the verified inner unit
            else:
                errors.append("package_contents_unit_used_as_product")

    # --- correction target ---
    correction_of, revision = None, 1
    if operation == "correction" or item.operation == "correction":
        target, why = _resolve_target(item, product if product_in_evidence else None, unit, known, created,
                                      positions, position, already, segment_index)
        if target is None:
            errors.append(why)
        else:
            correction_of, revision = target.observation_id, target.revision + 1
            if product and not product_in_evidence and fold(product) != fold(target.product_spoken or ""):
                errors.append("correction_product_not_in_evidence_or_target")
            if (product and not product_in_evidence and not re.search(
                    r"\b(non|pardon|plutot|rectification|corrige)\b", joined)):
                errors.append("correction_cue_missing_from_evidence_for_inherited_product")
            quantity, maximum, unit, note = _harmonize_unit(quantity, maximum, unit, target)
            if note:
                conversions.append(note)
            if not product_in_evidence:
                product = target.product_spoken  # bare "non, deux kilos" keeps the corrected product
        if quantity is None:
            errors.append("correction_without_quantity")
    elif item.correction_of is not None:
        errors.append("correction_of ignored: operation is not correction")

    spans = [[e.segment_id, e.start, e.end] for e in evidence]
    key = (json_key(spans), operation)
    n = tie.get(key, 0)
    tie[key] = n + 1
    return ValidatedObservation(
        observation_id=stable_id("obs", run_id, spans, operation, n),
        product_spoken=product, quantity_text=quantity_text, quantity=quantity, quantity_max=maximum,
        qualifier=qualifier if (quantity is not None or qualifier == "unknown") else "unknown",
        unit_spoken=unit_spoken, unit=unit, packaging=packaging, location=location,
        operation=operation, evidence=evidence, correction_of=correction_of, revision=revision,
        status="needs_review" if errors else "ok", errors=errors)


def json_key(spans) -> str:
    return repr(spans)


def _packaging(item: LLMObservation, joined: str, quantity, qualifier, unit, errors) -> Packaging | None:
    pkg = item.package
    if pkg is not None and (pkg.per_package_text or pkg.per_package_unit_spoken):
        per_q = per_unit = None
        explicit = False
        if unit is None or unit_dim(unit):
            errors.append("package_contents_without_package_unit")
        elif not (pkg.per_package_text and pkg.per_package_unit_spoken and item.unit_spoken):
            errors.append("package_contents_incomplete")
        else:
            connector = r"(?:de|d|du|des|contenant|avec|a|renfermant)"
            pattern = (rf"(?<![\w-]){re.escape(_words(item.unit_spoken))}\s+{connector}\s+"
                       rf"{re.escape(_words(pkg.per_package_text))}\s+{re.escape(_words(pkg.per_package_unit_spoken))}(?![\w-])")
            if not re.search(pattern, joined):
                errors.append("package_contents_not_explicit_in_evidence")
            else:
                pp = parse_quantity(pkg.per_package_text)
                if pp is None or pp.value is None or pp.qualifier != "exact":
                    errors.append(f"per_package_text {pkg.per_package_text!r} not an exact number")
                else:
                    explicit = True
                    per_q, per_unit = dec_str(pp.value), normalize_unit(pkg.per_package_unit_spoken)
        total = total_unit = None
        if explicit and quantity is not None and qualifier == "exact":
            total, total_unit = dec_str(Decimal(quantity) * Decimal(per_q)), per_unit
        elif explicit:
            errors.append("package total not derivable: package count not exact")
        return Packaging(count=quantity if unit else None, unit=unit,
                         per_package_quantity=per_q, per_package_unit=per_unit, total=total, total_unit=total_unit)
    if unit in PACKAGE_UNITS and quantity is not None:
        return Packaging(count=quantity, unit=unit)  # contents unknown, never invented
    return None


def unit_dim(unit: str | None) -> str | None:
    return dimension(unit)


# ------------------------------------------------------------------ correction targets
def _resolve_target(item, product, unit, known, created, positions, position, already, seg_index):
    """Return (target, '') or (None, error). Never links when the target is wrong or ambiguous."""
    label = item.correction_of
    if label is None:
        return None, "correction_target_missing"
    target = created.get(label) if label in created else known.get(label)
    if target is None:
        return None, f"correction_target_unknown:{label}"
    if label in created and positions[label] >= position:
        return None, f"correction_target_not_earlier:{label}"
    everything = list(already) + list(created.values())
    # Follow the chain to the latest valid revision so a retry/second correction never re-applies to stale rows.
    corrected_by: dict[str, list[ValidatedObservation]] = {}
    for o in everything:
        if o.correction_of and o.status == "ok":
            corrected_by.setdefault(o.correction_of, []).append(o)
    while target.observation_id in corrected_by:
        successors = corrected_by[target.observation_id]
        if len(successors) != 1:
            return None, f"correction_target_ambiguous:{label}:{len(successors)} competing corrections"
        target = successors[0]
    problem = _target_mismatch(product, unit, target)
    if problem:
        return None, f"correction_target_{problem}:{label}"
    # Ambiguity guard: other live (not yet corrected) candidates that fit the same description.
    cur_ord = position[0]
    pool = {}
    for lab, o in list(known.items()) + [(r, c) for r, c in created.items() if positions[r] < position]:
        if o.operation == "negation" or o.observation_id in corrected_by:
            continue
        if _target_mismatch(product, unit, o):
            continue
        if product is None:  # bare correction: only nearby statements compete
            ev_seg = seg_index.get(o.evidence[-1].segment_id)
            if ev_seg is not None and ev_seg.ordinal < cur_ord - 1:
                continue
        pool[o.observation_id] = lab
    pool.setdefault(target.observation_id, label)
    if len(pool) > 1:
        return None, f"correction_target_ambiguous:{label}:candidates {sorted(pool.values())}"
    return target, ""


def _harmonize_unit(quantity, maximum, unit, target: ValidatedObservation):
    """Resolved correction in another unit of the SAME physical dimension (g vs kg, ml/cl vs l):
    express quantity/quantity_max in the corrected observation's unit with exact decimal arithmetic.
    The spoken words (`quantity_text`, `unit_spoken`) are untouched, so the original is recoverable.
    Counts, packages and unknown units have no dimension and are never converted."""
    if quantity is None or unit is None or target.unit is None or unit == target.unit:
        return quantity, maximum, unit, None
    dim = dimension(unit)
    if dim is None or dim != dimension(target.unit):
        return quantity, maximum, unit, None
    new_q = convert(Decimal(quantity), unit, target.unit)
    new_max = convert(Decimal(maximum), unit, target.unit) if maximum is not None else None
    if new_q is None or (maximum is not None and new_max is None):  # unreachable for same dimension; stay safe
        return quantity, maximum, unit, None
    note = (f"{dec_str(Decimal(quantity))}{'-' + maximum if maximum else ''} {unit} -> "
            f"{dec_str(new_q)}{'-' + dec_str(new_max) if new_max is not None else ''} {target.unit} "
            f"(correction of {target.observation_id})")
    return dec_str(new_q), (dec_str(new_max) if new_max is not None else None), target.unit, note


def _target_mismatch(product: str | None, unit: str | None, target: ValidatedObservation) -> str | None:
    if product and target.product_spoken:
        a, b = fold(product), fold(target.product_spoken)
        if a not in b and b not in a:
            return "product_mismatch"
    if unit and target.unit and not compatible(unit, target.unit):
        return "unit_mismatch"
    return None
