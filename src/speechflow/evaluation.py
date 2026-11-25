"""Evaluate French WER and extraction on separately reported model inputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import assets, extraction, pipeline
from .config import Settings
from .contracts import Packaging, TranscriptSegment, ValidatedObservation, stable_id

WER_POLICY = (
    "NFC, casefold, curly apostrophe to straight. Keep accents. Split on punctuation "
    "and whitespace except internal apostrophes and decimal separators. Keep number "
    "surface forms without expanding words or equating decimal comma and point. "
    "Concatenate ordered normalized segments. Repetitions count as insertions."
)
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_SOURCE = "google/fleurs:fr_fr/validation"
FLEURS_CLIPS = {
    "10132087162443371989.wav": (
        "8bb53b14f4cc695c54a136a5db57e534344a225d81a6b5314ca6cdb0a12c7005",
        "ced94c3d3ded4d89b53562fb3671c54317658875afef5c20c5aa8c950e350a95",
    ),
    "10198958753510138668.wav": (
        "a9f2f1630034aae5bbe5273a25166d9ed1e92bedf4d613c7398fbda92b65ed12",
        "7fd9a2c5ce0d6f338602981a8f7d97b76df8a08f2527f008871fd82b69f4aeb2",
    ),
}
FIELDS = (
    "product_spoken", "quantity", "quantity_max", "qualifier", "unit", "operation",
    "location", "negation", "correction", "packaging.count", "packaging.unit",
    "packaging.per_package_quantity", "packaging.per_package_unit",
    "packaging.total", "packaging.total_unit",
)
DECIMALS = {
    "quantity", "quantity_max", "packaging.count", "packaging.per_package_quantity",
    "packaging.total",
}


class GoldObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_spoken: str | None = None
    quantity: str | None = None
    quantity_max: str | None = None
    qualifier: Literal["exact", "approximate", "range", "unknown"] = "exact"
    unit: str | None = None
    operation: Literal["count", "negation", "correction", "unknown"] = "count"
    location: str | None = None
    packaging: Packaging | None = None
    correction_of: int | None = Field(default=None, ge=0)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str | None = None
    audio: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    duration_s: float | None = Field(default=None, gt=0)
    reference: str | None = None
    reference_segments: list[str] | None = None
    reference_review: bool = False
    expected_observations: list[GoldObservation] | None = None
    condition: str = "unspecified"
    language: str = "fr"
    source: str | None = None
    revision: str | None = None
    license: str | None = None
    human_verified: bool = False
    synthetic: bool = False
    sample_rate: int | None = Field(default=None, gt=0)
    frames: int | None = Field(default=None, gt=0)
    wav_format: int | None = None
    bits: int | None = None
    channels: int | None = None
    tsv_line: int | None = None
    prompt_id: str | None = None
    corpus_normalized_reference: str | None = None

    @model_validator(mode="after")
    def check_inputs(self):
        if self.reference is not None and self.reference_segments is not None:
            raise ValueError("supply reference or reference_segments")
        if not self.audio and self.expected_observations is None:
            raise ValueError("reference-only cases require expected_observations")
        if self.expected_observations is not None and not self.texts:
            raise ValueError("annotated extraction cases require a reference")
        if self.human_verified and not (
            self.audio and self.sha256 and self.duration_s and self.source and self.revision
            and self.texts and not self.synthetic
        ):
            raise ValueError("human_verified needs pinned, non-synthetic audio and a reference")
        for i, observation in enumerate(self.expected_observations or []):
            prior = observation.correction_of
            if observation.operation == "correction" and (prior is None or prior >= i):
                raise ValueError("a gold correction must link an earlier zero-based observation index")
            if observation.operation != "correction" and prior is not None:
                raise ValueError("only gold corrections may link a prior observation")
            _values(observation)
        return self

    @property
    def texts(self) -> list[str]:
        if self.reference_segments is not None:
            return self.reference_segments
        return [self.reference] if self.reference else []


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.findall(r"\d+(?:[.,]\d+)+|[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def _edits(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def _label(value: str | None) -> str | None:
    if value is not None:
        return " ".join(_words(value)) or None
    return None


def _decimal(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite decimal {value!r}")
    return number


def _values(observation: GoldObservation | ValidatedObservation) -> dict:
    fields = {}
    for field in FIELDS:
        if field in {"negation", "correction"}:
            continue
        if field.startswith("packaging."):
            value = getattr(observation.packaging, field.split(".")[1], None)
        else:
            value = getattr(observation, field)
        if value is not None:
            fields[field] = _decimal(value) if field in DECIMALS else _label(value)
    if observation.operation == "negation":
        fields["negation"] = True
    return {key: value for key, value in fields.items() if value is not None}


def _align(observations):
    """Align by normalized product and occurrence, without using scored values."""
    seen = Counter()
    mentions = []
    for observation in observations:
        product = _label(observation.product_spoken)
        seen[product] += 1
        mentions.append((product, seen[product]))
    return mentions


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _pr(tp: int, predicted: int, expected: int) -> dict:
    return {"tp": tp, "fp": predicted - tp, "fn": expected - tp,
            "predicted": predicted, "expected": expected,
            "precision": _ratio(tp, predicted), "recall": _ratio(tp, expected)}


def _score(gold: list[GoldObservation], candidates: list[ValidatedObservation],
           segments: list[TranscriptSegment] | None = None) -> dict:
    if segments:
        positions = {s.segment_id: s.ordinal for s in segments}
        candidates = sorted(candidates, key=lambda observation: min(
            ((positions[span.segment_id], span.start) for span in observation.evidence
             if span.segment_id in positions),
            default=(len(segments), 0),
        ))
    expected_ids, predicted_ids = _align(gold), _align(candidates)
    expected = {}
    accepted = {}
    links = {o.observation_id: key for key, o in zip(predicted_ids, candidates, strict=True)}
    for key, obs in zip(expected_ids, gold, strict=True):
        expected[key] = _values(obs)
        if obs.correction_of is not None:
            expected[key]["correction"] = expected_ids[obs.correction_of]
    for key, obs in zip(predicted_ids, candidates, strict=True):
        if obs.status == "ok":
            accepted[key] = _values(obs)
            if obs.operation == "correction":
                accepted[key]["correction"] = links.get(obs.correction_of, "unlinked correction")
    matches = expected.keys() & accepted.keys()
    fields = {}
    for field in FIELDS:
        tp = sum(field in expected[key] and field in accepted[key]
                 and expected[key][field] == accepted[key][field] for key in matches)
        fields[field] = _pr(tp, sum(field in values for values in accepted.values()),
                            sum(field in values for values in expected.values()))
    return {"gold_mentions": len(gold), "returned_candidates": len(candidates),
            "accepted_candidates": len(accepted), "review_candidates": len(candidates) - len(accepted),
            "covered_gold_mentions": len(matches), "fields": fields}


def _aggregate(results: list[dict]) -> dict:
    scores = [result["scores"] for result in results]
    counts = {key: sum(score[key] for score in scores) for key in (
        "gold_mentions", "returned_candidates", "accepted_candidates", "review_candidates",
        "covered_gold_mentions",
    )}
    counts["abstention_rate"] = _ratio(counts["review_candidates"], counts["returned_candidates"])
    counts["gold_coverage"] = _ratio(counts["covered_gold_mentions"], counts["gold_mentions"])
    counts["gold_uncovered_rate"] = _ratio(
        counts["gold_mentions"] - counts["covered_gold_mentions"], counts["gold_mentions"])
    counts["cases_with_no_accepted_candidate"] = sum(not score["accepted_candidates"] for score in scores)
    counts["positive_cases_with_no_accepted_candidate"] = sum(
        bool(score["gold_mentions"]) and not score["accepted_candidates"] for score in scores)
    counts["cases"] = len(scores)
    counts["blocked_cases"] = sum(result["status"] == "BLOCKED" for result in results)
    counts["failed_cases"] = sum(result["status"] == "error" for result in results)
    counts["completed_cases"] = len(scores) - counts["blocked_cases"] - counts["failed_cases"]
    counts["measurement_status"] = (
        "NOT MEASURED" if not counts["completed_cases"] else
        "MEASURED" if counts["completed_cases"] == counts["cases"] else "PARTIAL")
    return {**counts, "fields": {field: _pr(*(
        sum(score["fields"][field][key] for score in scores) for key in ("tp", "predicted", "expected")
    )) for field in FIELDS}}


def _read_manifest(manifest: Path, content: bytes) -> list[tuple[str, Case]]:
    cases = {}
    for n, line in enumerate(content.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case.model_validate_json(line)
            key = case.id or case.audio or f"line-{n}"
            if key in cases:
                raise ValueError(f"duplicate case ID {key!r}")
            cases[key] = case
        except ValueError as exc:
            raise ValueError(f"{manifest}:{n}: {exc}") from exc
    if not cases:
        raise ValueError(f"{manifest}: no evaluation cases")
    return list(cases.items())


def _reference_segments(case: Case, key: str) -> list[TranscriptSegment]:
    source_id = stable_id("evaluation_source", key, case.texts)
    return [
        TranscriptSegment(segment_id=stable_id("evaluation_segment", source_id, i),
                          source_id=source_id, ordinal=i, input_start=i * 16000, core_start=i * 16000,
                          core_end=(i + 1) * 16000, input_end=(i + 1) * 16000,
                          normalized_text=text, raw={})
        for i, text in enumerate(case.texts)
    ]


def _inventory_result(result) -> dict:
    return {"status": result.status, "errors": result.errors, "metrics": result.metrics,
            "observations": [o.model_dump(mode="json") for o in result.observations],
            "attempts": result.attempts}


def _blocked(error: str) -> dict:
    return {"status": "BLOCKED", "errors": [error], "observations": []}


def _failed(error: Exception) -> dict:
    return {"status": "error", "errors": [f"{type(error).__name__}: {error}"], "observations": []}


def _wer_basis(case: Case) -> str | None:
    if case.synthetic or case.language != "fr" or not _words(" ".join(case.texts)):
        return None
    if case.human_verified:
        return "manifest human verification attestation"
    reference_digest = hashlib.sha256(" ".join(case.texts).encode()).hexdigest()
    if ((case.source, case.revision) == (FLEURS_SOURCE, FLEURS_REVISION)
            and FLEURS_CLIPS.get(case.audio) == (case.sha256, reference_digest)):
        return "pinned FLEURS corpus verification, no independent listening labels"
    return None


def _wer_summary(items: list[dict]) -> dict:
    attempted = [item for item in items if item["edits"] is not None]
    usable = any(item["inference_completed"] or item["hypothesis_words"] for item in attempted)
    reference_words = sum(item["reference_words"] for item in attempted)
    edits = sum(item["edits"] for item in attempted)
    return {"eligible_clips": len(items), "eligible_duration_s": round(sum(
        item["duration_s"] or 0 for item in items), 6),
        "eligible_reference_words": sum(item["reference_words"] for item in items),
        "scored_clips": len(attempted) if usable else 0,
        "scored_duration_s": round(sum(item["duration_s"] or 0 for item in attempted), 6) if usable else 0,
        "reference_words": reference_words if usable else 0, "edits": edits if usable else None,
        "wer": _ratio(edits, reference_words) if usable else None,
        "status": ("MEASURED" if len(attempted) == len(items) and
                   all(item["inference_completed"] for item in attempted) else "PARTIAL")
        if usable else "NOT MEASURED"}


def evaluate(manifest: Path, out: Path, settings: Settings) -> dict:
    """Evaluate a JSONL manifest using pinned local inference and write evaluation.json."""
    content = manifest.read_bytes()
    cases = _read_manifest(manifest, content)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    audio_cases = any(case.audio for _, case in cases)
    annotated_cases = any(case.expected_observations is not None for _, case in cases)
    hojo_issues = assets.hojo_issues(settings) if audio_cases else []
    extractor_error = None
    if annotated_cases:
        try:
            digest = assets.extractor_digest(settings)
            if digest != assets.DIGEST:
                extractor_error = f"Pinned {settings.ollama_model} digest unavailable (found {digest})"
        except (ValueError, OSError, KeyError) as exc:
            extractor_error = f"Loopback extractor unavailable: {exc}"

    exact_results, asr_results, wer_items = [], [], []
    rows = []
    for index, (key, case) in enumerate(cases):
        row = {"id": key, "audio": case.audio, "source": case.source, "revision": case.revision,
               "license": case.license, "sha256": case.sha256, "language": case.language,
               "duration_s": case.duration_s, "condition": case.condition, "synthetic": case.synthetic,
               "reference": case.texts, "expected_review": case.reference_review}
        gold = case.expected_observations
        if gold is not None:
            row["gold"] = [o.model_dump(mode="json") for o in gold]
            reference_segments = _reference_segments(case, key)
            if extractor_error:
                result = _blocked(extractor_error)
                candidates = []
            else:
                try:
                    extracted = extraction.extract(reference_segments,
                                                   run_id=stable_id("evaluation", key, case.texts),
                                                   settings=settings)
                    result, candidates = _inventory_result(extracted), extracted.observations
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    result, candidates = _failed(exc), []
            result["scores"] = _score(gold, candidates, reference_segments)
            result["review_decision_correct"] = (
                None if result["status"] in {"error", "BLOCKED"} else
                (result["status"] == "needs_review") == case.reference_review)
            exact_results.append(result)
            row["reference_extraction"] = result

        if case.audio:
            audio_path = manifest.parent / case.audio
            run = None
            try:
                if not audio_path.is_file():
                    audio = _blocked(f"Audio file missing: {audio_path}")
                elif case.sha256 and assets.file_digest(audio_path) != case.sha256:
                    audio = _failed(ValueError(f"Audio SHA-256 differs from manifest: {audio_path}"))
                elif hojo_issues:
                    audio = _blocked("; ".join(hojo_issues))
                else:
                    mode = "run" if gold is not None and not extractor_error else "transcribe"
                    run = pipeline.process(audio_path, mode=mode, settings=settings,
                                           out=out / "runs" / str(index + 1))
                    if case.sha256 and run.source.digest != case.sha256:
                        run = None
                        raise ValueError(f"Pipeline source digest differs from manifest: {audio_path}")
                    audio = {"status": run.transcription_status, "errors": run.errors,
                             "provenance": run.provenance, "metrics": run.metrics,
                             "actual_duration_s": run.source.sample_count / 16000,
                             "segments": [s.model_dump(mode="json") for s in run.segments]}
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                audio = _failed(exc)
            row["asr"] = audio
            if gold is not None:
                if run and run.extraction_status not in {"not_run", "no_speech"}:
                    asr_candidates = run.observations
                    result = {**_inventory_result(run), "status": run.extraction_status}
                else:
                    asr_candidates = []
                    error = extractor_error or "; ".join(audio["errors"])
                    state = ("BLOCKED" if extractor_error or audio["status"] == "BLOCKED" else
                             "error" if audio["status"] == "error" else
                             run.extraction_status if run else "not_run")
                    result = {"status": state, "errors": [error] if error else [], "observations": []}
                result["scores"] = _score(gold, asr_candidates, run.segments if run else None)
                asr_results.append(result)
                row["asr_extraction"] = result

            basis = _wer_basis(case)
            if basis:
                reference = _words(" ".join(case.texts))
                hypothesis = _words(" ".join(seg.normalized_text for seg in run.segments
                                             if seg.status != "error")) if run else []
                if run and case.duration_s is not None and not math.isclose(
                    case.duration_s, run.source.sample_count / 16000, abs_tol=0.02
                ):
                    audio["errors"] = [*audio["errors"], "Decoded duration differs from manifest"]
                    audio["status"] = "error"
                    run = None
                # A completed no_speech run contributes deletions, while missing inputs and
                # models are purposely excluded. Partial runs retain failures in their WER.
                item = {"basis": basis, "condition": case.condition, "duration_s": case.duration_s,
                        "reference_words": len(reference), "hypothesis_words": len(hypothesis),
                        "inference_completed": bool(run and run.transcription_status != "error"),
                        "edits": _edits(reference, hypothesis) if run else None}
                row["wer"] = item
                wer_items.append(item)
        rows.append(row)

    by_condition = defaultdict(list)
    for item in wer_items:
        by_condition[item["condition"]].append(item)
    stage_states = [row[stage]["status"] for row in rows for stage in (
        "reference_extraction", "asr", "asr_extraction"
    ) if stage in row]
    if "error" in stage_states or "BLOCKED" in stage_states:
        status = "error"
    elif any(
        row[stage]["status"] == "needs_review" and not row["expected_review"]
        for row in rows for stage in ("reference_extraction", "asr", "asr_extraction") if stage in row
    ):
        status = "needs_review"
    else:
        status = "ok"
    report = {
        "status": status, "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "requested_cases": len(cases), "requested_audio_cases": sum(bool(c.audio) for _, c in cases),
        "elapsed_s": round(time.perf_counter() - start, 6), "cases": rows,
        "wer": {"normalization": WER_POLICY, **_wer_summary(wer_items),
                "by_condition": {label: _wer_summary(items) for label, items in sorted(by_condition.items())}},
        "extraction": {"alignment": "normalized product and chronological occurrence, including revisions",
                       "policy": "Only ok candidates score as predictions. Missing or misrecognized mentions "
                       "contribute false negatives. Incorrect accepted fields contribute false positives.",
                       "reference": _aggregate(exact_results), "asr": _aggregate(asr_results)},
    }
    (out / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return report
