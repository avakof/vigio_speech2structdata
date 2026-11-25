"""Manifest builders and process helpers for the persistence/CLI tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from speechflow.contracts import (
    AudioSource, EvidenceSpan, Packaging, RunManifest, TranscriptSegment, ValidatedObservation,
    stable_id,
)

FLOUR = "Trois kilos de farine non deux kilos"


def make_segments(source_id: str, texts: list[str]) -> list[TranscriptSegment]:
    out = []
    for i, text in enumerate(texts):
        core_start = i * 160_000 + 8_000
        core_end = core_start + 120_000
        out.append(TranscriptSegment(
            segment_id=stable_id("segment", source_id, core_start, core_end),
            source_id=source_id, ordinal=i, input_start=core_start - 3_200,
            core_start=core_start, core_end=core_end, input_end=core_end + 3_200,
            raw={"text": text, "chunk": i}, normalized_text=text))
    return out


def make_obs(seg: TranscriptSegment, key: str, *, quote: str | None = None, product="tomates",
             quantity="2.5", unit="kg", **extra) -> ValidatedObservation:
    quote = quote or seg.normalized_text.split()[0]
    start = seg.normalized_text.index(quote)
    fields = dict(
        observation_id=stable_id("observation", seg.segment_id, key),
        product_spoken=product, quantity_text="x", quantity=quantity, qualifier="exact",
        unit_spoken=unit, unit=unit,
        evidence=[EvidenceSpan(segment_id=seg.segment_id, start=start,
                               end=start + len(quote), quote=quote)])
    fields.update(extra)
    return ValidatedObservation(**fields)


def make_manifest(*, seed="a", mode="run", pv="pv-1", texts=None, status="ok", observations="default",
                  extraction_status=None, kind="file", metrics=None, errors=None,
                  segments=None) -> RunManifest:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    source_id = stable_id("source", kind, digest)
    texts = texts if texts is not None else ["Il reste deux virgule cinq kilos de tomates"]
    segs = segments if segments is not None else make_segments(source_id, texts)
    if observations == "default":
        observations = [make_obs(s, "o") for s in segs] if mode != "transcribe" else []
    if extraction_status is None:
        extraction_status = "not_run" if mode == "transcribe" else (
            "ok" if status == "ok" else status)
    return RunManifest(
        run_id=stable_id("run", source_id, mode, pv), processing_version=pv, mode=mode,
        status=status, transcription_status="no_speech" if status == "no_speech" else "ok",
        extraction_status=extraction_status,
        source=AudioSource(source_id=source_id, kind=kind, digest=digest, name=f"{seed}.wav",
                           native_rate=48_000, native_channels=2,
                           sample_count=max(1, len(segs)) * 160_000, channel_mix="mean"),
        segments=segs, observations=observations,
        provenance={"backend": "cpu", "dtype": "float32"},
        metrics=metrics or {"asr_seconds": 1.25}, errors=errors or [])


def packaged(seg) -> ValidatedObservation:
    return make_obs(seg, "pk", quote=seg.normalized_text.split()[0], product="lait",
                    quantity=None, unit=None, qualifier="unknown",
                    packaging=Packaging(count="2", unit="carton", per_package_quantity="12",
                                        per_package_unit="bouteille", total="24",
                                        total_unit="bouteille"))


def save_in_worker(args):
    """Top-level so a spawned process can import it."""
    manifest_json, database, out = args
    from speechflow import persistence
    result = persistence.save_run(RunManifest.model_validate_json(manifest_json),
                                  Path(database), Path(out))
    return result.disposition, [p.read_bytes() for p in result.files]


def export_in_worker(args):
    database, run_id = args
    from speechflow import persistence
    return persistence.export_run(Path(database), run_id, "csv")
