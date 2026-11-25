"""Shared wire contracts. All sample offsets use the decoded 16 kHz timeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "inventory.v1"
Status = Literal["ok", "no_speech", "needs_review", "error"]
StageStatus = Literal["ok", "no_speech", "needs_review", "error", "not_run"]
DecimalString = str  # Base-10 strings; validator must derive them from spoken evidence.


def stable_id(kind: str, *parts: Any) -> str:
    data = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{kind}_{hashlib.sha256(data.encode()).hexdigest()}"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioSource(Contract):
    source_id: str
    kind: Literal["file", "mic"]
    digest: str
    name: str
    native_rate: int = Field(gt=0)
    native_channels: int = Field(gt=0)
    sample_rate: Literal[16000] = 16000
    sample_count: int = Field(ge=0)
    channel_mix: str


class AudioSegment(Contract):
    segment_id: str
    source_id: str
    ordinal: int = Field(ge=0)
    input_start: int = Field(ge=0)
    core_start: int
    core_end: int
    input_end: int

    @model_validator(mode="after")
    def ordered_bounds(self):
        if not self.input_start <= self.core_start < self.core_end <= self.input_end:
            raise ValueError("Invalid half-open segment boundaries")
        if self.input_end - self.input_start > 400_000:
            raise ValueError("ASR input exceeds 25 seconds at 16 kHz")
        return self


class TranscriptSegment(AudioSegment):
    raw: dict[str, Any]
    normalized_text: str
    status: Status = "ok"
    errors: list[str] = Field(default_factory=list)


class EvidenceSpan(Contract):
    """Half-open Unicode character offsets into one segment's normalized_text."""

    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class Packaging(Contract):
    count: DecimalString | None = None
    unit: str | None = None
    per_package_quantity: DecimalString | None = None
    per_package_unit: str | None = None
    total: DecimalString | None = None
    total_unit: str | None = None


class ExtractionCandidate(Contract):
    product_spoken: str | None = None
    quantity_text: str | None = None
    quantity: DecimalString | None = None
    quantity_max: DecimalString | None = None
    qualifier: Literal["exact", "approximate", "range", "unknown"] = "unknown"
    unit_spoken: str | None = None
    unit: str | None = None
    packaging: Packaging | None = None
    location: str | None = None
    operation: Literal["count", "correction", "negation", "unknown"] = "count"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    correction_of: str | None = None


class ValidatedObservation(ExtractionCandidate):
    observation_id: str
    catalog_match: str | None = None
    revision: int = Field(default=1, ge=1)
    status: Literal["ok", "needs_review"] = "ok"
    errors: list[str] = Field(default_factory=list)


class AudioResult(Contract):
    source: AudioSource
    segments: list[TranscriptSegment]
    status: Status
    errors: list[str] = Field(default_factory=list)
    backend: str
    dtype: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(Contract):
    observations: list[ValidatedObservation]
    status: Status
    errors: list[str] = Field(default_factory=list)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RunManifest(Contract):
    run_id: str
    processing_version: str
    schema_version: str = SCHEMA_VERSION
    mode: Literal["transcribe", "run", "mic"]
    status: Status
    transcription_status: StageStatus
    extraction_status: StageStatus
    source: AudioSource
    segments: list[TranscriptSegment]
    observations: list[ValidatedObservation]
    provenance: dict[str, Any]
    metrics: dict[str, Any]
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
