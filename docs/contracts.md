# Speechflow interfaces (v1)

The wire models live in `speechflow.contracts`. The public configuration is `speechflow.config.Settings`. Callers pass model objects within a process and serialize with Pydantic's JSON mode across processes. Constructors reject undeclared fields.

* `speechflow.audio.transcribe_file(path: Path, settings: Settings) -> AudioResult` normalizes once. `speechflow.audio.capture_microphone(settings: Settings, spool: Path) -> AudioResult` owns bounded capture and final flush. An error before source identification raises an actionable exception and causes a nonzero CLI exit. Segments with recognized identity retain their errors in the result.
* `speechflow.extraction.extract(segments: list[TranscriptSegment], *, run_id: str, settings: Settings) -> ExtractionResult` accepts ordered segments including adjacent context. The extractor owns one bounded repair and retention of replies, application validation, and stable revision events.
* `speechflow.pipeline.process(path: Path | None, *, mode: Literal["run","transcribe","mic"], settings: Settings, out: Path) -> RunManifest` handles the ASR worker process and extraction in sequence. For microphone capture, `path` is None. The caller persists the returned manifest even on a review or partial error. Fatal errors without an identifiable source are exceptions.
* `speechflow.assets.prepare(settings) -> dict` acquires and verifies pinned assets. In offline mode it verifies locally. `speechflow.assets.doctor(settings) -> dict` inspects installed assets, devices and loopback service without provisioning.
* `speechflow.persistence.save(manifest: RunManifest, database: Path, out: Path) -> None` commits history and writes JSON/CSV/table views. `speechflow.evaluation.evaluate(manifest: Path, out: Path, settings: Settings) -> dict` runs evaluation. The CLI calls those interfaces.

All times in audio segments use half-open source-relative sample offsets on the 16,000 Hz decoded timeline. Core ranges own disjoint speech. Padded input ranges may overlap. One model response per submitted segment is required. Raw retains the unchanged returned model object. Normalized text and Unicode span offsets are separate. Only a uniquely evidenced overlap can be removed from extraction context.

The file source ID derives from the original file SHA-256. The mic source ID is a unique capture session, with its finalized PCM digest stored separately. Segment identity uses source and sample locations. Behavior settings, runtime versions, artifacts, pipeline implementation, schema and actual backend/dtype form the processing version. Run ID additionally distinguishes source and requested mode.

Quantity decimals serialize as base-ten strings. The upper bound encodes a range, and the qualifier distinguishes unknown and approximate values. Separate count and per-package fields support explicit totals. A correction appends a revision with its own stable observation ID, references the previous ID and preserves the original evidence. Missing or ambiguous targets remain reviewable.

Each evidence span quotes the exact segment normalized text at its half-open Unicode code-point offsets. Raw attempts remain attached to the run. Stage status distinguishes a completed transcription-only request from finalized extraction.

## Adding another extraction domain

The installed CLI ships one inventory schema. A second domain is a versioned Python package change. It can reuse `transcribe_file`, `capture_microphone`, the ordered `TranscriptSegment` records, sample offsets, and `EvidenceSpan`. The `ChatBackend` protocol accepts messages and a generation schema independently of the audio source.

1. Define the new candidate and validated observation in `src/speechflow/contracts.py`, give the public contract its own `SCHEMA_VERSION`, and publish its JSON Schema alongside `schemas/inventory.v1.json`. Keep the existing inventory schema readable for historical runs.
2. Define the model reply and `GENERATION_SCHEMA_VERSION` in `extraction/schema.py`, publish the matching JSON under `extraction/schemas/`, and version the field instructions and examples with `PROMPT_VERSION` in `extraction/prompt.py`. In `extraction/validate.py` implement the domain's evidence and value checks. Adapt the domain fields in `extraction/core.py` used for carryover and repair reconciliation. These paths are relative to `src/speechflow/`.
3. Thread the new observation type through `ExtractionResult` and `RunManifest`. Update `persistence.py` for manifest checks, SQLite inserts, CSV columns and terminal projections. Add the next numbered SQL migration alongside `migrations/0001_initial.sql` for any new stored fields or SQL views. Exercise old-run export and new-run replay, and supply the domain's expected fields and fixtures to `evaluation.py`. The new schema and implementation participate in processing identity.
