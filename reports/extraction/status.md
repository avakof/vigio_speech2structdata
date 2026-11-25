# Extraction status (Gate A feasibility: BLOCKED on model transfer; implementation: unit-tested with a fake backend)

## Gate A — real-model feasibility: BLOCKED
- Ollama 0.40.16 on 127.0.0.1:11434; `qwen3:8b` absent at start; no other local model used.
- `ollama pull qwen3:8b` (PID 98476, started 2026-11-25 ~15:18) is network-bound: 5.2 GB layer a3de86cd1c13,
  ~325 MB done at last check; speed swings 25 KB/s–2.7 MB/s (general link also slow: 12 KB/s Cloudflare test while pulling).
  Partial blobs are in `~/.ollama/models/blobs/*-partial*`; re-running `ollama pull qwen3:8b` resumes.
- NOT RUN: digest verification, real schema-constrained extraction, unload/idle check, schema-subset probe.
- Heavy-model slot was never used (no model loaded).

## Ready to run when the model lands (one exclusive slot, then unload + release)
    .venv/bin/python tools/extraction/probe.py run --out reports/extraction/run-001
    .venv/bin/python tools/extraction/probe.py validate --out reports/extraction/run-001
The probe: reads digest from /api/show + /api/tags, runs 5 fixtures (exact inventory, exact correction, correction in next
segment, cartons x2) with private feasibility schema via `format`=JSON Schema, `think:false`, temperature 0/seed/top_k/num_ctx/num_predict,
one repair retry, Pydantic + independent checks, then 13 single-keyword schema-subset cases, then production `extract()` on the same fixtures,
then `keep_alive:0` unload and a /api/ps idle poll. Raw request/response JSON is saved under `<out>/raw`.
Also verify before use: sha256 of manifest file == `/api/tags` digest and blob sha256 == layer digests.

## Implementation (src/speechflow/extraction/) — tested ONLY against a scripted fake backend
text.py (folding/exact span locate), numbers.py (French numbers, decimal comma, ranges, approximations, collectives),
units.py (normalization, same-dimension conversion only; cartons never converted), schema.py (conservative generation schema,
versioned `inventory.v1.gen1`, exported to `schemas/inventory.v1.gen1.schema.json`), prompt.py, backend.py (stdlib Ollama adapter, replaceable Protocol),
validate.py (evidence spans, derived numbers, units, packaging totals only from explicit contents, correction-target checks),
core.py (`extract()`; bounded windows, read-only previous-segment context, carried observations as correction targets,
one repair retry, truncation/context-overflow detection, raw attempts retained). `extract()` gained optional kw `backend=`.
Unit tests: `tools/extraction/tests/test_extraction_unit.py` — 44 passed (`.venv/bin/python -m pytest tools/extraction/tests -q`).
These tests are mock-backed and say nothing about real-model quality.

## Known limits / open items
- Numeric/unit/evidence derivation is deterministic, but the model's choice of product/target/segments is unverified until the real run.
- Windows cap (8 segments / 1600 chars), context 1 segment, 30 carried observations: heuristics, unmeasured.
- Observations beyond 30 earlier ones are not offered as correction targets (reported in errors, not hidden).
- Tests live under tools/extraction/tests because tests/ is not an owned path; QA may relocate.

## Update — integration findings fixed (generation schema `inventory.v1.gen2`, prompt `inventory.v1.prompt2`)
Still mock-backed; pull still running (~570 MB / 5.2 GB), no model loaded, heavy slot not used.
1. Split statements: an observation cites `segment`/`quote` plus ordered `more_evidence` parts in consecutive segments; the first part
   may be read-only context if a later part is in the segment(s) to extract. Each part gets its exact original offsets; id is derived from all spans
   (anchored once). Context-only statements are dropped and counted; a partial `needs_review` observation whose evidence lies inside a completed
   one is replaced with an explicit note; equal evidence is an idempotent duplicate; any other overlap is kept and flagged. Identical utterances
   in different places stay distinct. Same wording in segments whose padded inputs overlap is KEPT and flagged `possible_overlap_duplicate_of`.
   Segments with ASR status needs_review propagate `asr_needs_review:<segment>` + their errors to every citing observation.
2. Validation: whole-word matching (no "deux" in "vingt-deux"), negation needs a cue in the evidence (comparative "plus de dix" is not negation),
   product/quantity missing => review, location must be evidence words, ambiguous correction targets (named or bare) are not linked, stale targets
   follow the correction chain, package contents only from literal "<package> de <n> <unit>" evidence; no totals for mass/volume units.
3. Prompt bound: send-time upper bound (1 token per 2 UTF-8 bytes + per-message overhead, audited against `prompt_eval_count`), carry list shrinks
   openly, repair prompt includes the prior answer only if it fits (else `repair_skipped_overflow`, evidence kept), no request when output room is missing;
   context after a flush is the tail of the window just closed and is dropped after a failed window. `extract()` verifies the pinned digest on the default
   path before sending text (injected backends only with `expected_digest`); `OllamaBackend` is loopback-only and keeps the shared `loopback_request` policy.
Evidence: `.venv/bin/python -m pytest -q tools/extraction/tests tests/persistence_own tests/qa`.
Open: Overlap decisions are conservative (flag, never drop) pending audio's real overlap output; model reliability for `more_evidence` and the
optional property is unmeasured until the real run (probe has a subset case for optional properties).

## Update — R05 (overlap policy + language evidence); still mock-backed, no model loaded, pull ongoing
- Overlap: identical wording OR identical values (product/quantity/unit/range) in two observations whose segments' ASR inputs physically share samples
  are both kept; the later one gets `possible_overlap_duplicate_of:<id>` -> needs_review. Same rule inside one window and across windows (previously only
  across). Disjoint inputs are never flagged or deduplicated.
- Language: new dependency-free `extraction/language.py` (function-word lexicon veto for fr/en/it/es/de/pt). The model's `language` claim is no longer trusted
  alone: segment text and quote are checked; a clear foreign label gives `language_evidence:<segment>:<code>` (+ `language_disagreement` when the model said fr)
  on every citing observation and at run level, even with zero observations. Short/ambiguous French strings ("farine", "pommes de terre") stay inconclusive.
  Unknown normalized units give `unit_not_recognized:<unit>` (kept, never converted).
- Limits: only 6 languages modelled; <3 alphabetic tokens or content-word-only text is inconclusive; French sentences dominated by quoted foreign phrases may be
  flagged; other languages (e.g. Dutch) are not detected; lexicons are curated, not trained or benchmarked on real transcripts. Added no packages.
- Evidence: `.venv/bin/python -m pytest -q tools/extraction/tests tests/persistence_own` = 175 passed; `tests/qa` = 39 passed;
  R05 rerun saved to `reports/extraction/r05_after.out` (reviewer's original output left untouched).
