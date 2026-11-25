# Local extractor verification

The installed CLI ran the pinned qwen3:8b on the M3 Max. On the 15 curated restaurant transcript fixtures, 16/16 gold observations were covered and all accepted fields matched. The validator routed three extra candidates to review. Both the original failure and the passing 17-call run are retained. [Final reference evaluation](../evaluation/reference-final.json) · [Initial reference evaluation](../evaluation/reference-baseline.json)

The local feasibility experiment covered French counts, two kinds of correction and both packaging forms. Pydantic and deterministic evidence checks ran on actual saved model replies. Schema probes found a counterexample for `uniqueItems`. The production validator checks evidence independently and retains both attempts when it uses a repair. [Raw experiment and runner metrics](run-001/summary.json) · [Offline validation of saved replies](run-001/revalidation.json) · [Pinned bytes](../integration/ollama-artifacts.json)

The notes below record the specialist's earlier transfer and mock-backed implementation work.

## Initial transfer blocker
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

## Update — finding 11 (deterministic unit conversion) and runner lifetime; mock-backed, no model loaded, no downloads, no commits
**Decision (finding 11).** On a *resolved* correction (target linked, not ambiguous/mismatched) whose unit is a verified physical unit of the SAME
dimension as the corrected observation (mass g/kg, volume ml/cl/l) but differs, `quantity` and `quantity_max` are converted with exact Decimal arithmetic
into the corrected observation's unit (`validate._harmonize_unit`, `units.convert`) and `unit` becomes that unit. `quantity_text` and `unit_spoken` keep the exact
spoken words, qualifier (range/approximate) is kept, the earlier observation is untouched (e.g. 3 kg retained; "Non, cinq cents grammes" -> 0.5 kg, unit_spoken
"grammes"). Chains follow the head unit; replays are idempotent (same evidence => same ids; already-extracted evidence is skipped, so nothing converts twice).
Nothing is converted for counts, packages (carton/bouteille/...), unknown or missing units, or across dimensions (carton vs kg, l vs kg stay unlinked
`correction_target_unit_mismatch` + needs_review). Original error statuses are preserved (e.g. model quantity disagreement still needs_review). Each applied
conversion is listed in `ExtractionResult.metrics["unit_conversions"]` (not an error). Ordinary non-correction observations keep their spoken unit.
**Runner lifetime.** `extract()` unloads (keep_alive 0) a runner it created itself when the call ends (normal, error, or exception, with a note on the exception),
only if a chat request was attempted, then polls /api/ps for up to 5 s (`UNLOAD_POLL_SECONDS`); outcome in `metrics["unload"]` = {attempted, ok, error,
idle_confirmed}, separate from `errors`/status. Caller-injected backends are never unloaded; no chat sent (no_speech, digest mismatch, model missing, nothing fits)
=> no unload call. The model stays warm across all windows/repairs of one call. `tools/extraction/probe.py` now injects a shared backend (with `expected_digest`)
so fixtures stay warm and the probe unloads explicitly once.
**Commands/results.** `.venv/bin/python -m pytest -q tools/extraction/tests tests/persistence_own` = 196 passed; `tests/qa` = 40 passed;
`tests -m "not real_models and not microphone"` = 100 passed, 12 deselected; R05 reproducer exit 0. No failures.

The second pinned production probe passed all five required cases with prompt 4 and generation schema 4. Both flour corrections preserve the earlier count and link the revised observation. Two unspecified cartons keep their contents unknown, while two cartons of twelve bottles yield 24 bottles. The local runner unloaded. [Raw generation and production records](run-002/summary.json)

The separate installed-CLI evaluation and its independent QA rerun matched all 15 observation sets. The curated reference set had 16 gold observations and three additional reviewable candidates. [Independent QA evidence](../qa/real-qwen-independent.md)
