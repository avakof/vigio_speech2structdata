# Independent review: initial implementation audit

**Reviewer:** Claude Code 2.1.329, `--model=claude-opus-5-5 --effort=high`. This comes from the process arguments of PID 5281, which Orca launched through `orca-provider-exec`. All 102 response records in provider session `2061ab3a-a0af-4592-bb5b-8e25df20eb8a` report `"model":"claude-opus-5-5"`. Orca session `7d26416e-e5ee-4c68-bb42-5cca16f4830e`, task `task_8c9fa970b29e`, dispatch `ctx_6fb7cf719a3c`, local checkout.
**Scope and constraints:** The audit read the source, docs and reports. It loaded no models, started no downloads and opened no microphone. The reproducers call production code with hand-written model replies and manifests. They live in `repro/`, and each has its raw `.out` file next to it. The source changed while the audit was running. Snapshots are in `source-snapshot-initial.sha256` (2026-11-25T14:46Z). Every reproducer was re-run after the integrator's persistence fix.

## Acceptance command

`.venv/bin/python -m pytest -q tools/extraction/tests tests/persistence_own tests/qa`

| Run | Result | Raw output |
|---|---|---|
| Initial snapshot | 152 passed, **EXIT=0** | `acceptance-initial.txt` |
| After the new tests/qa files and fixes | 176 passed, 1 failed, **EXIT=1** | `acceptance-rerun.txt` and `acceptance-rerun2.txt` |

The single failure is a test-isolation bug, not a product defect (see finding 10). The failing test is `test_own_cli.py:196-199` and it passes when run alone.

## Findings, ranked

Label key: **OBSERVED** means a reproducer ran here and showed the failure. **SOURCE** means it comes from reading the code and could not be executed with the current dependencies.

1. **HIGH, OBSERVED: microphone mode is not implemented.**
   - `asr_process.py:12` imports `capture_microphone`, but `audio/__init__.py` does not define it. `__all__` is `["transcribe_file"]`.
   - `speechflow mic --seconds 0.1` exits with code 3. Its message is "install dependencies with `uv sync --frozen`", which is wrong: no dependency is missing.
   - The README (lines 65 and 83-89) and `docs/contracts.md` describe the mic interface as if it exists. `doctor` requires sounddevice.
   - Gate D (callback, bounded buffer, endpointing, backpressure, flush, provisional vs. finalized records) has no code.
   - `acceptance.json` should record this as FAIL (not implemented), not as NOT RUN gated on file evidence.
   - Evidence: `repro/r04_mic_ipc.out`.

2. **HIGH, SOURCE (pending): the Hojo MPS adapter's overrides are unverified and have no guard.**
   - `audio/hojo.py:60-71` subclasses `HOJO_ASR` and overrides `autocast_context` and `encode_speech`.
   - The pinned wheel is not installed in `.venv` or `~/.cache/speechflow/hojo-env`, so the method names cannot be checked against the base class yet.
   - The overrides do nothing (no error) in either of these cases:
     - the wheel's `load_model` builds `HOJO_ASR(...)` rather than `cls(...)`;
     - a method name differs from the one the wheel actually calls.
     
     MPS would then run under CUDA autocast or hit a float32/bf16 mismatch, while the docs still claim the fix applies.
   - Add `assert isinstance(model, LocalHojo)` and `callable(getattr(HOJO_ASR, name))` checks for both names. Record proof in the MPS benchmark that the override actually ran (for example, a call counter).
   - `cuda_enabled = model.device.type == "mps"` (line 87) and `model.device` also depend on wheel behaviour that has not been checked here.

3. **HIGH → MEDIUM, OBSERVED: a stored partial run could not be replaced by its retry. The integrator's fix covers only part of it.**
   - **Before the fix:** one window that timed out produced a `needs_review` run. Its complete retry got `ConflictError`, so the degraded result stayed permanently. Evidence: `repro/r03_transient_lockin.out`.
   - **After the integrator's fix:** I confirmed the case is now `replaced` and stored as `ok`. Evidence: `repro/r03_transient_lockin.after-fix.out`.
   - **Residual:** `persistence.py:386-392` lets a retry replace the stored run only when the new status is `ok`. A complete retry that is itself legitimately `needs_review` is still rejected with `ConflictError`. Evidence: `repro/r03b_partial_retry_residual.out`.
   - Real runs will often be `needs_review`, because of `unit_missing`, the ASR overlap flags (finding 4) and similar causes, so the fix covers a narrow case. Suggested rule: allow replacement when the stored save recorded failed windows or transport errors, the retry keeps every earlier `ok` observation, and it has fewer failures.
   - **Pending real evidence:** run the same file twice on CPU and on MPS. If Hojo or Ollama is not deterministic, every replay ends in `ConflictError`. That fails loudly, but replay would then never be idempotent.

4. **MEDIUM, OBSERVED: duplicate handling at chunk boundaries depends on word matching and on how windows are packed.**
   - **Audio level.** `audio/__init__.py:57-73` `_flag_reemissions`:
     - It sets both neighbouring segments to `needs_review` when a single word repeats across the boundary, including legitimate repeats such as `de` or `lait`.
     - It misses partial-word re-emissions such as `bouteil` / `bouteilles`.
     - Evidence: `repro/r05_overlap_language.out` (a1-a3).
   - **Extraction level.** `validate.py:172` checks for padding-overlap duplicates only against `already` (earlier windows), not against observations created in the same reply.
     - Identical text in overlapping padded inputs inside one window: both are accepted as `ok`.
     - The same pair split across windows: the second is flagged.
     - Evidence: (b) and (b').
   - **Segmentation settings make overlaps frequent.** `segmentation.py:14` uses `VAD_MIN_SILENCE_MS = 100` with 200 ms padding on each side. Neighbouring ASR inputs therefore overlap at every pause shorter than 400 ms, which is common in speech. Expect many short Hojo inputs and many review flags.
   - The requirement that boundary phrases survive without duplicate observations is unproven. `long_boundary_and_no_loss` is NOT RUN.

5. **MEDIUM, OBSERVED: language support depends only on the model's self-report.**
   - `validate.py:115` and `:198` take the reply's `language` field at face value.
   - With `language: "fr"`, both of these are accepted with status `ok`: English "3 kilos of flour" and Italian "3 kili di farina".
   - Unknown units such as `kili` pass without being flagged, because `normalize_unit` returns any text unchanged.
   - Evidence: `repro/r05_overlap_language.out` (c, c').

6. **MEDIUM, SOURCE: the processing version leaves out the Ollama runtime version.**
   - `pipeline.py:71-97` hashes the code, config, Python packages, Hojo revision and the *expected* extractor digest.
   - It does not include the server version, which `/api/version` reports (0.40.16 here).
   - After an Ollama upgrade, the same processing version can produce different output, so replays fail with `ConflictError` instead of creating a new version.

7. **MEDIUM, OBSERVED (methodology): the offline verifier mislabels outcomes.**
   - `tools/offline/verify.py:74` defaults to `"BLOCKED"`, and `:105-109` gives PASS only when the run status is `ok`.
   - `reports/offline/file-attempt-1.json` therefore records an application crash (`ModuleNotFoundError`, exit 1) as BLOCKED instead of FAIL.
   - A `needs_review` run that was genuinely offline would also be labelled BLOCKED. That mixes up network isolation with functional success.
   - The network controls probe only the process that is then exec'd. The ASR child and `ollama runner` inherit Seatbelt under OS rules but are never probed. Add a denied-connect probe inside the ASR child, or an `lsof -i`/`nettop` capture taken during the run.
   - `offline_control_app_and_daemon: PASS` is acceptable only as evidence that the controls work. It must not count towards offline inference.

8. **MEDIUM, SOURCE (performance): every transcription re-hashes the full Hojo weights.**
   - `audio/hojo.py:20` calls `hojo_issues(verify=True)`, which computes a SHA-256 over the 11.96 GB safetensors on every `transcribe`/`run` and every future mic session.
   - This time is included in `pipeline_seconds` and `pipeline_rtf`.
   - Measure and disclose it, or cache the verification keyed on (device, inode, size, mtime).

9. **LOW, SOURCE: one failed ASR chunk blocks extraction for the whole run.**
   - When any segment has an error, `audio.status` becomes `error`, and `pipeline.py:114-116` then skips extraction for every segment.
   - The failure is visible and the run can be replaced, but a long file loses all of its observations.

10. **LOW, OBSERVED: the acceptance suite fails depending on test order.**
    - `test_own_cli.py:197` sets `sys.modules["speechflow.pipeline"] = None`. That has no effect once an earlier test has imported the module, because `tests/qa/test_evaluation.py` imports `speechflow.evaluation`, which imports `pipeline`. `from . import pipeline` then resolves to the package attribute.
    - The real pipeline runs instead and returns exit code 4, where the test expects 3.
    - Fix: also `monkeypatch.delattr(speechflow, "pipeline", raising=False)`.

11. **LOW, OBSERVED: gaps in units and number phrases.**
    - "un kilo et demi" and "deux kilos cinq" are not parsed. The observation goes to `needs_review`, which is the safe outcome. Evidence: `r01`, `r02`.
    - `units.convert` is never called, so no deterministic unit conversion happens anywhere, although the requirement asks for it.

12. **LOW: exit codes, privacy and the repository.**
    - A missing `imageio_ffmpeg` exits with 4 (input error) instead of 3 (dependency). Evidence: `r04`.
    - `.gitignore` excludes `*.wav`, so the WAVs that the evaluation manifest refers to are not in a clone. Confirm that `evaluation/fleurs-fr/prepare.py` re-fetches them and checks the pinned hashes.
    - Reports contain absolute paths that include the home directory.
    - A credential scan of reports, docs, src, tools, examples and evaluation found no tokens.

## Verified as correct here (validator level, scripted replies, `repro/r01` and `r02`)

- **French numbers:** decimal commas, number words (soixante-dix, quatre-vingt-dix-sept, mille deux cents), "1 000", ranges, environ / à peu près, "une dizaine" as approximate.
- **Exact transcripts:** tomatoes 2.5 kg and milk 12 bottles (no volume invented). Same-segment and next-segment corrections give revision 2 linked to the superseded value.
- **Packaging:** "Deux cartons de lait" keeps contents unknown. "Deux cartons de douze bouteilles" gives a total of 24 bouteille (product missing, so needs_review).
- **Negation:** "plus de dix" is treated as a comparative, not a negation. "ne … que" is not a negation. An ambiguous bare correction is left unlinked.
- **Injection:** the injection text is treated as data.
- **Duplicates:** two identical utterances 3 s apart are both kept as distinct observations.
- **IPC:** exit-code mapping works (`.ogg` gives 4).
- **Output safety:** the persistence layer neutralises CSV formulas and escapes terminal control characters.
- **Ollama endpoint:** only numeric-loopback addresses are accepted, with no proxy and no redirects.

## Is the acceptance report claiming more than its evidence?

- **`validated_exports_and_replay: PASS`** uses fixture manifests only. Finding 3's residual shows that replay after a partial failure is still not fully recoverable. Mark it as qualified or PARTIAL.
- **`reference_semantics: PASS`** comes from scripted replies. It says nothing about how well qwen3:8b extracts; keep that label explicit.
- **`microphone_and_backpressure: NOT RUN`** should be FAIL, because the feature is not implemented (finding 1).

## Hardware and metrics notes

- `asr_realtime_factor` divides by the duration of the *whole source*, silence included.
- `pipeline_rtf` includes child start-up, the weight hash and the model load.
- `ru_maxrss` is in bytes on macOS, so the units are correct.
- The pipeline has no MPS or unified-memory metric. Only `tools/hojo` has one.
- The pipeline does not measure mic endpointing, queue latency, or the delay from end of speech to final output.

## Still needed from the real runs for the final follow-up

1. Wheel installed, then the finding 2 checks (`isinstance`, override names, proof the override ran on MPS).
2. Raw logs from the CPU and MPS Hojo runs.
3. Two identical file runs per backend, to test determinism and replay.
4. A forced-boundary run longer than 60 s, with segment counts and flag rate on FLEURS.
5. qwen3:8b digest and real extraction attempts.
6. An offline run with a probe inside the child process.
7. The mic implementation, then its gate. Until then that gate stays FAIL.
