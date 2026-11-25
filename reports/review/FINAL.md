# Independent release review (final audit)

## Reviewer and base

- **Who ran it:** Claude Code 2.1.329, process `5281`, launched by Orca with `--model=claude-opus-5-5 --effort=high`. All response records in provider session `2061ab3a-a0af-4592-bb5b-8e25df20eb8a` report `claude-opus-5-5`.
- **Orca identifiers:** session `7d26416e-…`. Initial audit dispatch `ctx_6fb7cf719a3c`. Release audit `ctx_002f36402c51`, task `task_572cb37073c2`.
- **Source audited:** HEAD `ae495c8`. Its only change after `6b083c8` is in reports and tests; `src/` is identical.
  - Implementation SHA-256, recomputed as in `pipeline._provenance`: **`e57cfade…1079`**.
  - Snapshot: `final/source-snapshot-final.sha256`.
- **What this audit did not do:** load model weights, open the microphone, play audio, download anything, or edit production code or tests. `r10` reads only the pinned `tokenizer.json`.

## Acceptance command

`.venv/bin/python -m pytest -q tools/extraction/tests tests/persistence_own tests/qa`

**Result: 297 passed, EXIT=0** (`final/acceptance-final.txt`).

## Reproducers

All are in `repro/`, each with its raw `.out` file. The final reruns are in `final/*.final.out`. Every one exited 0.

| Check | What it shows | Status |
|---|---|---|
| r01 numbers | French number words, decimal comma, ranges, approximations | PASS |
| r02 semantics | Required sentences, both correction paths, carton contents unknown, carton total 24, negation and comparative cases, ambiguous corrections | PASS (current source flags more for review; all expected cases hold) |
| r03 / r03b | Retry after a transient extraction failure | FIXED. Both now `replaced`, including a complete retry that is itself `needs_review` |
| r04 | Mic import and IPC exit codes | Historical. `capture_microphone` now exists |
| r05 | Boundary re-emission flags, duplicate detection inside one window, language check | FIXED, except partial-word re-emission, which is still missed (low) |
| **r06** | Near-cap Hojo guard (no model) | **OPEN, material.** Reported to the coordinator, who routed it to the audio engineer |
| r07 | Read-only SQLite against run.json and CSV for the offline file and fixture-mic runs | CONSISTENT: integrity ok, content digests match, evidence slices match |
| r08 | Which implementation produced each piece of runtime evidence | See section 3 |
| r09 / r11 | CPU vs MPS replies on identical inputs | **OPEN, material.** MPS corrupts the first words of every segment |
| r10 | Archived real replies retokenized against the cap guard | Only the real 200-token reply (CPU long segment 1) is flagged. No archived real reply sat unflagged at a cap |

## 1. Open material findings

1. **MPS garbles the first word or two of every segment.** *(The count below was corrected in Addendum A: 4 of 4 paired inputs, plus 2 unpaired M4A replies.)*
   - **Evidence:** `r11`, `r09`, from archived outputs. 6 of 6 MPS replies differ from CPU at the start:
     - Jacques clip: CPU "il reste deux…", MPS "Présence de deux…" (both MPS iterations).
     - Long file: CPU "je compte huit kilos", MPS "ACS huit kilos"; CPU "les étagères de la", MPS "selectors de la réserve"; MPS "توقع refrigerateur je compte".
     - M4A: MPS "bfs pour continuer…", "Présentement que…".
   - The CPU/MPS pairs used the same sample ranges.
   - **Why it matters:** an utterance that starts with its quantity can lose or change it on MPS and still be marked `ok`. The current language check returns `fr` for the segment that begins with Arabic script, and the audio stage has no script check.
   - **Claim affected:** `hojo_mps: PASS` describes this honestly but should be qualified as functional with a systematic accuracy defect. Keep CPU as the default backend. Cause not yet diagnosed. *(Addendum B rules out the first decoder prefill on identical embeddings. The native MPS audio/embedding path and later beam steps remain open.)*

2. **The near-cap guard leaves silent-truncation gaps.**
   - **Code:** `audio/hojo.py:21-45` raises `needs_review` only when there are at least 3 exact copies of a block of at least 16 tokens, at 196–204 tokens, measured against a fixed cap of 200.
   - **Gaps shown by `r06`:**
     - (a) A non-repeating reply that fills the cap is marked clear. That is possible truncation, and the wheel gives no stop reason.
     - (b) The wheel's actual cap is `min(200, 2*feat_len+10)` (`hojo_asr_model.py:255-258`). This is lower for segments shorter than about 8 s (about 110 tokens at 4 s, derived assuming 12.5 Hz embeddings). A loop truncated at that lower cap is marked clear.
     - (c) A short-period loop that fills 200 tokens is marked clear.
   - **Real data (`r10`):** CPU long segment 1 reached 200 tokens and is flagged. Dense segments run at about 5.6 tokens/s (140 tokens in 24.8 s), against 8 tokens/s at the 200-token cap for 25 s, so (a) is a plausible risk.
   - Status: **CLOSED at implementation `23096a70`.** See Addendum A.

## 2. Original findings and what happened to them

| # | Original finding | Disposition | Evidence |
|---|---|---|---|
| 1 | Mic mode not implemented | **CLOSED.** `microphone.py`: bounded callback queue, bounded ASR queue, controlled stop on saturation, retained spool, provisional JSONL kept apart from finalized records, final flush. Physical gates are **BLOCKED** by the user's choice (file input only) | source; `reports/offline/fixture-mic.json` |
| 2 | Hojo override unguarded | **CLOSED.** Wheel `load_model` uses `cls(...)` (`hojo_asr_model.py:365`). The adapter now asserts the hooks exist, `isinstance(LocalHojo)`, that the methods are bound, and the device. Real runs logged `adapter_hooks {autocast_context: 2, encode_speech: 1}` on CPU and MPS | `audio/hojo.py:90-129`; `reports/hojo/{cpu,mps}.jsonl` |
| 3 | Partial-retry lock-in | **CLOSED.** Including the residual case where the retry is itself `needs_review` | `final/r03*.final.out` |
| 4 | Boundary re-emission / VAD settings | **MOSTLY CLOSED.** A match now needs 2 or more words. Duplicates inside one window are flagged. VAD minimum silence is now 350 ms. Partial-word re-emission is still missed (low) | `final/r05…` |
| 5 | Language trusted from the model | **CLOSED** for whole sentences in another language. **New gap:** a foreign-script prefix inside French text still passes (see 1.1) | `final/r02`, `r05`; `r09` |
| 6 | Ollama version missing from processing version | **CLOSED** | `pipeline.py:114-118` |
| 7 | Offline verifier labels / controls not reaching the child | **CLOSED.** FAIL vs BLOCKED are now separated. The CLI, the dedicated `ollama serve` and the ASR child (`parent_pid` = CLI pid) each received EPERM for external connections and HTTP 200 on loopback. The instrumentation is injected through `tools/offline/instrument/sitecustomize.py` and does not live in `src`. The Ollama runner subprocess is not probed itself; it inherits the sandbox through the OS (low) | `reports/offline/positive-file.json`, `fixture-mic.json` |
| 8 | 11.96 GB weight re-hash on every run | **OPEN (low, disclosed).** Still done every run, about 5.4 s measured. This is included in pipeline RTF | `audio/hojo.py:51`; `long-cpu-bisection.jsonl` |
| 9 | One ASR chunk error blocks extraction for the whole run | Unchanged, but visible (low) | source |
| 10 | Test-order failure | **CLOSED** (297 passed) | `final/acceptance-final.txt` |
| 11 | `units.convert` never used | **CLOSED.** Used when resolving corrections (`validate.py:502`) | source |
| 12 | Exit codes / repository / privacy | **CLOSED.** A missing imageio-ffmpeg now raises `ImportError` (exit 3). FLEURS restore is verified against hashes. No WAV, SQLite or output files are tracked in git. A credential scan of reports, docs, src, tools, tests, examples and evaluation found nothing | greps |

## 3. Do the acceptance claims hold?

**Claims that hold.**

- `offline_file_inference` and `fixture_microphone_real_models`: the controls and the committed data are consistent (`r07`).
- `validated_exports_and_replay`: the replay evidence shows `inserted` then `duplicate` on both backends, with byte-identical views.
- `french_wer`: 2 edits in 61 words over 20.46 s. The report states it does not estimate wider accuracy.
- `long_boundary_and_no_loss` and `spoken_negation` are honestly marked FAIL.
- Physical mic and user-file gates are honestly marked BLOCKED.

**Problem 1: most runtime evidence was produced by older code.**
`r08` recomputed the implementation hash behind each runtime evidence file. Only these come from the current source (`e57cfade`):
- `fleurs-guarded-final`
- `synthetic-guarded-final`
- `semantic-audio-reviewed`
- `guarded-wheel`

These were produced by earlier implementations:

| Evidence | Implementation |
|---|---|
| Offline `positive-file` | `94a4235` |
| `fixture-mic` | `acf2f06` |
| M4A CPU run | `009762f` |
| M4A MPS, CPU/MPS replay, `synthetic-real` and `synthetic-mps` | `f4744f5` |

Once the source is frozen after the R06 and MPS fixes, every runtime PASS needs one of two things: a rerun on the frozen source, or an explicit argument that the code differences do not affect it.

**Problem 2: some PASS labels need qualifying.**
- **`hojo_mps`:** systematic leading-word defect (finding 1.1).
- **`microphone_and_backpressure`:** the ASR step was injected, and the input was fixture callbacks, not a real device.
- **`field_scores`:** this is the development set, so it is not a held-out score.
- **`runtime_performance`:** the 3.57 s clip runs slower than real time on both backends (CPU RTF 1.9–2.5, MPS warm 1.89). No real-time claim should be made.

## 4. Still open or blocked

- **Open defects:** findings 1.1 and 1.2.
- **Blocked by the user's choice:** physical microphone, live queue behaviour and end-of-speech latency, and the user's own audio file (none has been supplied).
- **Pending:** the post-freeze evidence. Once it lands, re-run `r06`, `r08`, `r10` and `r11`, and compare the new raw MPS replies to CPU.

## Addendum A — 2026-11-25: R06 recheck and R11 count at implementation `23096a70`

**Base.** Working tree on HEAD `ae495c8` with uncommitted changes in `src/speechflow/audio/{hojo,__init__}.py`.
- Implementation SHA-256, recomputed: `23096a70df38…e867`. This matches `reports/hojo/generation-cap-source.sha256`.
- Reviewer: same session and process (`--model=claude-opus-5-5 --effort=high`).
- No model, device or download work was done.
- Acceptance command: **297 passed, EXIT=0** (`final/acceptance-23096a70.txt`).
- Engineer's `tools/hojo/test_generation_loop.py`: **17 passed, EXIT=0** (`final/r06-engineer-tests.txt`).

### R06: CLOSED

**What the new code does.** `infer_wav` routes each call through `LocalHojo._run_observed_wav`. That function wraps `decoder_model.generate` for one call and restores it afterwards. It records three things:
- the cap actually used for that call (`max_new_tokens`),
- the returned `(1, L)` output length,
- whether the last token is EOS.

`_infer_segment` then runs both `generation_cap_warning` and `generation_loop_warning`. If the observation is missing, the code falls back to the per-call feature cap and a retokenized ±4-token proxy.

**My independent reproducer** is `repro/r06b_entry_path_cap.py` with `.out`, EXIT=0. It goes through the real `_infer_segment` → `infer_wav` → `_observe_wav_generation` path. The model is a weightless fake that mimics `hojo_asr_model.py:236-266`.

| Case | Status |
|---|---|
| 110 of 110 tokens, no repetition, no EOS | `needs_review` (cap reached) |
| 110 of 110 tokens ending in EOS | `needs_review` (conservative) |
| EOS early, at 60 of 110 | `ok` |
| Period-4 loop filling 110 | `needs_review` |
| 16-token block loop filling 110 | `needs_review`, both warnings |
| 200 of 200 tokens, no repetition | `needs_review` |
| 140 tokens with EOS (normal dense speech) | `ok` |
| 1 s input with cap 34, 34 tokens | `needs_review` |
| Observation unavailable (two decoder calls) at 110 tokens | `needs_review` (proxy) |
| Observation unavailable, 60 tokens | `ok` |

**The original helper.** `repro/r06_generation_cap.py`, re-run as `final/r06_generation_cap.23096a70.out`, still returns `None` for some of its cases. That is expected: it calls only the loop function, which is no longer the guard that closes these cases. The original `.out` is kept unchanged.

**Genuine receipts on the pinned weights** (`reports/hojo/generation-cap-{cpu,mps}.jsonl`, exits 0, source `23096a70`):

| Input | Backend | Feature frames | Cap | Output tokens | EOS | Result |
|---|---|---|---|---|---|---|
| Jacques, 3.57 s | CPU | 45 | 100 | 23 | present | clear |
| Jacques, 3.57 s | MPS | 45 | 100 | 25 | present | clear |
| Long-file centre segment, 25 s | CPU | 313 | 200 | 200 | absent | flagged: cap reached and 48-token loop |

The 45 frames for 3.57 s confirm the 12.5 Hz derivation used in R06.

**Remaining gaps, low.**
- On MPS, the cap-reached branch has been exercised only on scripted output, not on a real cap hit. The counting mechanism is shared with CPU.
- The observation (count, cap, EOS) is used for the warning but is not stored in the segment's `raw` field or in the manifest metrics. Persisting it would make the review flag auditable after the fact.

### R11: count corrected and qualified

**Paired inputs.** Over **four distinct paired WAV inputs**, every MPS reply changes the leading word(s) where CPU on the same sample range does not:
1. Jacques, 3.57 s. Repeated short calls count once.
2. Long-file segment 0.
3. Long-file segment 1.
4. Long-file segment 2.

**Unpaired.** The two M4A-MPS replies (`bfs…`, `Présentement…`) came from a different file with no CPU pair. They show the same pattern but are not paired evidence. The "6/6" in section 1.1 should be read as 4/4 paired plus 2 unpaired.

**The genuine receipt shows that the opening IDs of the selected four-beam winner differ.** *(Corrected in Addendum B. This does not establish that the first decoding step itself differs. On matched embeddings, CPU and MPS give the same first-step ranking.)*

| Backend | First three token ids | Spellings |
|---|---|---|
| CPU (SDPA attention) | `[321, 65067, 25552]` | `il`, ` reste`, ` deux` |
| CPU forced to eager attention | `[321, 65067, 25552]` | `il`, ` reste`, ` deux` |
| MPS | `[2340, 5397, 763]` | ` Pr`, `és`, `ence` |

The CPU-eager row rules out eager attention by itself as the cause.

**Dispositions.**
- **CPU** stays the default backend.
- **MPS functionality: PASS.** The model loads on the device, keyed replies come back, the outputs are stable on replay, and the run is EPERM-isolated offline.
- **MPS leading-word fidelity: FAIL.** This is a separate gate from functionality.
- **No universal trim fix.** The defect replaces whole words ("il reste" becomes "Présence de"). It does not just add a removable prefix, so trimming leading tokens would also delete real speech such as a leading quantity.

### Evidence identity (`final/r08_evidence_source_identity.23096a70.out`)

The guarded FLEURS, fixture and M4A reruns (`fleurs-guarded-final`, `fixture-guarded-final-manifest`, `m4a-guarded-final-manifest`) were produced by `e57cfade`. That is before the R06 change. Only `generation-cap-{cpu,mps}.jsonl` were produced by `23096a70`. The R06 change affects only segment review status, but the final CPU QA, MPS and installed-release runs still need to land on `23096a70` before the runtime gates can cite the current source.

## Addendum B — 2026-11-25: matched-embedding prefill (R11 cause narrowed, disposition unchanged)

**Sources.**
- Raw: `reports/hojo/generation-prefill-{cpu,mps}.jsonl`. Both children exited 0, each after one decoder forward (`generation-prefill-exits.log`).
- Method: `reports/hojo/generation-loop.md`, section "Identical-embedding prefill".
- I re-read the raw fields myself. I ran no inference and changed no source.

**Inputs were identical.**
- The MPS run loaded the CPU run's saved BOS-plus-speech embeddings for the Jacques clip.
- The prepared tensors have identical SHA-256 hashes on both devices:
  - `inputs_embeds` `2a65e85a…` (shape 4×46×2560, bf16),
  - `attention_mask` `48a63e86…`,
  - `position_ids` `deb234fd…`,
  - `cache_position` `94b2606d…`.
- Both devices used the eager bf16 decoder.

**First-step result was the same.**
- On all four beams, both devices rank the same top eight IDs: `321 8669 263 3756 73 98934 295 88`, which spell `il ils on je j reste et y`.
- The top three raw logits are the same on both devices: 24.125, 18.000, 16.625.
- The rank-1 minus rank-2 margin is 6.125 on every beam on both devices.
- The full logit tensors are not identical (logit SHA-256 differs; some lower-ranked values differ). Those differences do not change the top eight.

**Correction to Addendum A.** I wrote that the receipt "puts the change at the very first decoding step". That overstates the evidence. The receipt shows only that the *selected four-beam winner* has different opening IDs: MPS `[2340, 5397, 763]` against CPU `[321, 65067, 25552]`. This prefill test shows that, on identical embeddings, positions and masks, the first decoder forward ranks `il` first on MPS exactly as on CPU, with the same margin.

**Where the divergence can still come from:**
- the native MPS audio and embedding path (encoder, bottleneck and normalisation on MPS before this point), or
- later beam steps and the KV cache.

The probe tests only the first prefill. The cause is still not diagnosed.

**Dispositions are unchanged.**

| Item | Status |
|---|---|
| R11, MPS leading-word fidelity | **FAIL**: 4 of 4 paired inputs in native end-to-end MPS runs |
| MPS functionality | PASS (separate gate) |
| Default backend | CPU |
| Trimming leading tokens as a fix | Not a universal fix |
| R06 (cap guard) | **CLOSED** at `23096a70` (Addendum A) |
