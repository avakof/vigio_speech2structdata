# Final independent real-model QA

The CPU suite used the installed CLI and pinned local Hojo, Silero and Ollama
Qwen. The separate MPS replay used the same pinned short Jacques WAV and the
same tests with `backend = "mps"` in a private TOML. The tests ran serially.

## CPU suite

Pytest exited **1** in **690.03 s** with **10 passed, 2 failed and 1
deselected**. The seven codec and Silero cases, short-file replay,
15-case exact-transcript Qwen evaluation and Seatbelt file test passed.
Both long-file content tests failed. The physical microphone test was
deselected by the default marker expression. [Full command and pytest log](final-real-cpu.log)

The exact-transcript CLI evaluation took 119.80 s, covered all 16 gold
observations and had zero false positives and false negatives in every
predeclared field. All 15 review decisions matched. Its raw report is at
`outputs/qa-final-real-002/test_real_extractor_installed_0/evaluation.json`.
The Seatbelt file test measured 37.54 s of pipeline time and confirmed the
2.5 kg tomatoes and 12 milk bottles. The app and its dedicated Ollama server
each denied external IPv4 and IPv6 and reached loopback.
[Seatbelt receipt](final-real-offline-file.json)

The 325 process and default Ollama snapshots saw at most one CLI Hojo
child and zero samples with both a CLI Hojo child and loaded default
Ollama. These samples were taken approximately every two seconds. Ollama
reported an empty model list before and after pytest.
[Runtime samples](final-real-cpu-process.jsonl)

### CPU replay

The two independent CLI processes took 38.40 s and 33.76 s. Both
observed the pinned Qwen runner, succeeded, and returned Ollama to idle.
Replay resolved to the same run ID and processing version. SQLite retained
one generation, one 57,142-sample segment, two observations, two evidence
rows and two save events (`inserted`, `duplicate`). JSON, both CSVs and
both tables matched byte for byte across first run, replay and CLI export.
[Commands, raw manifests, SQL and view hashes](final-real-cpu-replay-evidence.json)

## Independent MPS replay

Pytest exited **0**, **1 passed in 87.74 s**. The two separate CLI processes
took 44.08 s and 43.10 s and observed the pinned Qwen digest. Both reported the
same MPS run ID and processing version. SQL retained one generation, one
segment, two observations, two evidence rows, and the `inserted` and `duplicate`
events. All five exports matched byte for byte across first run, replay and
independent export. [Full log](final-real-mps.log),
[commands, raw manifests, SQL and view hashes](final-real-mps-replay-evidence.json)

CPU consistently returned `il reste deux virgule cinq kilos de tomates et
douze bouteilles de lait`. MPS consistently returned `Présence de deux virgule
cinq kilos de tomates et douze bouteilles de lait`. Raw and normalized text
were identical within each backend, including between both processes. Both
backends used the same segment key, input `[0,57142)` and core `[1024,57142)`.
Both accepted 2.5 kg tomatoes and 12 milk bottles with no per-bottle capacity.
Their processing versions and run IDs differed by backend as expected. The
unchanged checker compares transcript evidence within each backend and
applies the same semantic, persistence and export criteria to both.

The 43 MPS samples saw at most one CLI Hojo child and zero samples with that
child and a loaded default Ollama model. Ollama was empty at 18:39:44 UTC
after the test. The exclusive slot was returned.
[Runtime samples](final-real-mps-process.jsonl)

## Provenance

Both runs used commit `a000660c0092522e02e603410be09d9f2176d893`. The same 37
production and test files had identical SHA-256 values before and after both
stages. The sorted hash list has SHA-256
`dd38d0a3d48e63a2d572be675f50378b3729f7598ade850d97130a38992aca06`.
The short-file manifests report implementation SHA-256
`f4744f576ab739a2d0cb030f1e227d3949306b9282ef500127a8da85f144d1b8`.
The logs include the per-file hashes, installed package versions, the Hojo
revision, Silero SHA-256, Qwen artifact digest and Ollama version.
[CPU log](final-real-cpu.log), [MPS log](final-real-mps.log)

## Commands

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q -s tests/real_models \
  --basetemp outputs/qa-final-real-002

qa_config=$(mktemp -t speechflow-qa-final-mps)
chmod 600 "$qa_config"
printf 'backend = "mps"\noffline = true\n' > "$qa_config"
SPEECHFLOW_CONFIG="$qa_config" SPEECHFLOW_QA_MODEL_SLOT=1 \
  .venv/bin/python -m pytest -q -s \
  tests/real_models/test_file_replay.py::test_real_file_installed_cli_replay \
  --basetemp outputs/qa-final-mps-replay-002
```

## Long CPU file

The WAV has SHA-256
`03293fee0a0826dd6359087582be59770998a18df193808c6ebda6b249ee8021` and
1,159,360 samples at 16 kHz (72.46 s). The forced cuts at samples 394,112
and 787,712 produced Hojo inputs of 397,312, 400,000 and 374,848 samples.
The final core reached EOF and the final anchor, `deux kilos de gingembre`,
matched. Hojo used 398.98 s for ASR, or 5.51 times the audio duration.
The [raw Hojo return](final-real-long-hojo.json) keeps all text, normalized
text, segment keys, bounds and errors.

`test_real_hojo_long_audio_boundary_content_and_tail` failed at
`AssertionError: neuf kilos de pommes` (samples 486,400–511,200). Four of eight
gold anchors matched. Apples (30.40–31.95 s), pears (40.36–42.02 s),
kiwis (50.43–52.18 s) and plural lemons (60.59–62.32 s) missed. Raw text,
normalized text, keys and bounds of all three segments matched the earlier
run byte for byte. The middle segment now carries a `needs_review` warning
for 200 re-tokenized tokens and at least three consecutive copies of a
48-token block. Its repeated `cinq kilos de poissons` text is preserved.

`test_real_extractor_long_hojo_outputs_have_no_duplicate_observations`
failed at `assert result.status == "ok"` because the extraction returned
`needs_review`. Across two successful Qwen windows, five observations were
`ok`. Four matched gold (8 kg tomatoes, 4 kg carrots, 6 kg courgettes and
2 kg ginger), and the extra 4 kg lemon count remained `ok`. The 5 kg fish row
was flagged on ASR evidence from the repeated middle segment. The 7 kg lemon
candidate was flagged for an ambiguous correction target. The proof retains
both candidates and their evidence for review.
[Raw Qwen attempts and candidates](final-real-long-qwen.json),
[anchor and observation comparison](final-real-long-analysis.json)

The shortest retained input that contains both missed middle anchors is the
original 400,000-sample (25.00 s) Hojo window at `[390912,790912)`. The
[isolated-input receipt](real-full-long-reproducer.json) records the extracted
window and the unchanged reply from its original full-file invocation.
