# Independent release-candidate real-model QA

Implementation SHA-256
`23096a70df38af25f14f4275c9c0d0935f57d6d9a55ec421794a91906902e867`.
The CPU suite ran first, followed by the separately selected MPS replay.
The physical microphone test is excluded by the existing selection rule.
The run logs retain the pinned asset versions, source and test hashes, and
preflight and postflight Ollama snapshots. The [previous QA report](final-real-suite.md)
contains the earlier baseline for comparison.

## CPU result

Pytest exited **1** after **495.04 s** with **10 passed, 2 failed and 1
deselected**. The seven codec and Silero cases, short-file replay,
exact-transcript Qwen evaluation and controlled offline file test passed.
The two failures were the long Hojo anchor check and long extraction check.
[Complete CPU command log](release-real-cpu.log)

The exact-transcript CLI evaluation completed in 91.13 s. It recovered all
16 gold observations across 15 cases, with zero false positives and false
negatives in the predeclared fields and 15 correct review decisions. The raw
report remains at `outputs/qa-release-real/test_real_extractor_installed_0/evaluation.json`.
The controlled Seatbelt file run finished in 33.25 s of pipeline time with the
same two accepted quantities. The app and its dedicated Ollama process each
reached loopback and denied external IPv4 and IPv6.
[Controlled offline receipt](release-real-offline-file.json)

The 239 runtime snapshots saw at most one CLI Hojo child and zero samples
with both that child and a loaded default Ollama model. The default Ollama
runner was empty before the suite and at its 20:39:26 UTC exit.
[CPU process snapshots](release-real-cpu-process.jsonl)

## MPS result

Pytest exited **0** with **1 passed in 88.10 s**. Separate CLI processes took
43.07 s and 44.47 s and returned `inserted` and `duplicate` for the same run
and processing version. SQLite retained one generation, one segment, two
observations and two exact evidence rows. All five exports matched byte for
byte across initial run, replay and independent export.
[MPS command log](release-real-mps.log),
[commands, manifests, SQL and view hashes](release-real-mps-replay-evidence.json)

MPS consistently said `Présence de` where CPU said `il reste`. Segment key,
sample bounds and the two accepted inventory fields matched across backends.
Each backend retained its own run ID and its own stable raw and normalized
transcript. Read-only SQLite `quick_check` and `foreign_key_check` passed on
both databases. [Machine-readable release summary](release-real-summary.json)

The 43 MPS snapshots saw at most one CLI Hojo child and zero samples with
that child and a loaded default Ollama model. At 20:40:55 UTC Ollama reported
an empty model list and the model slot was returned.
[MPS process snapshots](release-real-mps-process.jsonl)

## Source integrity

The same 37 source and test hashes matched before and after each stage. The
SHA-256 of the sorted list is
`0a012f1dd0313513101e72624433e15c9f28232210fbe136c6a088edb5553709`.
The in-flight Git commit of the audio work changed HEAD from `ae495c8` to
`2716290`. Both preflights and all short-file manifests recorded the same
release implementation SHA-256 above. No test assertions were changed.

## Commands

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q -s tests/real_models \
  --basetemp outputs/qa-release-real

qa_config=$(mktemp -t speechflow-release-qa-mps)
chmod 600 "$qa_config"
printf 'backend = "mps"\noffline = true\n' > "$qa_config"
SPEECHFLOW_CONFIG="$qa_config" SPEECHFLOW_QA_MODEL_SLOT=1 \
  .venv/bin/python -m pytest -q -s \
  tests/real_models/test_file_replay.py::test_real_file_installed_cli_replay \
  --basetemp outputs/qa-release-mps-replay
```

A temporary pytest hook recorded the long-test return values from genuine
model calls. The unchanged assertions and all release tests ran their normal
paths. Each replay made two fresh installed-CLI processes and independent
JSON, CSV and table export calls against that backend's SQLite database.

## Short CPU replay

The two CLI calls completed in 37.89 s and 33.31 s. Both identified the
release implementation. The second call resolved to the same run and
processing version with disposition `duplicate`. SQLite had one
generation, one segment, two observations, two evidence rows and the two
save events `inserted` and `duplicate`. The expected 2.5 kg of tomatoes
and 12 bottles of milk remained accepted. All five view files matched byte
for byte across the first run, replay and independent export.
[Exact commands, manifests, SQL rows and view hashes](release-real-cpu-replay-evidence.json)

## Preserved long-file failure

Hojo again failed the first missed anchor, `neuf kilos de pommes`. All three
raw and normalized transcripts, keys and sample bounds matched the previous
run. Four of eight gold anchors matched and the final ginger anchor and EOF
were reached. The latest receipt identifies 200 of 200 actual decoder
output tokens with terminal EOS absent on the looping middle chunk, and
retains the repeated raw words and loop warning.

The long extraction assertion again received `needs_review` where the test
requires `ok`. Four exact gold rows were accepted and the extra 4 kg lemon
row was still accepted. The fish row and the ambiguous 7 kg lemon revision
stay under review. [Raw Hojo return](release-real-long-hojo.json),
[raw Qwen attempts and fields](release-real-long-qwen.json),
[anchor and observation comparison](release-real-long-analysis.json)
