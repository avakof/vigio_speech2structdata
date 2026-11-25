# speechflow

Local French speech → structured inventory data. Audio file or microphone →
normalization → Silero VAD → **HojoAI/Hojo-ASR-Multi-V1** → local extraction
(Ollama `qwen3:8b`) → deterministic validation → JSON, CSV / terminal table and
SQLite. Inference needs no cloud account; cloud coding agents are development
tools only. See `docs/contracts.md` for the frozen interfaces.

Target machine: Apple Silicon Mac (developed for M3 Max / 48 GB), macOS, Python
3.12. The lock file only resolves for `darwin` / `arm64`.

The installed CLI on the M3 Max transcribed French speech, extracted inventory,
and saved the results to SQLite and matching exports with external networking
blocked. CPU and MPS both ran the pinned Hojo checkpoint. The default CPU
backend reproduced the full reference sentence in the short synthetic
inventory test. [Acceptance report](reports/acceptance.json) · [Backend measurements](reports/compatibility.md)

On the same French inputs, MPS sometimes changed or inserted words at the
start of a segment, including one Arabic-script prefix. A backend can execute
successfully while an exported count still needs comparison with the recording.
Both backends recovered four of eight expected inventory phrases in the 72.46-second synthetic test. On a separate spoken negation, CPU Hojo lost the product name. These failures leave overall acceptance incomplete. [Paired transcripts](reports/review/repro/r11_mps_leading_tokens.out) · [Acceptance by gate](reports/acceptance.json)

## Setup

### 1. Install dependencies (network, no models)

```sh
uv sync --frozen          # exact versions from uv.lock; never re-resolves
source .venv/bin/activate # or prefix commands with: uv run --no-sync
```

Install [Ollama](https://ollama.com) separately (app or `brew install ollama`)
and keep its service running; speechflow only talks to it on a numeric loopback
address.

### 2. Check the machine (no network, downloads nothing)

```sh
speechflow doctor
```

Once the CLI is installed, `doctor` reports missing dependencies, assets and
services and exits `3` while setup is incomplete. `speechflow --help` needs
only the standard library.

### 3. Prepare model assets (network, large download)

```sh
speechflow models prepare
```

This fetches the pinned Hojo revision (weights, tokenizer, configuration,
preprocessing files; the weight file alone is about 12 GB) into
`hojo_dir` (default `~/.cache/speechflow/assets/hojo`), verifies sizes, checks
the Silero VAD artifact shipped in the `silero-vad` package, and pulls the pinned
`qwen3:8b` through the local Ollama API. Re-running verifies and fetches only
what is missing.

After preparation, `speechflow --offline models prepare` and
`speechflow --offline doctor` verify locally and never download; missing assets
are listed, not fetched. `--offline` is a configuration flag, not proof of
isolation: demonstrate offline operation separately with external networking
blocked for the application.

## Commands

```sh
speechflow doctor
speechflow models prepare
speechflow transcribe path/to/audio.m4a --out outputs/run     # transcript only
speechflow run        path/to/audio.m4a --out outputs/run     # + observations
speechflow mic        --out outputs/live --seconds 60         # segmented mic transcription
speechflow evaluate   path/to/manifest.jsonl --out outputs/evaluation
speechflow export     RUN_ID --format csv                     # re-render a stored run
speechflow runs                                               # list stored runs, to find RUN_ID
speechflow recover    --out outputs/run                       # commit an interrupted save
```

Common options (accepted before or after the command; they override the file):

| option | effect |
|---|---|
| `--config FILE` | TOML settings (default `$SPEECHFLOW_CONFIG`, else built-ins). Unknown keys are errors. Template: `examples/settings.toml` |
| `--offline` | never download; report missing assets |
| `--backend cpu\|mps` | ASR backend |
| `--database FILE` | SQLite history (default `~/.local/share/speechflow/inventory.sqlite3`) |
| `--json` | one JSON document on stdout instead of text |
| `-q` / `--debug` | only the result line / tracebacks for unexpected errors |

`mic` also takes `--seconds N` (stop after N seconds, else until interrupted)
and `--device NAME_OR_INDEX`. `export` takes `--format json|csv|table`,
`--part observations|segments` (stdout) and `--out DIR` (write that format's
files). `RUN_ID` may be a unique prefix of at least four characters.

Microphone mode provides segmented transcription. Once the local stream opens
the CLI prints `Listening for microphone input.` to stderr. Press Ctrl-C once
to end capture and flush pending audio. Hojo processes queued utterances using
one model instance. Final extraction and persistence follow capture.

Each session has an `OUT/mic_<uuid>/` directory. Its `session.json` records the
device, source clock, stop reason, counters and paths. `capture.f32le` contains
16 kHz mono float32 samples and `provisional-transcript.jsonl` contains the
individual Hojo replies. Use the finalized observations in `OUT/run.json`
for revision history. If a queue fills, capture stops and preserves the spool.
Check `known_lost_native_frames` and `device_lost_frames_unknown` in the
metrics, including for interrupted sessions. A raw spool can be converted
locally with `ffmpeg -f f32le -ar 16000 -ac 1 -i capture.f32le recovered.wav`.

The tested microphone transport used paced digital samples with real local
models. Physical microphone access and live latency were left untested at
the user's request. [Fixture evidence](reports/offline/fixture-mic.json)

### Exit codes

| code | meaning |
|---|---|
| 0 | run finished `ok` (a transcription-only run is `ok` with extraction `not_run`) |
| 10 / 11 / 12 | run saved with status `no_speech` / `needs_review` / `error` |
| 1 | unexpected error |
| 2 | usage or configuration error (`doctor` too, when the settings file is invalid) |
| 3 | missing dependency, asset or service (`doctor` not ready, `models prepare` incomplete) |
| 4 | audio input or device error, including a missing, unreadable or empty input file |
| 5 | database or export error (no such run, locked database, conflicting result) |
| 130 | interrupted |

Runs with status 10–12 are still persisted and exported. The result is always
printed on stdout (`run_id`, `status`, stage statuses); diagnostics go to stderr.

## What a run writes

`--out DIR` receives views of the committed run, all rendered from the manifest
stored in SQLite (so `export` reproduces them byte for byte):

| file | content |
|---|---|
| `run.json` | the full validated `RunManifest`: raw ASR output, normalized text, provenance, metrics, extractor attempts, every observation |
| `observations.csv`, `segments.csv` | UTF-8, `\n` line ends, header row, empty cell = null |
| `observations.txt`, `segments.txt` | fixed-width terminal tables, null shown as `-`, ids shortened (`obs_148d2cff71`), long cells end with `…` |

Saves, exports and recovery share a per-directory `.speechflow.lock` on the local filesystem. A writer waits up to 30 seconds. If a save times out, `speechflow recover --out DIR` can commit the saved recovery copy once the writer finishes. For a consistent read during an active export, read the single `run.json` manifest or use `speechflow export RUN_ID` to render a committed database record.

For the synthetic French recording “Il reste deux virgule cinq kilos de tomates
et douze bouteilles de lait”, the CPU file pipeline produced these rows.

| Product as spoken | Quantity | Unit | Status |
|---|---:|---|---|
| tomates | 2.5 | kg | ok |
| lait | 12 | bouteille | ok |

The original Hojo reply, finalized evidence spans, CSV/table files and direct
SQLite rows are available for this run. [Raw manifest](reports/integration/jacques-350-manifest.json) · [CSV](reports/examples/jacques/observations.csv) · [Terminal table](reports/examples/jacques/observations.txt) · [SQLite comparison](reports/examples/jacques/sqlite-export-check.json)

Check the run status and `errors` even when the observation list is empty. For
example, Hojo heard a sugar-negation fixture as “il ne reste pas deux ou trois”.
Speechflow saved the exact unresolved span for review. The recording's product
could not be recovered. On the continuous 72.46 s synthetic inventory test,
CPU and MPS each recognized four of eight predeclared phrases. One incorrect
product count passed validation on CPU. [Spoken negation and correction](reports/evaluation/semantic-audio-reviewed.json) · [Long-file scores](reports/evaluation/synthetic-real.json)

* Quantities are base-ten decimal strings. An unknown quantity is empty/`-`,
  never `0`; `qualifier` says `exact`, `approximate`, `range` or `unknown`.
* Packaging keeps count, per-package contents and total in separate columns; a
  total exists only when the transcript states it.
* `evidence` is `segment_id:start-end` (Unicode code points, half open) and
  `evidence_quotes` the quoted text. Segment times are source-relative sample
  offsets at 16 kHz plus rounded seconds; they are segment bounds, not word
  alignment.
* Corrections are new observations (`operation=correction`, higher `revision`,
  `correction_of` = target). The superseded value stays. `is_final` is `false`
  only when a revision with status `ok` names the row; a correction still at
  `needs_review` leaves the original `true` and the pair visible via
  `corrected_by` / `corrects`. A correction whose target is missing or ambiguous
  must be `needs_review`.
* Safety: terminal cells escape control characters (a transcript cannot emit
  escape sequences). In CSV, text cells that begin with `= + - @` get a leading
  `'` against spreadsheet formula injection; numeric columns and `run.json` are
  untouched.

Export files are staged in a temporary directory and swapped in one file at a
time, `run.json` last.

## Database

SQLite, WAL mode, `synchronous=FULL`, foreign keys on, parameterized SQL only,
file mode `0600`, schema applied by checksummed migrations in
`src/speechflow/migrations/` (a modified, missing or newer migration is
refused). Tables: `sources`, `runs`, `run_saves`, `segments`, `observations`,
`observation_evidence`, `save_events`; views `current_runs`,
`current_segments`, `current_observations`, `observation_history`. Open it with
any SQLite client; keep it out of Git (`*.sqlite*` is ignored).

* **One transaction per save** (`BEGIN IMMEDIATE`). A failure or killed process
  leaves no partial rows. Concurrent writers serialize; readers see one
  complete snapshot.
* **Immutable history.** Triggers abort `UPDATE`/`DELETE` on every history table;
  the only mutable value is the `runs.current_save_id` pointer.
* **Identity.** `run_id` already encodes source, mode and processing version, and
  `(source, mode, processing_version)` is unique. Segments and observations are
  keyed by stable ids *within a save*; text equality never deduplicates, so two
  genuinely identical utterances at different offsets stay two rows, and
  `transcribe` and `run` of one file are separate runs. A changed
  model/schema/configuration is a new processing version and a new run; earlier
  runs stay untouched.
* **Idempotent retry.** Saving the same result again adds no history rows
  (only a `save_events` line). "Same result" ignores timings, provenance, raw
  model output and extractor replies, so a repeat run with identical transcript
  and observations is a duplicate and a correction is never applied twice.
* **Conflicts and recovery.** A retry may append a generation to a failed run.
  For a `needs_review` run it must preserve the transcript, all prior event IDs
  and every accepted observation. It must also improve the status, add events
  or reduce the error set. Other conflicting changes exit 5 and leave the
  attempted result in `OUT/.recovery/`. SQLite retains every committed generation.
* **Invariants checked before writing**: unique segment/observation ids and
  ordinals, evidence quotes equal to the segment text at their offsets, decimal
  strings, correction targets resolvable or `needs_review`, no correction cycles,
  mode/source kind and `not_run` consistent.
* **Recoverable writes.** Before the transaction a copy of the manifest goes to
  `OUT/.recovery/`; it is deleted after the database commit *and* the exports
  succeed. If anything fails, `speechflow recover --out OUT` commits and exports
  the leftovers (idempotently), and `speechflow export RUN_ID --out DIR` redoes
  just the views.
* **Backup**: `sqlite3 inventory.sqlite3 ".backup backup.sqlite3"` (safe while in use).
  `doctor` reports integrity problems of an existing database.

## Configuration

`examples/settings.toml` lists every key with its default. Paths expand `~`.
Precedence: command-line option > `--config` file (or `$SPEECHFLOW_CONFIG`) >
defaults. Invalid values exit 2 before any audio is touched.

## Tests

Run these commands from the checkout after the frozen install. The default
suite exercises validation, error handling, retries, concurrency, storage and CLI behavior.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/qa
```

For real-model acceptance, prepare the assets and start local Ollama. The short
synthetic inventory recording uses the French Jacques voice supplied by macOS.
The two human-recorded FLEURS examples have separate provenance and references.

```sh
mkdir -p "$HOME/.cache/speechflow"
say -v Jacques -o "$HOME/.cache/speechflow/hojo-jacques.wav" --file-format=WAVE \
  --data-format=LEI16@16000 'Il reste deux virgule cinq kilos de tomates et douze bouteilles de lait.'
.venv/bin/python evaluation/fleurs-fr/prepare.py
.venv/bin/speechflow --offline models prepare
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q tests/real_models
.venv/bin/speechflow --offline evaluate evaluation/fleurs-fr/manifest.jsonl --out outputs/fleurs
.venv/bin/speechflow --offline evaluate evaluation/fixtures/reference-fr.jsonl --out outputs/reference
.venv/bin/python tools/qa/make_long_french.py outputs/qa-fixtures/long-jacques.wav
.venv/bin/speechflow --offline evaluate outputs/qa-fixtures/long-jacques.manifest.jsonl --out outputs/synthetic-real
.venv/bin/python tools/qa/make_semantic_french.py outputs/qa-fixtures/semantic-fr
.venv/bin/speechflow --offline evaluate outputs/qa-fixtures/semantic-fr/manifest.jsonl --out outputs/semantic-audio
.venv/bin/python tools/offline/verify.py fixture-mic --audio "$HOME/.cache/speechflow/hojo-jacques.wav" --seconds 8 --out outputs/fixture-control
```

Run heavyweight commands one at a time. Set `SPEECHFLOW_CONFIG` to a TOML
configuration to choose the ASR backend across CLI commands and tests.
The real file suite creates a dedicated loopback Ollama server and blocks
external connections from both the application and server during offline verification.
The `fixture-mic` command feeds digital samples into the callback transport and
uses the real models. The default real-model suite excludes physical capture.
The separately selected physical microphone test opens the device and needs a speaker.
Set `SPEECHFLOW_QA_FILE_EVIDENCE` to the passing file run's `verification.json`
and select it independently with `-m microphone`.
