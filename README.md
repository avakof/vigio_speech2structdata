# speechflow

Local French speech → structured inventory data. Audio file or microphone →
normalization → Silero VAD → **HojoAI/Hojo-ASR-Multi-V1** → local extraction
(Ollama `qwen3:8b`) → deterministic validation → JSON, CSV / terminal table and
SQLite. Inference needs no cloud account; cloud coding agents are development
tools only. See `docs/contracts.md` for the frozen interfaces.

Target machine: Apple Silicon Mac (developed for M3 Max / 48 GB), macOS, Python
3.12. The lock file only resolves for `darwin` / `arm64`.

Current verification results are in [the acceptance report](reports/acceptance.json).
File processing, extraction and storage are implemented. The file end-to-end
gate awaits the pinned local weights. Microphone capture is gated on that run.

## Setup — three separate steps

Dependency installation and model preparation are different steps. The first
installs Python packages only; the second is the only step that downloads
weights.

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

`doctor` works before steps 1 and 3: it reports each missing dependency, missing
asset and unreachable service with a fix, and exits `3` while anything is
missing. `speechflow --help` needs only the standard library.

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

Microphone mode is *segmented* transcription: speech is cut at pauses and each
utterance is transcribed as a unit. It is not native streaming.

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

```
#  obs             status  op          product  qty  max  qualifier  unit  packaging                               final  corrects
0  obs_148d2cff71  ok      count       farine   3    -    exact      kg    -                                       false  -
1  obs_cf649f0393  ok      correction  farine   2    -    exact      kg    -                                       true   obs_148d2cff71
2  obs_64be05706a  ok      count       lait     -    -    unknown    -     2 carton x 12 bouteille = 24 bouteille  true   -
```

(Layout illustration rendered from a test fixture; it is not model output.)

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
* **Conflicts.** A different result under an already-stored `ok`, `no_speech` or
  `needs_review` run is refused (exit 5) because one processing version must
  reproduce its result; the attempted result is kept in `OUT/.recovery/`. A run
  whose stored status is `error` is replaced by a new *generation*; the failed
  generation stays in `run_saves`.
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

```sh
pytest tests/persistence_own          # unit and CLI tests, no models, no network
```

These cover replay after restart, distinct identical utterances, two modes of one
source, changed processing versions, corrections, immutability, transaction
rollback, a process killed mid-transaction, failed exports, concurrent writers
and readers, output consistency across JSON/CSV/table/SQL views, hostile
text, exit codes, and `--help` / `doctor` with dependencies absent. The pipeline
and model checks are separate (`-m real_models`, `-m microphone`).
