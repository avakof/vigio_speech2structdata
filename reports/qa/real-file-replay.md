# Installed CLI file replay acceptance

The test uses the pinned synthetic Jacques recording
`~/.cache/speechflow/hojo-jacques.wav`. It is 57,142 samples of mono PCM at
16 kHz with SHA-256
`d92c0759b7d4d3e03ae22eb3c47aa97ef7122789eac187c7cf42e8e6035d1330`.
The spoken fixture is “Il reste deux virgule cinq kilos de tomates et douze
bouteilles de lait.” The gold observations are 2.5 kg of tomatoes and 12 bottles
of milk with no per-bottle quantity supplied.

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q -s \
  tests/real_models/test_file_replay.py::test_real_file_installed_cli_replay \
  --basetemp outputs/qa-real-file-replay
```

Run this command with the exclusive slot after the real file gate. Use a fresh
`--basetemp` when retaining results from a previous run. The two `speechflow run`
processes use `model_settings.backend` with the same audio bytes and database
and distinct output directories. The test records each subprocess command, PID,
wall time, exit code, stdout, stderr, and the Ollama model snapshots for the two
runs under the pytest artifact directory. It requires genuine Hojo chunk keys
and sample bounds, pinned Qwen generation metadata, and the pinned Ollama runner
observed during each CLI invocation. `--offline` is the application setting used
for this test.

Independent read-only SQLite queries verify that matching run IDs, processing
versions, chunk IDs, bounds, and evidence spans remain at generation 1. There
must be exactly one physical set of two observations and two save events,
`inserted` followed by `duplicate`. The newly attempted manifest has a distinct
receipt digest. The first run, replay, and fresh database exports must produce
byte-identical JSON, observation and segment CSVs, and observation and segment
tables. The test checks the committed JSON bytes and compares the CSV and table
content with direct SQL rows.

| Check | Status | Evidence |
| --- | --- | --- |
| Separately selectable node and pinned fixture preflight | PASS | [Collection and fixture log](real-file-replay-preflight.log) |
| SQL and export assertion smoke on seeded data | PASS | [Model-free log](real-file-replay-model-free.log) |
| Selecting the heavy test without a slot | BLOCKED as expected, exit 1 before test execution | [Slot gate log](real-file-replay-gate.log) |
| Real file replay and export acceptance | NOT RUN | Awaiting the exclusive slot and the first complete real file run |
