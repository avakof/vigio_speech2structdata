# Final real-model QA readiness

Model-free collection selected 12 tests in the CPU suite and deselected the
physical microphone test. The installed-file replay test also collects on
its own. Both output paths are fresh. [Collection evidence](final-real-readiness.log)

The replay checker reads the requested backend from `model_settings` and
compares it to the actual manifest backend. It checks keyed Hojo replies,
evidence spans, and exactly 2.5 kg of tomatoes and 12 bottles of milk. It does
not require a CPU-specific transcript string or compare MPS transcript bytes
against CPU bytes. It requires stable processing and content across the two
runs made with the same backend. The CPU and MPS runs retain their own raw text
for comparison. The existing CPU criteria remain in the test.

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q -s tests/real_models \
  --basetemp outputs/qa-final-real-002

qa_mps_config=$(mktemp -t speechflow-qa-final-mps)
printf 'backend = "mps"\noffline = true\n' > "$qa_mps_config"
SPEECHFLOW_CONFIG="$qa_mps_config" SPEECHFLOW_QA_MODEL_SLOT=1 \
  .venv/bin/python -m pytest -q -s \
  tests/real_models/test_file_replay.py::test_real_file_installed_cli_replay \
  --basetemp outputs/qa-final-mps-replay-002
```

The two commands are prepared for serial execution after the coordinator
grants the exclusive model slot. The evidence will retain source hashes,
installed artifact versions, process samples, SQLite receipts and the raw
long-audio results.
