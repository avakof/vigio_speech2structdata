# French inventory evaluation

```sh
.venv/bin/speechflow evaluate evaluation/fixtures/reference-fr.jsonl --offline --out outputs/reference-fr
.venv/bin/speechflow evaluate evaluation/fleurs-fr/manifest.jsonl --offline --out outputs/fleurs-fr
.venv/bin/python tools/qa/make_long_french.py outputs/qa-fixtures/long-jacques.wav
.venv/bin/speechflow evaluate outputs/qa-fixtures/long-jacques.manifest.jsonl --offline --out outputs/long-fr
```

Each JSONL row describes one case. Give it a unique `id` and a `reference` or
ordered `reference_segments`. Set `expected_observations` to enable extraction
scoring. An empty array describes a case where every accepted inventory candidate
is a false positive. Omit the field for WER-only audio. A correction carries
`operation: "correction"` and a zero-based `correction_of` pointing to an earlier
gold observation. Preserve original and corrected values as separate observations.
Set `reference_review` for cases that should have reviewable uncertainty.

For audio cases, set `audio` to a path relative to the manifest, `sha256` to
the digest of those original bytes and `duration_s` to the source duration.
Set `human_verified` only after verifying the reference against a real French
recording. Include `source`, `revision`, and the condition if known. The two
pinned FLEURS samples inherit the corpus verification recorded in their provenance.
Set `synthetic: true` for generated audio. Synthetic examples can test ASR-backed
inventory extraction and are ineligible for French WER.

`evaluation.json` includes model outcomes, available raw replies, gold and
predicted denominators, and measurements per input type. Alignment uses normalized
product names and chronological occurrence within each case. A misrecognized
product produces a missing gold mention as well as an extra prediction. Manual
review candidates count toward the abstention rate and cannot improve
precision. Missing candidates lower coverage and recall. Scores include historical
correction links, quantities and units, negation, and explicit packaging totals.
Undefined rates are null. A measurement tagged `NOT MEASURED` may still display
gold counts and zero coverage when provisioning blocked every case.
