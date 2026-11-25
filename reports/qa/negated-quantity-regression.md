# Incomplete negated quantity regression

The production extractor now returns `needs_review` for the archived
`il ne reste pas deux ou trois` segment. It retains the raw Hojo segment and
the archived empty Qwen reply, adds one exact diagnostic, and creates zero
observations. The diagnostic records segment
`segment_6fd3158fe4e6ae3a2be823ce33f278a49d8040ea01c00e475a1cc8b03951b89e`,
Unicode span `[0,29)` and the exact source quote. The ASR-derived sugar
negation remains a gold miss. [Historical archive replay](negation-empty-replay.md)

The alternative fragments use `quatre` and `six ou huit`. The latter yields
`[25,52)` after a separate preceding statement. Both use one scripted
response. An independent 12-bottle milk observation survives, with its
original evidence in the second segment at `[9,33)`. An unrelated repair
makes two calls in the same window and leaves the incomplete quantity
diagnostic present exactly once. Standalone review requires no repair.

The normal factual negation, `pas de temps`, same-segment and split
`minutes`, and same-segment and split preceding questions with a question
mark all return `ok`, no observations and one call. Named sugar absence
still requests a repair. Repeated empty replies leave the omission under
review. A grounded repaired reply yields one sugar negation with null quantity
and unit and exact `[0,24)` evidence.

The focused cases use the installed production extractor and only replace
the chat backend. The exact archive segment and service reply are frozen
under `tests/qa/fixtures/` with the source report digest. No model weights
were loaded.

| Command | Result | Evidence |
| --- | --- | --- |
| `PYTHONPATH="$qa_baseline" .venv/bin/python -m pytest -q -s tests/qa/test_semantics.py -k negated_quantity_fragment` | Exit 1, 6 failed, 7 passed, 45 deselected | [Before source, output and failures](negated-quantity-before.log) |
| `.venv/bin/python -m pytest -q -s tests/qa/test_semantics.py -k negated_quantity_fragment` | Exit 0, 13 passed, 45 deselected | [After source and full focused output](negated-quantity-after.log) |
| `.venv/bin/python -m pytest -q tests/qa` | Exit 0, 80 passed | [Independent QA](negated-quantity-qa.log) |
| `.venv/bin/python -m pytest -q` | Exit 0, 299 passed, 13 deselected | [Whole default suite](negated-quantity-default.log) |

The baseline used a temporary copy of the local package taken before the
production edits. Its six failures were the five missing fragment
diagnostics and the false review of `pas de temps`. The focused and default
logs retain unchanged before/after hashes for the three updated production
files. The default run deselected the real-model and physical microphone
tests. The frozen long tests and curated gold fixtures were untouched.
