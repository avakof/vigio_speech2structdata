# Independent extraction evidence-scope QA

The baseline run exercised production `extract` and its validator using a
scripted chat backend. No model ran. Each case checks retained reply content and
verifies that every returned evidence span matches the source at its exact
half-open offsets. The [baseline transcript](evidence-scope-before.log)
preserves the full inputs and outputs, production file hashes, and pytest result.

| Transcript | Scripted evidence | Baseline accepted as `ok` |
| --- | --- | --- |
| Il ne reste pas deux kilos de sucre. | `[16,35)` deux kilos de sucre | exact 2 kg sucre, count |
| Il reste environ trois kilos de farine. | `[17,38)` trois kilos de farine | exact 3 kg farine |
| Il reste environ trois kilos de farine. | `[9,38)` environ trois kilos de farine, quantity text `trois` | exact 3 kg farine |
| Il reste entre deux et cinq kilos de farine. | `[23,43)` cinq kilos de farine | exact 5 kg farine |
| Il reste entre deux et cinq kilos de farine. | `[9,43)` entre deux et cinq kilos de farine, quantity text `cinq` | exact 5 kg farine |
| Trois kilos de farine et cinq kilos de sucre | `[0,44)` full quote, product farine and quantity from the sucre clause | exact 5 kg farine |
| Trois kilos de farine et cinq grammes de sucre | `[0,46)` full quote, product farine and quantity and unit from the sucre clause | exact 5 g farine |

All seven unsafe outputs had empty error lists. All seven positive controls
passed. The controls cover a valid narrow quote in a multi-item statement,
`vingt et un`, `entre deux et cinq`, `herbes et épices`, a decimal split across
segments, and positive counts following a negative clause or sentence.

```sh
.venv/bin/python -m pytest -q -s --tb=short tests/qa/test_semantics.py -k evidence_scope
```

## Verification of the first production change

| Command | Result | Output |
| --- | --- | --- |
| `.venv/bin/python -m pytest -q -s --tb=short tests/qa/test_semantics.py -k evidence_scope` | Exit 0. All 14 original cases passed. | [Focused run](evidence-scope-after.log) |
| `.venv/bin/python -m pytest -q tests/qa` | Exit 0. 54 passed. | [Independent QA suite](evidence-scope-qa-after.log) |
| `.venv/bin/python -m pytest -q` | Exit 0. 259 passed, 12 deselected. | [Default suite](evidence-scope-default-after.log) |
| `PYTHONPATH=tests/qa .venv/bin/python reports/qa/evidence-scope-adjacent.py` | Exit 1. Eight passed, four failed. | [Additional cases after the change](evidence-scope-adjacent-after.log) |

The additional cases passed when a preceding segment supplied an omitted
`environ` or lower range endpoint, and when the scripted reply cited both
segments and supplied the complete approximate quantity. `Noix et raisins`
remained one product. A positive inventory fragment passed after a punctuated
negative segment, as did a positive statement with its own explicit predicate.
The new validator correctly flagged a negative predicate split from its count.

| Confirmed concern | Baseline | After change |
| --- | --- | --- |
| `Il ne reste pas de sucre` followed five seconds later by `Deux kilos de farine.` | Accepted 2 kg farine | False review with `count_conflicts_with_source_context` |
| `Il reste trois kilos de farine` followed immediately by `environ.` | Accepted exact 3 kg | Accepted exact 3 kg |
| `Il reste trois kilos de farine environ.` cited in full with quantity text `trois` | Accepted exact 3 kg | Accepted exact 3 kg |
| Inventory segments `Trois kilos de farine` and `Cinq kilos de sucre.` cited together with no punctuation between them | Accepted 5 kg farine | Accepted 5 kg farine |

The baseline additional probe used a Git archive of the production sources
whose validator SHA-256 matches the original 14-case baseline. Both probe
outputs preserve the scripted replies, exact evidence spans and validator
outcomes. [Before](evidence-scope-adjacent-before.log),
[after](evidence-scope-adjacent-after.log), [probe](evidence-scope-adjacent.py).

## Final verification

| Command | Result | Output |
| --- | --- | --- |
| `PYTHONPATH=tests/qa .venv/bin/python reports/qa/evidence-scope-adjacent.py` | Exit 0. 12 passed, 0 failed. | [Adjacent probes](evidence-scope-adjacent-final.log) |
| `.venv/bin/python -m pytest -q -s --tb=short tests/qa/test_semantics.py -k evidence_scope` | Exit 0. 14 passed, 18 deselected. | [Focused suite](evidence-scope-final.log) |
| `.venv/bin/python -m pytest -q` | Exit 0. 259 passed, 12 deselected. | [Default suite](evidence-scope-default-final.log) |

The valid 2 kg flour count after the five-second gap returned `ok`. Both
trailing-`environ` cases returned `needs_review` with
`quantity_or_qualifier_conflicts_with_source_context`. The cross-segment
product and quantity mismatch returned `needs_review` with
`quantity_unit_and_product_not_in_one_source_clause`. The probe used scripted
replies and the real-model tests were deselected from the default run.
