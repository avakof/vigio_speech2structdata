# Independent QA for packaging, corrections and repair

The initial run reproduced five false acceptances using scripted replies through
production extraction. Both narrowed packaging quotes produced an accepted
count of 12 bottles, and the narrowed `pardon` quote produced an accepted 2 kg
count at revision 1. Both incomplete repairs returned `ok` after dropping the
verified 3-bottle water observation. The four initial positive controls passed.
The [baseline log](inventory-context-before.log) includes the exact transcripts,
reply JSON, validation results and evidence offsets.

| Case | Final result |
| --- | --- |
| `Deux cartons de douze bouteilles` with only the inner quantity quoted | `needs_review` for unnamed contents and named water |
| `Trois kilos de farine, pardon, deux kilos de farine` with only the new amount proposed as a count | `needs_review`. The repair requests both the original and its linked revision. |
| Fully quoted cartons with named water contents | `ok`, two cartons, twelve bottles per carton, total 24 |
| Cartons of milk followed by a separate water count | `ok`, carton contents remain unspecified and the 12-bottle water count remains separate |
| Explicit correction targeting one earlier measure | `ok`, 2 kg at revision 2 linked to the original 3 kg |
| Explicit stock negation after a separate count | `ok`, negation without a quantity or correction link |
| Empty repair or repair containing only corrected flour | `needs_review`, original water observation retained and omission reported |
| Complete repair | `ok`, both water and flour retained |

| Command | Final result | Output |
| --- | --- | --- |
| `.venv/bin/python -m pytest -q -s --tb=short tests/qa/test_semantics.py -k inventory_context` | Exit 0. 10 passed, 32 deselected. | [Focused run](inventory-context-focused-final.log) |
| `.venv/bin/python -m pytest -q tests/qa` | Exit 0. 64 passed. | [Independent QA](inventory-context-qa-final.log) |
| `.venv/bin/python -m pytest -q` | Exit 0. 271 passed, 12 deselected. | [Default suite](inventory-context-default-final.log) |

The [first integrated default run](inventory-context-default.log) preserved
four failures before the follow-up fixes to carry budgeting, correction-chain
matching, and the explicit-package test expectation. The final runs recorded
matching hashes for the extraction sources. The targeted tests ran production
validation with scripted generation, and the default suite deselected the
real-model tests.
