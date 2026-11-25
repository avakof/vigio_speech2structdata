# Independent real-Qwen exact-transcript test

The installed CLI ran all 15 frozen reference cases through local Ollama with
`qwen3:8b` at digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
All 17 generation attempts identify the pinned artifact and contain nonzero
evaluation counts and durations. The report records prompt 4 and generation
schema 4.

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q \
  tests/real_models/test_runtime.py::test_real_extractor_installed_cli_exact_transcript \
  --basetemp outputs/qa-independent-qwen-5252b986aa1e
```

Pytest exited 0 with **1 passed in 93.98 s**. Wall time was 94.10 s and
the evaluator elapsed time was 93.77 s. The comparison baseline took 70.59 s.
Ollama reported 0.53 s for the first load from an empty runner state and
confirmed unload for every case. An external request to `/api/ps` returned
`{"models":[]}` before and after the test. [Command and exit log](real-qwen-independent.log)

The independent run covered **16/16 gold observations**, correctly classified
all 15 review decisions and had zero false positives and false negatives for
every predeclared field. Three candidates remained reviewable. Every full
observation matched `outputs/reference-real-003.json` across all 15 cases,
including evidence spans, revision links and review status.

| Case | Independently verified fields |
| --- | --- |
| Decimal and bottles | 2.5 kg of tomatoes and 12 bottles of milk. Bottle capacity stayed unspecified. |
| Explicit carton contents | 2 cartons, 12 bottles per carton, 24 bottles total. |
| Unknown carton contents | 2 cartons of milk with no per-carton size or total. |
| Same-segment correction | Original 3 kg of flour at revision 1 and linked 2 kg at revision 2 within one segment. |
| Cross-segment correction | Original 3 kg of flour at revision 1 and linked 2 kg at revision 2 across separate evidence segments. |
| Ambiguous correction | Reviewable with no revision target assigned. |

The [raw evaluation](real-qwen-independent-evaluation.json) is a byte-identical
copy of the installed CLI output retained under the dedicated test directory.
The [comparison record](real-qwen-independent-comparison.json) contains the
fixture and output hashes, per-case equality results, and exact correction
identifiers. The serial heavy-model slot was returned after the idle check.
