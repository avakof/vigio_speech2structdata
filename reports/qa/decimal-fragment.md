# QA for an unresolved split decimal

The regression replays the original Qwen reply from
[the archived short-pause run](../integration/jacques-short-pause-before.json)
through production extraction using a scripted backend. The test retains the
original segment IDs, sample offsets, raw and normalized words, and raw Qwen
reply. Every returned evidence span must quote the original segment exactly.

| Input and reply | Before the fix | After the fix |
| --- | --- | --- |
| Archived `il me reste douze virgules` followed by `cinq kilos de tomates et douze bouteilles de lait` | Accepted 5 kg tomatoes and 12 bottles of milk | Tomato candidate is `needs_review` with `quantity_follows_unresolved_decimal_fragment`. Only the independent 12 bottles of milk are accepted. Neither 2.5 nor 12.5 is emitted. |
| Valid `deux virgule` followed by `cinq kilos de tomates` with two exact evidence spans | Accepted 2.5 kg tomatoes and 12 bottles of milk | Same accepted values and two tomato evidence spans |
| Complete `Douze bouteilles d’eau.` followed by the tomato and milk counts | All three independent counts accepted | Same accepted counts |

```sh
.venv/bin/python -m pytest -q -s --tb=short tests/qa/test_semantics.py -k decimal_fragment
.venv/bin/python -m pytest -q tests/qa
```

- `test_decimal_fragment_archived_qwen_cannot_accept_five_kilos_of_tomatoes`
- `test_decimal_fragment_singular_with_two_evidence_spans_preserves_decimal_and_milk`
- `test_decimal_fragment_complete_previous_utterance_leaves_the_next_count_independent`

The three targeted cases produced **1 failure and 2 passes** against the
hash-matched baseline, then **3 passes** against the production fix. The final
independent QA suite produced **67 passes**. All runs used either the archived
reply or scripted controls. [Baseline](decimal-fragment-before.log),
[focused rerun](decimal-fragment-after.log), [independent QA](decimal-fragment-qa.log).
