# Hojo generation-loop review

File and microphone inference share a diagnostic for a narrow output pattern. The adapter retokenizes the unchanged returned text with the loaded Hojo instance's tokenizer and no added special tokens. It checks for a configured cap of 200 and a measured length from 196 through 204 tokens inclusive. Within that band, it looks for three consecutive exact copies of a block whose smallest period is at least 16 tokens. The input source, time ranges, inventory words, and backend do not affect the rule.

The warning `hojo_possible_generation_loop` sets the transcript segment to `needs_review`. Existing extraction validation preserves observations that cite the segment and propagates the review status to their evidence. Raw text, normalized text, reply keys, sample ranges, and repeated occurrences stay intact.

Retokenized text length is a proxy. The wheel strips selected special markers and whitespace before returning text, and it can use a lower generation limit for a shorter speech embedding. It returns no generation stop reason. Genuine repeated speech can match the rule, and short, approximate, or lower-limit loops can escape it. The warning gives no confidence estimate and says nothing about the correctness of specific quantities or product words.

| Input | Segment | Extracted evidence |
| --- | --- | --- |
| Archived middle reply | Review, 200 tokens, 48-token period | Four independent 5 kg fish observations retained with review status |
| Two genuine-style repeated inventory utterances | Clear | Both occurrences retained at distinct text spans |
| Unicode with a decomposed accent | Clear | Raw spelling preserved and the normalized accented evidence span resolved |
| 200 tokens dominated by a short loop | Clear | Inventory observation retained as clear |
| 200 tokens with no qualifying long repeat | Clear | Inventory observation retained as clear |
| Four archived shorter bisections | Clear under this rule | Each reply measured 53–63 tokens. The expected words remain absent. |

The replay used the pinned local tokenizer and archived output. The PCM input and model call were scripted. Extraction chat replies were scripted and validated through production extraction. A separate test sent the archived output through the actual microphone ASR queue and verified the raw and review status in its provisional record. No model weights were loaded for these tests. [Archived reply](../qa/real-full-long-reproducer.json), [shorter replies](long-cpu-bisection.jsonl), [test log](generation-loop-pytest.log).

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -p no:cacheprovider -q tools/hojo/test_generation_loop.py \
  tools/hojo/test_microphone.py tests/qa/test_audio_contract.py tests/qa/test_semantics.py
```
