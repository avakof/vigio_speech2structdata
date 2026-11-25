# Synthetic French audio for negation and correction

The [frozen gold declarations](../../evaluation/fixtures/semantic-audio-fr.jsonl)
were set before speech synthesis or inference. The [generated manifest](../../outputs/qa-fixtures/semantic-fr/manifest.jsonl)
records the original WAV hashes, source, sample counts, duration and expected
observations. Both WAVs and the generated manifest are Git-ignored.

| Case | Spoken French | Gold | Samples at 16 kHz |
| --- | --- | --- | ---: |
| Negation | Il ne reste pas de sucre. | One sugar negation with null quantity and unit | 26,062 |
| Correction | Il reste trois kilos de farine, pardon, deux kilos de farine. | 3 kg of flour followed by a same-product 2 kg correction linked to gold index 0 | 71,387 |

Both clips were generated to file by macOS `say` using Jacques (`fr_FR`) at
160 words/min on macOS 27.2.1 (26B131). They contain nonempty 16 kHz mono
PCM-16. These are local QA artifacts. No redistribution license is asserted for
the macOS system voice output. The evaluator excludes these synthetic cases
from WER.

```sh
.venv/bin/python tools/qa/make_semantic_french.py outputs/qa-fixtures/semantic-fr
```

The model-free preflight parsed both manifests with the evaluator, independently
checked WAV headers, payload length and SHA-256, and confirmed one gold negation,
one gold correction, two gold quantities and two gold units. A second render
on the same OS produced byte-identical WAVs and manifest. [Preflight output](semantic-audio-fr-preflight.log)

Once the inference slot is assigned, this command measures extraction on the
exact reference and on actual ASR output for the two cases.

```sh
.venv/bin/speechflow evaluate outputs/qa-fixtures/semantic-fr/manifest.jsonl \
  --offline --out outputs/qa-semantic-fr-evaluation
```
