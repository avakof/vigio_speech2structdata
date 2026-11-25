# Full real-model QA on CPU

```sh
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q -s tests/real_models \
  --basetemp outputs/qa-full-real-001
```

The [full command log](real-full-suite.log) preserves pytest output, package
versions, artifact pins, and production source hashes at the beginning and end.
A temporary pytest plugin recorded the return values of the actual long-file
calls. Both test assertions and production inference ran unchanged.

Pytest exited **1** after **478.94 s** with **9 passed, 3 failed and 1
deselected**. All seven codec and Silero checks, the installed CLI real-Qwen
reference test, and the independent Seatbelt file test passed. The two
long-content tests and the replay assertion failed. The microphone test was
deselected. [Machine-readable summary](real-full-summary.json)

The reference run covered all 16 gold observations in 15 exact-transcript
cases with zero false positives and false negatives in the predeclared fields.
The Seatbelt file run finished in 29.53 s of pipeline time with the expected
tomatoes and milk counts. The app and dedicated Ollama server each confirmed
external IPv4 and IPv6 denial with loopback available.
[Independent Seatbelt receipt](real-full-offline-file.json)

The 210 recorded runtime snapshots saw at most one CLI Hojo child and zero
samples with both a CLI Hojo child and a loaded model in the default Ollama
instance. The default instance was empty at launch and confirmed empty twice
after pytest exited. The heavy slot was returned at suite completion.
[Runtime samples](real-full-process-samples.jsonl)

## Short Jacques fixture and replay

The installed CLI accepted both 3.571375-second invocations with one 57,142-sample
chunk and the expected 2.5 kg of tomatoes and 12 bottles of milk. The two CLI
processes took 34.39 s and 32.95 s. Both observed the pinned Ollama model during
the call and returned Ollama to idle. The first invocation passed the independent
committed-row and export checks. [First manifest](real-full-jacques-first.json)
and [second manifest](real-full-jacques-replay.json).

The replay assertion failed because the second invocation inserted a new
processing version. The only changed provenance field was the implementation
SHA-256. The full segment records and validated observation fields excluding
run-derived observation IDs matched. The final stable-source replay remains
pending a separate acceptance run. [Provenance and process receipts](real-full-replay-drift.json)

## Long synthetic clip

The 72.46-second clip yielded three keyed Hojo segments with incoming sample
counts of 397,312, 400,000, and 374,848. The cores meet at source samples
394,112 and 787,712. The last core reaches the end of the source at
sample boundary 1,159,360. CPU Hojo used an FP32 encoder and BF16 decoder, with 7.53 s
for model load and 239.25 s in ASR. The ASR real-time factor was 3.30 by dividing
ASR seconds by the full clip duration. Hojo's reported process peak was
13,757,087,744 bytes. Qwen recorded 22.43 s across two generation calls.

| Timed phrase | Start in source | Observed result |
| --- | --- | --- |
| huit kilos de tomates | 0.00 s | Exact anchor present |
| quatre kilos de carottes | 10.04 s | Exact anchor present |
| six kilos de courgettes | 20.33 s | Exact anchor present |
| neuf kilos de pommes | 30.40 s | Missing from the middle chunk |
| cinq kilos de poires | 40.36 s | The middle chunk repeated « cinq kilos de poissons » |
| trois kilos de kiwis | 50.43 s | Missing. The same chunk emitted « quatre kilos de citron ». |
| sept kilos de citrons | 60.59 s | The final chunk emitted singular « sept kilos de citron » |
| deux kilos de gingembre | 70.73 s | Exact tail anchor present |

Both long-content assertions failed. The ASR matched four of eight timed
phrases under the test's exact-anchor rule. Extraction returned `needs_review`
with six accepted observations, four matching gold and two extra accepted
counts. A seventh candidate was a 7 kg correction with no product and an
ambiguous target. The long fixture is synthetic.
[Raw keyed Hojo replies and normalized segments](real-full-long-hojo.json),
[raw Qwen replies and validated candidates](real-full-long-extraction.json),
and [anchor and inventory comparison](real-full-long-analysis.json).

The middle call's 25-second float32 WAV was reconstructed from the decoded
PCM for the same fixture and byte-checked against its PCM slice. It contains
both missing source anchors with room on either side. The evidence includes
its original Hojo key, half-open sample ranges, hashes and unchanged response.
The isolated input is retained under the suite output directory.
[Single-call reproducer](real-full-long-reproducer.json)
