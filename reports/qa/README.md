# Independent QA on the target Mac

| Gate | Status | Evidence |
| --- | --- | --- |
| Independent QA | PASS | [Command transcript](commands.log) |
| Regression fixes | PASS | [Initial extraction failures](initial-semantics.log), [initial storage failures](initial-storage.log), [integrated Python tests](integrated-python-tests.log) |
| Real codec decoding and Silero | BLOCKED | `imageio-ffmpeg` and the pinned Silero stack were missing from the installed environment. [Gated test output](real-audio-pytest.log) |
| Real Hojo and extractor | BLOCKED | The 11,956,909,856-byte Hojo weight and pinned `qwen3:8b` digest were unavailable. Heavyweight tests require the coordinator's model slot. [Gated test output](real-heavy-pytest.log) |
| French WER | BLOCKED | Two original French FLEURS validation recordings, 20.46 s and 61 reference words. The corpus has one unspecified condition and no clean/noisy labels for these clips. Zero clips ran inference. [Raw CLI report](fleurs.cli.json) |
| Exact-transcript extraction | BLOCKED | 14 cases and 15 gold observations. All 14 cases encountered the extractor provisioning gate. [Raw CLI report](reference.cli.json) |
| ASR-backed inventory extraction | BLOCKED | One 72.46 s synthetic Jacques recording and eight gold observations. Both model gates applied. The generated file is ineligible for WER. [Raw CLI report](synthetic.cli.json) |
| Network-isolated file inference | BLOCKED | Executable test requires both pins and verifies loopback and external network controls on both processes. [Gated test output](real-heavy-pytest.log) |
| Microphone, queue saturation and offline microphone inference | BLOCKED | No microphone capture before the real file gate. Queue configuration boundaries ran in the independent suite. [Gated test output](microphone-gate.log) |

Scripted chat tests use the production extraction and validators. Hojo response
and frame-planning tests use test doubles. Evaluator tests use fabricated inference
to verify WER normalization, denominators, and handling of absent model outputs.
Storage tests use real SQLite, filesystem exports and the installed CLI with
independently constructed French observations. These results establish
deterministic behavior and do not establish model accuracy or throughput.

The real-model suite avoids skips on selected cases. The software package, model
slot or model asset preconditions fail with a blocker. WAV, FLAC, MP3 and M4A
checks specifically exercise stereo right-channel retention, resampling, corrupt
input and the spool budget. The real Silero tests include a partial VAD frame at
active EOF. The long Hojo test inspects actual submitted WAV frames, verifies one
Hojo instance and one raw record per chunk, and checks timed content before and
after forced boundaries, after 30 seconds and at the tail. The long extraction
test requires exactly one observation per independently timed inventory phrase.

```sh
.venv/bin/python -m pytest -q tests/qa
.venv/bin/python -m pytest -q tests tools/extraction/tests
.venv/bin/python -m pytest -q tests/real_models -k 'silero or codec'
# Once the coordinator grants the model slot and both model pins are provisioned
SPEECHFLOW_QA_MODEL_SLOT=1 .venv/bin/python -m pytest -q tests/real_models
# Separate capture after successful real offline file inference
SPEECHFLOW_QA_MODEL_SLOT=1 SPEECHFLOW_QA_FILE_EVIDENCE=path/to/verification.json \
  .venv/bin/python -m pytest -q tests/real_models -m microphone
```
