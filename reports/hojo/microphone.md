# Segmented microphone evidence

The installed `speechflow mic` path captures native-rate input, averages its channels, and resamples to 16 kHz with a bounded polyphase converter and an end-of-stream flush. It keeps a durable session ID and a digest of the recovered PCM. The PortAudio callback uses a bounded queue with one emergency block, and capture stops with an incomplete result when that queue saturates. The run retains its recoverable PCM and accounts for loss where a count is available. The separate provisional transcript records utterance ASR while capture continues. One lazy Hojo instance serves utterances queued from per-stream Silero VAD. A 350 ms silence threshold is shared with file VAD. Long utterances get capped chunks with overlap, and the worker flushes any pending speech at EOF or interrupt.

## Deterministic checks

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest \
  -p no:cacheprovider -q tools/hojo/test_microphone.py tests/qa/test_audio_contract.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/hojo/fixture_input.py \
  "$HOME/.cache/speechflow/hojo-jacques.wav"
```

The first command passed 26 checks, including injected device frames, stereo and resampling clocks, capped chunks, EOF, interruption, device errors, and device and ASR queue saturation. The fixture transport self-test used a guard against native PortAudio import and library loading. It called the substitute twice with 1,440-frame blocks and verified elapsed pacing and the capture clock. [Pytest output](microphone-injected-pytest.log), [fixture self-test](fixture-input-self-test.json).

## Offline fixture run

The coordinator's offline harness ran the installed CLI with `mic --seconds 8.0 --device SPEECHFLOW_QA_FILE_INPUT_ONLY --json --offline`. An import guard and the acceptance-only module furnished the same Jacques synthetic WAV through paced 48 kHz mono callbacks with one second of leading digital silence and at least two seconds of trailing silence. The confined run used real Silero, pinned Hojo on CPU, and the pinned Qwen extractor. The CLI and both processing stages reported `ok`, with exit code 0. The stored result contains 2.5 kg of tomatoes and 12 bottles of milk. Physical capture remains blocked at the user's request. The run used file injection and did not open a native device or play audio. [Offline controls and invocation](../offline/fixture-mic.json), [stored manifest](../offline/fixture-mic-manifest.json), [CLI output](../offline/fixture-mic-cli.json), [verification receipt](../offline/fixture-mic-receipt.json).

All 384,000 native frames arrived and were processed. The durable spool contains 128,000 float32 PCM samples in 512,000 bytes. Reported loss was zero, peak capture and ASR queue depths were both one, and the single utterance had 0.395 s endpoint delay, about 21 µs ASR queue delay, and 14.087 s in the actual ASR call. The model load took 9.421 s and pinned asset verification took 5.033 s.

`metrics.audio.last_speech_end_monotonic_s` is the estimated host monotonic time of the final detected Silero core end. It is computed from the first block's ADC timestamp mapped to the host clock, plus the core end sample offset divided by 16,000. For this fixture the origin was synthetic. Sample 67,584 was 4.224 s into the capture. The parent measured 38.076 s from that estimate to validated output and 38.084 s to durable output. The field is null when Silero detects no speech.
