# Hojo feasibility on Apple silicon

## Pinned inputs and host

- OpenAI Codex `gpt-6.2-sol` at `high`, verified from the active rollout `01a2130a-657d-7421-8af1-24f3ce8db31b`. Orca session `8666ee97-35a9-4b30-b66f-8d4252fca38e` in `/Users/vakof/orca/vigio_speech2structdata`.
- MacBook Pro, Apple M3 Max, native arm64, 48 GiB unified memory, 16 CPU cores, 40 GPU cores, Metal 4, macOS 27.2.1 (26B131), and 748 GiB available at discovery. uv 0.13.9 and native CPython 3.12.16.
- [Hojo-ASR-Multi-V1 at aa1d0dfc369deb3b3235572fa151acd7b498481f](https://huggingface.co/HojoAI/Hojo-ASR-Multi-V1/tree/aa1d0dfc369deb3b3235572fa151acd7b498481f), Apache-2.0. The model card lists French and describes a Qwen decoder attached to a speech encoder and adapter. The merged weights are 11,956,909,856 bytes, SHA-256 `c046a5a26c142f5aded7331d36892293a025f579e342a27861e80509afaca88d`.
- [PyPI hojo-asr 0.1.3](https://pypi.org/project/hojo-asr/0.1.3/), Apache-2.0 with WeNet/ESPnet notices. Wheel SHA-256 matches PyPI, `8c3e39b55b77ff60bb6232ede09375a4e799233452ee0cfd7a77558fd6e0780a`. The distribution metadata says 0.1.3, while the wheel module reports `__version__ = "0.1.2"`. The adapter and benchmark use distribution metadata. Target stack is torch 2.5.1, torchaudio 2.5.1, Transformers 4.57.3, and the project's frozen transitive dependencies.
- The smoke input is macOS `say`, Jacques (`fr_FR`), speaking “Il reste deux virgule cinq kilos de tomates et douze bouteilles de lait.” The generated WAV has 57,142 samples, 3.571375 seconds, 16 kHz mono PCM signed 16-bit, with SHA-256 `d92c0759b7d4d3e03ae22eb3c47aa97ef7122789eac187c7cf42e8e6035d1330`. It is synthetic functional evidence. French recognition accuracy is **NOT MEASURED**.

## Wheel inspection and implementation

`HOJO_ASR.load_model(folder_path, device="cuda:0", *, cache_dir=None, revision=None, token=None, local_files_only=False)` and `run_infer(audio_input, *, seed=777, batch_size=1, cuda_enabled=True)` are the pinned wheel interfaces. In this version the `cuda_enabled` flag makes the data path use the explicit model device when set, so MPS uses `True`. CPU uses `False`. The encoder constructor forces eager attention even though the caller initially requests Flash Attention 2.

The wheel wraps every non-CPU model in CUDA autocast. The adapter subclasses the wheel and disables that context for the CPU and MPS paths. The merged header declares float32 encoder, bottleneck, and speech normalization, and bfloat16 decoder. The adapter explicitly casts the audio embedding to the decoder embedding dtype at the handoff. It checks that the loader returns the intended subclass with both method overrides bound. The benchmark records device and dtype by submodule, plus actual use of both overrides. The loader initializes decoder parameters with Accelerate's empty-weight context, creates the encoder and adapter, memory-maps CPU tensors from safetensors, assigns through a strict state dict load, and then moves the model. The wheel installs its own process-wide fast-init patches when imported. The application adds no global patches. The MPS transfer may transiently require both mapped CPU weights and accelerator allocation and the file path runs in one process with one Hojo model.

The bundle declares a 30-second front end (480,000 samples). The wheel passes `chunk_length=40` to the Whisper feature extractor and passes `padding=False` when producing features. Transformers 4.57.3 merges the override before constructing the extractor, recalculates `n_samples`, then uses `n_samples` as the default truncation length. `--frontend-only` probes 25, 30, 31, 40, and 41 seconds with each setting. [Configuration loader](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/feature_extraction_utils.py#L584-L595), [Whisper feature extractor](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/whisper/feature_extraction_whisper.py)

The application caps every submitted chunk at the configured maximum of 25 seconds, including overlap and padding. The file path keeps disjoint core ranges. Neighboring padded inputs can share 400 ms at the default 200 ms one-sided padding. A suffix-prefix match of at least two words on shared input samples flags both segments for review. Raw and normalized texts retain both replies because segment-level timing cannot resolve the ambiguity. Missing or mismatched keys surface as segment errors.

FFmpeg selects the first audio stream, downmixes according to its source layout, and resamples to 16 kHz float32 in one pass. The source digest covers the original file bytes. The PCM spool has a byte limit and is deleted on exit. At most one additional bounded Hojo WAV is open during inference. Silero uses the pinned local JIT model, samples 512-frame blocks with per-source state, and flushes active speech on the original decoded clock. A padded last VAD block cannot extend segment bounds past EOF.

## Reproduction

```sh
uv venv --python 3.12 "$HOME/.cache/speechflow/hojo-env"
uv export --frozen --no-dev --no-emit-project --format requirements-txt \
  --output-file "$HOME/.cache/speechflow/hojo-locked-requirements.txt"
uv pip install --python "$HOME/.cache/speechflow/hojo-env/bin/python" \
  -r "$HOME/.cache/speechflow/hojo-locked-requirements.txt"
say -v Jacques -o "$HOME/.cache/speechflow/hojo-jacques.wav" --file-format=WAVE \
  --data-format=LEI16@16000 'Il reste deux virgule cinq kilos de tomates et douze bouteilles de lait.'
HF_XET_HIGH_PERFORMANCE=1 "$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/prepare.py
"$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/prepare.py --verify-only
"$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/feasibility.py --frontend-only
"$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/file_probe.py "$HOME/.cache/speechflow/hojo-jacques.wav"
for backend in cpu mps; do
  "$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/feasibility.py \
    --backend "$backend" --primitives-only
  /usr/bin/time -l "$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/feasibility.py \
    "$HOME/.cache/speechflow/hojo-jacques.wav" --backend "$backend" \
    --reference 'Il reste deux virgule cinq kilos de tomates et douze bouteilles de lait.' \
    --provenance 'Synthetic macOS say Jacques fr_FR' \
    >"reports/hojo/$backend.jsonl" 2>"reports/hojo/$backend.log"
done
```

Each backend uses a fresh process. The asset digest check is timed separately and may prime the filesystem cache. Cold load means the first model construction in that process. The second inference with that model is the warm measurement. Real-time factor is the synchronized inference wall time divided by the WAV duration. Observed RSS, peak RSS, system available memory, and MPS allocation are separate views. The MPS counters must never be added to RSS because they refer to unified memory.

## Provisioning evidence

The first direct HF CDN attempt transferred approximately 1.3 MB before an HTTP/2 cancellation. Two bounded range probes returned partial payloads before timeout. The initial uv install timed out on shared cache locking for the `langid` source distribution. A private-cache retry hit PyPI transport timeouts. The shared lock selects SciPy 1.17.1 with NumPy 1.26.4. The offline lock-free resolution could not select that uncached wheel. The official HF Xet snapshot later fetched 17 files, leaving the merged weights in progress. The local verifier confirmed that only the merged weight was missing among the 11 runtime assets as of its recorded run.

| Check | Status | Evidence |
| --- | --- | --- |
| Input format and SHA-256 | PASS | Jacques and two local FLEURS files matched their expected sample metadata and hashes. |
| Forced split and overlap review | PASS | Arithmetic and review flags tested with fabricated segment and transcript data. |
| Full snapshot verification | BLOCKED | Xet is still transferring the merged weight. The sparse staging file's logical size does not measure successfully reconstructed bytes. |
| Frontend execution | BLOCKED | The isolated locked uv environment is still provisioning. The frontend probe stopped at the NumPy import. |
| File decoding and real Silero | BLOCKED | The file probe stopped before decoding on missing `silero-vad` package metadata. |
| CPU and MPS inference | NOT RUN | Awaiting verified weights, dependencies, and release of the exclusive inference slot. |

- [Direct download log](download-weight.log)
- [Xet snapshot log](download-snapshot.log)
- [Locked dependency provisioning log](provision-locked.log)
- [Missing asset verification](verify-before-complete.log)
- [Jacques and two FLEURS validation WAV preflight checks](input-preflight.log)
- [Forced-boundary arithmetic with a mocked continuous VAD range](boundary-arithmetic.jsonl)
- [Frontend dependency preflight](frontend-preflight.log)
- [File/VAD dependency preflight](file-preflight.log)
