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

FFmpeg selects the first audio stream, downmixes according to its source layout, and resamples to 16 kHz float32 in one pass. The source digest covers the original file bytes. Decoder provenance records the resolved executable's SHA-256 and the FFmpeg version printed during decoding. The PCM spool has a byte limit and is deleted on exit. At most one additional bounded Hojo WAV is open during inference. Silero uses the pinned local JIT model, samples 512-frame blocks with per-source state, and flushes active speech on the original decoded clock. A padded last VAD block cannot extend segment bounds past EOF. The VAD import restores the caller's Torch thread count after the upstream package changes it, so CPU Hojo retains its configured thread pool.

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
  /usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
    "$HOME/.cache/speechflow/hojo-env/bin/python" tools/hojo/feasibility.py \
    "$HOME/.cache/speechflow/hojo-jacques.wav" --backend "$backend" \
    --reference 'Il reste deux virgule cinq kilos de tomates et douze bouteilles de lait.' \
    --provenance 'Synthetic macOS say Jacques fr_FR' \
    >"reports/hojo/$backend.jsonl" 2>"reports/hojo/$backend.log"
done
```

The exact locked arm64 torch wheel was recovered after the long PyPI stream restarted. With the earlier wheel transfer stopped, `tools/hojo/download_torch_wheel.py --probe` validated HTTP 206 and the exact 256 KiB range, and `tools/hojo/download_torch_wheel.py --workers 24` assembled and verified all 244 resumable byte ranges. Install the resulting `~/.cache/speechflow/hojo-wheels/torch-2.5.1-cp312-none-macosx_11_0_arm64.whl` with `uv pip install --offline --no-deps --python "$HOME/.cache/speechflow/hojo-env/bin/python" WHEEL`, then finish the frozen requirements install. The recovered wheel matched SHA-256 `8c712df61101964eb11910a846514011f0b6f5920c55dbf567bff8a34163d5b1`. The private environment's offline install and `uv pip check` both succeeded.

Each backend uses a fresh process. The asset digest check is timed separately and may prime the filesystem cache. Cold load means the first model construction in that process. The second inference with that model is the warm measurement. Real-time factor is the synchronized inference wall time divided by the WAV duration. Observed RSS, peak RSS, system available memory, and MPS allocation are separate views. The MPS counters must never be added to RSS because they refer to unified memory.

## Genuine Jacques transcription

Both successful runs used batch size 1, 16 kHz mono input, float32 audio encoder and bottleneck, and a bfloat16 decoder on the requested device. Both calls per run entered the device and dtype hooks. CPU used eager encoder attention and SDPA in the decoder. The successful MPS retry used eager attention in both components with PyTorch's MPS fallback disabled.

| Backend | Cold load | First inference | Warm inference | Warm real-time factor | Peak RSS | macOS peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 7.370 s | 6.808 s | 8.751 s | 2.450 | 12.77 GB | 3.60 GB |
| MPS | 16.526 s | 27.339 s | 6.761 s | 1.893 | 3.75 GB | 15.13 GB |

The MPS run reported 11.96 GB of current tensors and 12.79 GB of driver allocation after each inference. The asset checks took 4.987 seconds on CPU and 5.386 seconds on MPS. Whole command times, including startup and verification, were 32.35 seconds and 59.07 seconds. Each warm inference took longer than the 3.571-second recording.

CPU returned “il reste deux virgule cinq kilos de tomates et douze bouteilles de lait” on both iterations. MPS returned “Présence de deux virgule cinq kilos de tomates et douze bouteilles de lait” on both. The opening words differed across devices and both retained the products and quantities.

The first MPS attempt loaded all components on MPS and then exited 134. The default Qwen3 SDPA path passed 32 query heads and eight KV heads to the PyTorch 2.5.1 MPS operator, which aborted on that native matmul shape. The small probe reproduces the same abort with BF16 tensors of shape `[4, 32, 1, 128]` and `[4, 8, 47, 128]`. Qwen3's installed eager implementation explicitly repeats the KV heads. The same probe then completed on MPS with maximum absolute difference from CPU of `3.815e-6`. The adapter uses the installed Transformers per-instance attention setter for the MPS decoder and checks that every decoder attention layer received `eager`. The CPU dispatch continues to use SDPA.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 "$HOME/.cache/speechflow/hojo-env/bin/python" \
  tools/hojo/mps_gqa_probe.py grouped-sdpa
PYTORCH_ENABLE_MPS_FALLBACK=0 "$HOME/.cache/speechflow/hojo-env/bin/python" \
  tools/hojo/mps_gqa_probe.py repeated-kv-eager
```

## Provisioning evidence

The first direct HF CDN attempt transferred approximately 1.3 MB before an HTTP/2 cancellation. Two bounded range probes returned partial payloads before timeout. The initial uv install timed out on shared cache locking for the `langid` source distribution. A private-cache retry hit PyPI transport timeouts. The shared lock selects SciPy 1.17.1 with NumPy 1.26.4. The offline lock-free resolution could not select that uncached wheel. The official HF Xet snapshot fetched 17 files. The local verifier confirmed that only the merged weight was missing among the 11 runtime assets as of its recorded run.

Xet allocated 4,164,296,704 bytes before its last completed body at 16:39 CET. The process continued receiving bytes and retrying HTTP 403 responses, with no additional durable blocks before handoff. An independent test of 12 official 512 KiB ranges confirmed exact HTTP 206 intervals and received 6,291,456 bytes in 12.973 seconds. The guarded APFS clone preserved 3,924 complete 1 MiB interiors after excluding boundaries and holes. A separate read-back verified all 48 MiB of the first staged checkpoint before the original Xet writer was stopped. The original sparse file remains intact. The continuous range writer saved complete parts with their digests, retained validated partial bodies in memory until their missing suffix arrived, and reduced concurrency on HTTP 429 or timeouts. It atomically published the exact 11,956,909,856-byte checkpoint after the full pinned SHA-256 passed. A separate local verifier confirmed all 11 runtime assets.

Comparable 270-second windows staged 269,484,032 unique bytes with 24 workers and 320,864,256 with 40 workers, at 0.997 MB/s and 1.185 MB/s respectively. Both windows recorded zero request retries and HTTP 429 responses. The 40-worker transfer continued from the verified checkpoint.

| Check | Status | Evidence |
| --- | --- | --- |
| Input format and SHA-256 | PASS | Jacques and two local FLEURS files matched their expected sample metadata and hashes. |
| Forced split and overlap review | PASS | Arithmetic and review flags tested with fabricated segment and transcript data. |
| Snapshot and pinned asset verification | PASS | The exact full-size merged weight passed SHA-256 and all 11 pinned runtime assets verified locally. |
| Frontend execution, NumPy | PASS | Ten weight-free cases on pinned Transformers 4.57.3 and NumPy 1.26.4 confirmed the 30-second bundle and 40-second override. |
| Frontend execution, PyTorch | PASS | PyTorch 2.5.1 paired with Transformers 4.57.3 confirmed the same ten cases. |
| Native CPU and MPS primitive probes | PASS | Float32 convolution and boolean indexing plus bfloat16 matmul, masked SDPA, and beam reorder ran on both requested devices with fallback disabled. |
| File decoding | PASS | The five real codec tests passed with the pinned imageio-ffmpeg wheel. Jacques and both FLEURS files normalized with matching source hashes and sample counts, using FFmpeg 7.1. |
| Real Silero | PASS | Silero VAD 6.2.0 detected speech in Jacques and both FLEURS recordings. A 65.768-second repeated synthetic input exercised two forced boundaries, produced at most 400,000 samples per chunk including padding, and flushed through sample 1,052,289 at EOF. |
| CPU inference | PASS | Two matching French transcripts returned using actual CPU weights. |
| MPS inference with eager decoder | PASS | Two matching French transcripts returned with MPS fallback disabled after the original grouped-SDPA abort was reproduced and isolated. |

- [Direct download log](download-weight.log)
- [Xet snapshot log](download-snapshot.log)
- [Locked dependency provisioning log](provision-locked.log)
- [Missing asset verification](verify-before-complete.log)
- [Completed pinned bundle verification](verify-complete.log)
- [Pinned wheel dispatch inspection](wheel-dispatch.json)
- [Jacques and two FLEURS validation WAV preflight checks](input-preflight.log)
- [Forced-boundary arithmetic with a mocked continuous VAD range](boundary-arithmetic.jsonl)
- [Frontend dependency preflight](frontend-preflight.log)
- [NumPy frontend verification](frontend-numpy.jsonl)
- [PyTorch frontend verification](frontend-torch.jsonl)
- [Native CPU primitives](primitives-cpu.jsonl)
- [Native MPS primitives](primitives-mps.jsonl)
- [Imported wheel versions and signatures](runtime-wheel.json)
- [Tokenizer, checkpoint header, and float WAV compatibility](weightless-metadata.json)
- [Silero segmentation of Jacques and two FLEURS WAVs](speech-segmentation.jsonl)
- [Published file probe with real FFmpeg and Silero](file-probe.jsonl)
- [Real Silero EOF and forced-boundary tests](real-silero-pytest.log)
- [Torch thread count after the real VAD import](vad-thread-regression.jsonl)
- [File/VAD dependency preflight](file-preflight.log)
- [Missing codec dependency IPC exit](codec-dependency-preflight.log)
- [Decoder fingerprint regression with a controlled subprocess](decoder-identity-test.jsonl)
- [Real codec tests](real-codec-pytest.log)
- [Jacques and FLEURS decoded with real FFmpeg](normalized-inputs.jsonl)
- [Audio contract tests](audio-contract-pytest.log)
- [Bounded PyPI range response](torch-range-probe.log)
- [Resumable PyTorch wheel transfer](torch-resumable.log)
- [Exact offline dependency install](provision-ranged-wheel.log)
- [Xet two-point allocation and socket measurements](xet-progress.jsonl)
- [Fresh bounded model-weight range](weight-late-range.log)
- [Parallel range validation and throughput](weight-range-batch.jsonl)
- [APFS stage and partial-response surrogate](weight-stage-continuation-surrogate.jsonl)
- [SHA mismatch guard and recovery of reused interiors](weight-stage-sha-mismatch.jsonl)
- [Initial staged checkpoint](weight-stage-checkpoint.log)
- [Read-back of all 48 MiB checkpoint parts](weight-stage-checkpoint-verified.jsonl)
- [Xet writer handoff evidence](xet-handoff.jsonl)
- [Continuous official range transfer](weight-stage-full.log)
- [Comparable throughput measurements](weight-stage-profiles.jsonl)
- [Completed 40-worker range transfer and publication](weight-stage-40.log)
- [CPU inference records](cpu.jsonl) and [raw stderr and process timings](cpu.log)
- [MPS inference records](mps.jsonl) and [raw stderr and process timings](mps.log)
- [Original MPS grouped-SDPA failure](mps-sdpa-failed.log) and [placement before the abort](mps-sdpa-failed.jsonl)
- [Grouped-SDPA primitive abort](mps-gqa-grouped-sdpa.log) and [repeated-KV eager primitive result](mps-gqa-repeated-kv-eager.jsonl)
