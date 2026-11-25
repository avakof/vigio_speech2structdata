# Target Mac and pinned runtime

Commands were executed in the project checkout on the local arm64 MacBook Pro. `sw_vers` returned macOS 27.2.1 (26B131). `sysctl hw.memsize machdep.cpu.brand_string` returned 51,539,607,552 bytes and Apple M3 Max. System Profiler reported 40 GPU cores and Metal 4. The data volume had 748 GiB available at discovery. GPU availability alone does not establish a working Hojo backend.

Orca ADE 1.4.274, Codex CLI 0.171.0 and Claude Code 2.1.329 were reachable and authenticated for development. The application uses local installed packages, local model files and a numeric-loopback Ollama service. The installed Ollama server reported 0.40.16. The operating system supplied French speech synthesis, audio input devices and process-scoped Seatbelt network policies.

## Dependency and artifact identity

| Component | Pin or provenance |
|---|---|
| Python | Native CPython 3.12.16. The project lock targets Python 3.12 on Darwin arm64. |
| PyTorch / Torchaudio / Transformers | 2.5.1 / 2.5.1 / 4.57.3 |
| Hojo Python distribution | 0.1.3, wheel SHA-256 `8c3e39b55b77ff60bb6232ede09375a4e799233452ee0cfd7a77558fd6e0780a` |
| Hojo weights | HojoAI/Hojo-ASR-Multi-V1, Apache-2.0, revision `aa1d0dfc369deb3b3235572fa151acd7b498481f`. Expected 11,956,909,856 bytes, SHA-256 `c046a5a26c142f5aded7331d36892293a025f579e342a27861e80509afaca88d`. |
| Silero | silero-vad 6.2.0 wheel, bundled JIT artifact. Runtime records its SHA-256. |
| Local extractor | Ollama qwen3:8b. Expected manifest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`. |
| Decoder | Local FFmpeg, with a native binary supplied by imageio-ffmpeg 0.6.0. System FFmpeg was absent at discovery. |

The upstream class accepts `load_model(folder_path, device, *, cache_dir, revision, token, local_files_only)` and `run_infer(audio_input, *, seed, batch_size, cuda_enabled)`. The bundle stores a 30-second frontend setting. The wheel constructs a 40-second frontend. The adapter has no language or timestamp parameter, uses batch size one, and retains the upstream result dictionary. The package enables fast-loading patches at import. Its encoder overrides the initially selected attention implementation. The explicit MPS autocast and mixed-precision boundaries are isolated in the ASR child. [Pinned model card](https://huggingface.co/HojoAI/Hojo-ASR-Multi-V1/tree/aa1d0dfc369deb3b3235572fa151acd7b498481f) · [Hojo wheel](https://pypi.org/project/hojo-asr/0.1.3/)

## Verification

`uv lock --check --offline` validated the frozen 71-package resolution. The first full install timed out transferring the Silero wheel. A constrained retry encountered the audio feasibility install's cache lock. The integrator assigned that single installer ownership of wheel transfers. The editable CLI and the validation/test stack installed from local caches. The raw first runtime attempt demonstrates the two process-specific network controls and identifies its missing decoder dependency. [File attempt](offline/file-attempt-1.json) · [Runtime error](offline/file-attempt-1.stderr)

The independent backend review verified the complete checkpoint header from a partial transfer and matched the prefix against the publisher mirror. The header describes mixed float32 audio and bfloat16 decoder weights. MPS execution, CPU throughput, peak inference memory and recognition quality require completed model runs. [Hojo evidence](hojo/README.md) · [Acceptance state](acceptance.json)
