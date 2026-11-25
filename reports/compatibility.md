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

The final missing Python dependency was the native torch wheel. The recovery fetched 244 checked HTTP ranges, assembled 63,858,109 bytes and matched SHA-256 `8c712df61101964eb11910a846514011f0b6f5920c55dbf567bff8a34163d5b1`. All 71 packages now pass `uv sync --offline --frozen` and `uv pip check`. Native CPU and Metal primitive probes passed with fallback disabled. Ten cases exercised the real frontend in both NumPy and PyTorch. The CPU and MPS Hojo model gates remain separate from those operator checks. [Frozen environment](integration/frozen-environment.txt) · [CPU primitives](hojo/primitives-cpu.jsonl) · [MPS primitives](hojo/primitives-mps.jsonl) · [Frontend](hojo/frontend-torch.jsonl)

The installed CLI used FFmpeg 7.1 and the pinned Silero JIT asset (SHA-256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`). Seven real audio tests passed across the four file formats, EOF flushing, silence and forced duration boundaries. An actual silence import and replay shared one generation and the CLI CSV matched the saved CSV byte-for-byte. [Audio checks](integration/native-audio-tests.txt) · [Silence replay](integration/silence-replay.json)

## Local extractor

The pinned Ollama artifact completed on the Mac. Its 5,225,374,496-byte Q4_K_M layer has SHA-256 `a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`. Ollama 0.40.16 appends one reference to the original 859-byte registry manifest when it writes its local tag. Verification checks the registry bytes, the exact relationship between the two descriptors, all five underlying config/model/template/license/parameter blobs and the local API tag. [Artifact hashes](integration/ollama-artifacts.json) · [Independent registration evidence](integration/ollama-registration.json)

Real local generation on the exact French inventory sentence returned 2.5 kg of tomatoes and 12 bottles of milk. The feasibility probe recorded a 0.783 s load and a 6.568 s total Ollama duration on its first 302-token reply. The API reported 4,795,903,507 resident accelerator bytes at a 4,096-token context and 5,238,423,551 at an 8,192-token context. Both the initial and final runner lists were empty. The generic schema produced valid outputs for all five fixtures after two repairs. Its `uniqueItems` probe produced a counterexample, so independent validation remains necessary. [Raw fixture](extraction/run-001/raw/fixture_inventory.json) · [Timings and schema probes](extraction/run-001/summary.json)

## Measurement definitions

* Pipeline real-time factor divides wall time from entry to `pipeline.process` through finalized extraction by the decoded source duration, including its silence. SQLite writes and export rendering follow this measurement.
* File ASR real-time factor divides the wall time spent processing the submitted segments on a loaded model by the entire decoded source duration. It excludes VAD and model load. The feasibility probe reports first and warm inference on the same short input and synchronizes MPS before recording completion.
* Peak process RSS uses Darwin bytes. System available memory, RSS and Metal allocator readings describe different views of unified memory and are reported individually.
* WER uses word-level edit distance divided by human-reference word count. Each evaluation report contains its token normalization policy, included corpus duration, coverage and recording condition. Field scores and abstentions are computed independently on reference and ASR transcripts.

The production extractor initially missed a same-segment antecedent and confused the contents of a carton with its outer count. The independent architect also reproduced false acceptances from clipped quotes. After the validator and generation changes, the installed CLI processed 15 curated reference cases in 70.59 s using 17 local service calls. It covered 16/16 gold observations and routed 3/19 returned candidates to review. Accepted-field precision and recall were 1.0 on this development set, including 15 quantities, 15 units, one negation and two correction links. The report retains raw service responses, evidence spans and zero-count denominators. [Initial evaluation](evaluation/reference-baseline.json) · [Final reference evaluation](evaluation/reference-final.json) · [Independent regression evidence](qa/inventory-context.md)

## Measured Hojo backends

Both measurements used the same complete pinned checkpoint, the same 57,142-sample French Jacques recording, one model instance and two calls in a fresh native process. The encoder and bottleneck used float32 and the decoder used bfloat16. The original MPS SDPA path aborted when a cached 32-query-head tensor reached matmul with eight KV heads. Qwen's built-in eager attention on this decoder instance explicitly repeats the KV heads. The independent shape reproducer agreed with its CPU reference within 0.00000382. [Original native failure](hojo/mps-sdpa-failed.log) · [Primitive comparison](hojo/mps-gqa-repeated-kv-eager.jsonl)

| Backend | Cold load | First inference | Warm inference | Warm inference / input duration | Peak process RSS | Metal driver allocation after warm call |
|---|---:|---:|---:|---:|---:|---:|
| CPU, decoder SDPA | 7.370 s | 6.808 s | 8.751 s | 2.450 | 12,767,412,224 bytes | n/a |
| MPS, decoder eager, fallback disabled | 16.526 s | 27.339 s | 6.761 s | 1.893 | 3,745,595,392 bytes | 12,791,005,184 bytes |

CPU returned “il reste deux virgule cinq kilos de tomates et douze bouteilles de lait” twice. MPS rendered the opening as “Présence de” and preserved both inventory facts on both iterations. The application defaults to CPU. The single-utterance timings exclude asset hashing and cannot establish sustained microphone throughput. [CPU raw records](hojo/cpu.jsonl) · [MPS raw records](hojo/mps.jsonl)

The first application run on an original six-second FLEURS recording completed on CPU in 83.20 s, including 33.69 s of ASR. Its non-inventory transcript yielded an empty list from actual local Qwen generation. A positive inventory test exposed a 160 ms intra-phrase pause that had split “deux virgule cinq” into two short inputs and led to an incorrect accepted 5 kg. A five-point real Silero sweep identified a 350 ms silence interval that kept the entire 3.571375 s utterance together. With that endpoint setting, the installed CPU pipeline finished in 34.98 s, returned the full spoken sentence and exported 2.5 kg of tomatoes and 12 bottles of milk. [Human file](integration/first-human-file.json) · [Endpoint sweep](integration/vad-short-pauses.json) · [Positive file](integration/jacques-350-manifest.json) · [Exact SQLite/export comparison](examples/jacques/sqlite-export-check.json)

## French speech accuracy

The installed CPU CLI evaluated the two preselected French FLEURS dev clips using their corpus-verified references. Across 20.46 seconds and 61 reference words, Hojo made two edits, for a corpus WER of **3.28%**. The 14.46-second clip had 0/38 edits and the six-second clip had 2/23. The corpus selection supplies no clean/noisy labels. With two examples, this is a smoke measurement. The report includes the exact normalization policy, hypotheses and clip-level scores. [Evaluation](evaluation/fleurs-real.json) · [Corpus selection and license](../evaluation/fleurs-fr/provenance.json)
