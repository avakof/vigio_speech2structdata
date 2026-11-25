# Exact-crop Hojo diagnostic

The three inputs are the unpadded sample ranges of previously missed utterances from the original long Jacques file. Production FFmpeg decoding regenerated the entire archived float32 PCM byte for byte. Each exact PCM SHA-256 matched the predeclared digest, and each neutral WAV was read back and verified. The WAVs use a deterministic IEEE float header. The first preparation attempt used libsndfile's timestamped PEAK header, which changed the WAV byte digest even though the PCM matched. [Input ranges and both digest sets](exact-crop-inputs.jsonl), [first header check](exact-crop-prepare-header-failed.log).

| Input | Source samples at 16 kHz | Duration |
| --- | --- | ---: |
| A | `[486400, 511200)` | 1.55 s |
| B | `[645760, 672320)` | 1.66 s |
| C | `[806880, 834880)` | 1.75 s |

This diagnostic isolates audio from its surrounding context. Its results do not establish general recognition accuracy or select production segment boundaries. Preparation and scoring are separate from inference. Each backend command loads the pinned bundle once, runs exactly one call per verified input in A/B/C order, and records the returned text and active attention placement. The score step reads the previously declared gold only after both logs contain exactly three completed calls.

```sh
.venv/bin/python tools/hojo/exact_crop_probe.py prepare >reports/hojo/exact-crop-inputs.jsonl

/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/exact_crop_probe.py run cpu \
  >reports/hojo/exact-crop-cpu.jsonl 2>reports/hojo/exact-crop-cpu.log

/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/exact_crop_probe.py run mps \
  >reports/hojo/exact-crop-mps.jsonl 2>reports/hojo/exact-crop-mps.log

.venv/bin/python tools/hojo/exact_crop_probe.py score \
  reports/hojo/exact-crop-cpu.jsonl reports/hojo/exact-crop-mps.jsonl \
  >reports/hojo/exact-crop-score.jsonl
```

## Results

Both children exited 0 and returned exactly three keyed replies. All 36 decoder layers used 32 query heads, eight KV heads, and dimension 128. Q, K, V, and output projection weights were bfloat16 on the requested device. Pre-hooks on the first and last attention layers confirmed live bfloat16 activations on that device for each crop. CPU used SDPA. MPS used eager attention, with fallback disabled. The pinned Hojo generation settings matched between backends, including beam size 4 and the configured 200-token cap. The model loaded in 9.55 s on CPU and 15.01 s on MPS. [CPU JSONL](exact-crop-cpu.jsonl), [MPS JSONL](exact-crop-mps.jsonl), [exit status](exact-crop-exit.log), [CPU stderr and timings](exact-crop-cpu.log), [MPS stderr and timings](exact-crop-mps.log).

| Input | Expected phrase | CPU reply | CPU time | MPS reply | MPS time |
| --- | --- | --- | ---: | --- | ---: |
| A | neuf kilos de pommes | j'ai compté nos kilomètres de route | 4.29 s | ! je compte de nouveau kilo de riz | 5.66 s |
| B | cinq kilos de poires | je commence à croire que je vais mourir | 4.27 s | disguised as a kilo of cocaine | 4.34 s |
| C | trois kilos de kiwis | je compte trois kilos de crêpes | 4.58 s | -rays kilos de criolite | 1.88 s |

Each backend recovered 0 of 3 original quantity, unit, and product phrases. The six replies retokenized to 7–10 tokens. The expected phrases remained absent when the marked audio was presented without padding or surrounding speech. [Post-run scoring](exact-crop-score.jsonl).
