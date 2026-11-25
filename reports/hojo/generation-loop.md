# Hojo generation cap and loop review

File and microphone inference share a check for a filled generation budget. For each chunk the existing Hojo encoder hook records the speech embedding length. A temporary wrapper on the loaded decoder instance forwards the original arguments unchanged and returns the original generated tensor. It observes the actual `max_new_tokens` argument, length of the one selected sequence, and whether its last token equals the EOS supplied by Hojo. The wrapper restores the original instance method after the call, including on failure. The raw Hojo reply, key, normalized text, and sample ranges are preserved.

Transformers 4.57.3 starts this pinned decoder's embedding-only path with an empty ID sequence and crops the selected beam to its generated length. Its output count includes terminal EOS when one is present. A reply that fills the budget gets `hojo_generation_cap_reached` and `needs_review` even if it has no repeated text or its final token is EOS. A selected sequence below the bound passes this check. The measured token count describes the selected beam and does not expose why beam search stopped or whether the transcript ended naturally.

If the output IDs are unavailable, the check retokenizes the returned text and uses a tolerance of four tokens on either side of the per-call limit. When the decoder argument is unavailable, the Hojo 0.1.3 formula computes the limit from the observed embedding length, `max(10, min(configured_cap, 2 * embedding_length + 10))`. Archived or scripted records lacking both observations can only use the configured 200-token upper bound. The fallback warning is `hojo_possible_generation_cap`. The wheel strips selected markers and trims text, so retokenization is a proxy.

The separate `hojo_possible_generation_loop` note still requires three consecutive exact copies of a token block with a period of at least 16 tokens near the limit. Genuine speech can repeat. The cap and loop notes qualify evidence for review without assigning model confidence. Existing extraction validation retains observations that cite the segment and propagates the review flag to them.

| Weight-free input | Result |
| --- | --- |
| Archived middle reply, text only | Review by proxy, with a separate 48-token loop note. Four observations at distinct spans are retained. |
| 200 distinct selected IDs or a period-4 loop at 200 | Review from the observed budget. |
| Selected 110-token loop at a dynamic cap of 110 | Review from the observed budget, with a loop note. |
| Short EOS and EOS one token before the cap | Clear under this rule. |
| EOS at the cap | Review, with EOS presence explicit. |
| No usable decoder ID output, 108 retokenized IDs near a dynamic cap of 110 | Review by proxy. |
| Two short repeated statements and decomposed-accent Unicode | Clear. Repeated occurrences and normalized evidence spans resolve separately. |

The weight-free suite passed 101 tests. The replay used the pinned local tokenizer, archived replies, and scripted model output for the controls. Extraction chat replies were scripted and validated through production extraction. The microphone test checks the shared diagnostic through the real ASR queue and provisional output. [Reviewer reproducer](../review/repro/r06_generation_cap.py), [archived reply](../qa/real-full-long-reproducer.json), [weight-free test log](generation-cap-pytest.log).

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -p no:cacheprovider -q tools/hojo/test_generation_loop.py \
  tools/hojo/test_microphone.py tests/qa/test_audio_contract.py tests/qa/test_semantics.py
```

## Pinned inference verification

The production implementation SHA-256 was `23096a70df38af25f14f4275c9c0d0935f57d6d9a55ec421794a91906902e867` before and after these probes. The diagnostic captured the first three selected IDs from the original generated tensor before Hojo decoded it. The Jacques WAV hash was `d92c0759b7d4d3e03ae22eb3c47aa97ef7122789eac187c7cf42e8e6035d1330`. The archived 25-second WAV hash was `667925ed176b87a177c2f8073636e21c6221084ab7e296f7a12e101b16cf7b57`. Weight and config pins were verified before each load, the decoder was bfloat16 on the named device, and MPS fallback was disabled. Hojo's encoder, bottleneck, and speech normalization stayed float32 on the same device. The CPU eager contrast used the same loaded CPU model and changed only the diagnostic instance before it exited. [CPU raw JSONL](generation-cap-cpu.jsonl), [MPS raw JSONL](generation-cap-mps.jsonl), [CPU stderr](generation-cap-cpu.log), [MPS stderr](generation-cap-mps.log), [exit codes](generation-cap-exits.log).

| Call | Embedding frames | Selected IDs / cap | Terminal EOS | First three IDs and token spellings | Inference |
| --- | ---: | ---: | --- | --- | ---: |
| CPU SDPA, Jacques | 45 | 23 / 100 | Yes | `321 65067 25552` · `il Ġreste Ġdeux` | 13.726 s |
| CPU SDPA, 25-second center | 313 | 200 / 200 | No | `642 3958 4578` · `les ĠÃ© tag` | 91.114 s |
| CPU eager, Jacques | 45 | 23 / 100 | Yes | `321 65067 25552` · `il Ġreste Ġdeux` | 10.190 s |
| MPS eager, Jacques | 45 | 25 / 100 | Yes | `2340 5397 763` · `ĠPr Ã©s ence` | 7.294 s |

The long center raw record matched the archive exactly. It filled its measured cap and got both the exact-budget warning and the separate 48-token loop note. The three Jacques calls ended below the dynamic cap with EOS and stayed clear under this rule. The CPU runs opened with “il reste deux” and the MPS run opened with “Présence de deux.” Both quantities and products were preserved for this recording.

```sh
/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/probe_generation_cap.py cpu --long-center --cpu-eager \
  >reports/hojo/generation-cap-cpu.jsonl 2>reports/hojo/generation-cap-cpu.log
/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/probe_generation_cap.py mps \
  >reports/hojo/generation-cap-mps.jsonl 2>reports/hojo/generation-cap-mps.log
```

## Identical-embedding prefill

The CPU probe saved the exact generation arguments at the Hojo decoder boundary and ran the model's native beam search through its first decoder forward. The MPS probe loaded the same saved CPU BOS-plus-speech embeddings and mask, then let Transformers prepare its native prefill. An output hook stopped each run after the first decoder forward, before beam search consumed the raw logits. The four-beam input embedding, attention mask, position IDs, and cache position hashes all matched. The eager bfloat16 decoder ran on the named device, and the raw logits were moved to CPU for ranking. Both children exited 0 after exactly one forward each. [CPU prefill evidence](generation-prefill-cpu.jsonl), [MPS prefill evidence](generation-prefill-mps.jsonl), [CPU stderr](generation-prefill-cpu.log), [MPS stderr](generation-prefill-mps.log), [exit codes](generation-prefill-exits.log).

Each beam had the same eight ranked token IDs on both devices, `321 8669 263 3756 73 98934 295 88`. Both ranked `321` (`il`) first at 24.125 and `8669` (`ils`) second at 18.000, a margin of 6.125. Some lower-ranked logit values differed. First-forward time was 4.617 s on CPU and 1.691 s on MPS. This test covers the first decoder prefill on identical embeddings. It leaves the contributions of the native audio path and subsequent beam steps unresolved.

```sh
prefill_snapshot="$HOME/.cache/speechflow/generation-prefill/jacques-cpu-embeddings.pt"
/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/probe_prefill.py cpu --snapshot "$prefill_snapshot" \
  >reports/hojo/generation-prefill-cpu.jsonl 2>reports/hojo/generation-prefill-cpu.log
/usr/bin/time -l env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTHONUNBUFFERED=1 .venv/bin/python tools/hojo/probe_prefill.py mps --snapshot "$prefill_snapshot" \
  >reports/hojo/generation-prefill-mps.jsonl 2>reports/hojo/generation-prefill-mps.log
```
