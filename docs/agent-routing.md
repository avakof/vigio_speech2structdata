# Agent routing

All development work ran in the current checkout, `/Users/vakof/orca/vigio_speech2structdata`, under Orca ADE 1.4.274. The repository began on an unborn `main`. Workers used disjoint paths. The integrator owned public contracts, model manifest, dependency lock, pipeline, integration and acceptance. Orca run `run_3d8c7e339af1` used at most three active workers. Model loads require the integrator's exclusive slot.

| Role | Provider and effective model | Requested / recorded effort | Provider session | Orca session |
|---|---|---|---|---|
| Integrator | OpenAI Codex · gpt-6.2-astra | xhigh / xhigh | 01a21308-a0fb-79e0-8578-142793ef6b02 | b91cb6b2-44b1-4506-aaa2-3deccca4dc90 |
| Independent architect | OpenAI Codex · gpt-6.2-astra | xhigh / xhigh | 01a21309-4f14-7a32-a64e-3149c10a5287 | c69b8766-aa30-442e-b766-73b3c2686b06 |
| Audio and Hojo | OpenAI Codex · gpt-6.2-sol | high / high | 01a2130a-657d-7421-8af1-24f3ce8db31b | 8666ee97-35a9-4b30-b66f-8d4252fca38e |
| Extraction | Anthropic Claude Code · claude-sonnet-5-5 | high / high per provider request | 8dfa5484-1eb7-490b-8548-e305c8ac6263 | abb9e5a4-928c-46c0-b348-68c0fe2bda51 |
| CLI and persistence | Anthropic Claude Code · claude-sonnet-5-5 | high / high | edd4b6b4-5ea1-432e-9414-36420c2ada9d | f3c76094-c346-468f-90ea-082d2257566d |
| Independent QA | OpenAI Codex · gpt-6.2-sol | high / high | 01a2131e-c8de-7ba0-9b43-cbaef0edc579 | e04e8453-60f5-4a76-bc1f-080beedeebaf |
| Independent reviewer | Anthropic Claude Code · claude-opus-5-5 | high / high | 2061ab3a-a0af-4592-bb5b-8e25df20eb8a | 7d26416e-e5ee-4c68-bb42-5cca16f4830e |

**Routing evidence.** The Codex CLI was 0.171.0 with an authenticated ChatGPT session. Claude Code 2.1.329 reported first-party authentication. The integrator and architect independently matched `CODEX_THREAD_ID` to the read-only `threads` row and the actual rollout `turn_context`. The initial architect turn was `01a21309-4f63-7f62-8672-f7dad8d16aad`. Its backend follow-up remained in the same provider session at turn `01a2131b-c55f-7263-8d7c-73bd331aa8fb`. The integrator's initial turn was `01a21308-a50d-7482-ad6a-13bcf9c74ed5`.

The extraction worker reported a numeric effort tag of 10 and initially interpreted it as low. The installed orchestration guide directs workers to use named launch effort. Its dispatch and Claude request log both record high, and response records identify the requested Sonnet model. CLI/persistence recorded the named launch argument and provider model. The Astra/max relay (`01a21306-8f0b-7c63-bc9a-2d71c89eec5d`) supplied discovery, licensed clips, OS network-control evidence and user communication.

Dispatches were `ctx_95f1443d1968` (architecture), `ctx_b4805671dbf7` (backend), `ctx_7b7c13bf9c73` (audio), `ctx_69cd1950b284` and `ctx_3c72523b7e7d` (extraction), `ctx_eb50cc5867cd` (CLI), and `ctx_a737e7f5a715` (QA).

The same independent Astra/xhigh reasoner reviewed the real Ollama backend at dispatch `ctx_0688a7d0f712`. It verified the pinned layer and manifest, reproduced product/quantity and lost-correction defects with clipped evidence, and requested a bounded source-backed repair. The Sonnet persistence worker finished its concurrency changes and tests before its provider reported a five-hour usage limit with a stated 20:00 local reset. The integrator ran and repaired the actual Qwen pipeline while independent GPT QA reran the Claude-authored validator and storage. The original Opus implementation review was dispatch `ctx_6fb7cf719a3c`.

Independent QA dispatch `ctx_5252b986aa1e` repeated the installed-CLI reference evaluation in a fresh directory on the same Mac. All 15 observation sets matched, and the runner unloaded. The integrator completed the second extraction probe before granting the audio engineer the exclusive CPU/MPS feasibility slot. [Independent command and observations](../reports/qa/real-qwen-independent.md)

At follow-ups `ctx_be8cd7ae9502` and `ctx_90ba1eac6b3c`, the original independent architect checked the raw pinned replies and reproduced repair omissions, including two flour counts at different locations. The integrator retained the first attempt's grounded fields and required separate source occurrences to be accounted for during repair. [Regression outcomes](../reports/integration/repair-preservation-after.txt)

Independent backend dispatch `ctx_c084ff78ca83` identified the cached decoder's 32-query/8-KV mismatch from the MPS native abort and verified the pinned Transformers and PyTorch source hashes. The audio engineer independently reproduced it with weight-free tensors, applied the instance-local decoder setting and repeated the real model on the Mac. [MPS raw records](../reports/hojo/mps.jsonl)

The relay confirmed the user's choice of file input only in messages `msg_5b26459d0173` and `msg_7238cc0d60a4`. No physical microphone test, permission request or audible playback is authorized. Synthetic callback transport is a separate test on the same processing path. Actual device queue and latency measurements are blocked by that choice.

The original architect diagnosed the repeated long-file response at `ctx_2c4085a17497` and checked the two shorter cuts at `ctx_dd719c578e4d`. Its interim acceptance review at `ctx_78506ee85204` independently verified the source files in the built wheel, both real replay databases, all five rendered files across first/replay/export sets for CPU and MPS, the recorded word errors and the blocked physical gates. Independent QA's `ctx_d5a02e21ae28` ran the complete native CPU suite and the dedicated MPS replay. [Identity receipts](../reports/integration/astra-routing.json) · [Independent test report](../reports/qa/final-real-suite.md)

After the two short prerecorded French cases, the original architect at `ctx_a740d8a48d1f` and independent QA at `ctx_0e5d893d445d` reproduced the unmarked empty extraction from a damaged negation. The integrator added a source-span diagnostic, and QA at `ctx_0ecfe3311541` verified its negative controls and the unchanged linked correction. The original Opus release-audit attempt at `ctx_002f36402c51` encountered the provider's five-hour limit with a 20:00 Paris reset. [QA regressions](../reports/qa/negated-quantity-regression.md) · [Real speech rerun](../reports/evaluation/semantic-audio-reviewed.json)
