"""Bounded-window extraction with explicit carryover and at most one repair retry per window."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from pydantic import ValidationError

from ..config import Settings
from ..contracts import ExtractionResult, TranscriptSegment, ValidatedObservation
from .backend import BackendError, ChatBackend, OllamaBackend
from ..assets import DIGEST
from .prompt import (PROMPT_VERSION, SYSTEM_PROMPT, PromptSegment, estimate_messages, estimate_tokens,
                     render_user, repair_message)
from .schema import GENERATION_SCHEMA_VERSION, LLMOutput, generation_json_schema
from .validate import check_reply

# Accuracy-driven caps for an 8B model (the token budget below is the hard safety bound).
MAX_WINDOW_SEGMENTS = 8
MAX_WINDOW_CHARS = 1600
CONTEXT_SEGMENTS = 1  # previous target segments re-shown (read-only) at the start of the next window
MAX_KNOWN = 30  # earlier observations offered as correction targets; overflow is reported, not hidden
PROMPT_OVERHEAD_TOKENS = 128
KNOWN_RESERVE_TOKENS = 40  # planning reserve per carried observation; the real prompt is re-measured before sending


def generation_options(settings: Settings) -> dict:
    """Deterministic generation settings (Ollama /api/chat `options`)."""
    return {"temperature": 0, "seed": 0, "top_k": 1, "top_p": 1.0,
            "num_ctx": settings.context_tokens, "num_predict": settings.output_tokens}


@dataclass
class _Window:
    index: int
    context: list[TranscriptSegment]
    targets: list[TranscriptSegment]


def plan_windows(segments: list[TranscriptSegment], settings: Settings) -> tuple[list[_Window], list[str]]:
    """Greedy ordered packing under token/char/segment bounds. Never truncates a segment.

    `context` here is the worst case (previous window's tail); `extract` drops it when the previous
    window failed, so a failed window's text is never presented as "already processed".
    """
    base = (settings.context_tokens - settings.output_tokens - estimate_tokens(SYSTEM_PROMPT)
            - PROMPT_OVERHEAD_TOKENS - MAX_KNOWN * KNOWN_RESERVE_TOKENS)
    windows: list[_Window] = []
    errors: list[str] = []
    cur: list[TranscriptSegment] = []
    used = chars = 0
    prev_tail: list[TranscriptSegment] = []

    def cost(seg):
        return estimate_tokens(seg.normalized_text) + 16

    def flush():
        nonlocal cur, used, chars, prev_tail
        if cur:
            windows.append(_Window(len(windows), list(prev_tail), cur))
            prev_tail = cur[-CONTEXT_SEGMENTS:]
        cur, used, chars = [], 0, 0

    for seg in segments:
        if cost(seg) + sum(map(cost, prev_tail)) > base and cost(seg) + sum(map(cost, cur[-CONTEXT_SEGMENTS:])) > base:
            errors.append(f"segment {seg.segment_id} does not fit the {settings.context_tokens}-token context; "
                          "not truncated, not extracted")
            continue
        if cur and (len(cur) >= MAX_WINDOW_SEGMENTS or chars + len(seg.normalized_text) > MAX_WINDOW_CHARS
                    or used + cost(seg) + sum(map(cost, prev_tail)) > base):
            flush()
        # after a flush the context is the tail of the window just closed (not the stale earlier one)
        if cost(seg) + sum(map(cost, prev_tail)) > base:
            errors.append(f"segment {seg.segment_id} does not fit the {settings.context_tokens}-token context "
                          "together with its context; not truncated, not extracted")
            continue
        cur.append(seg)
        used += cost(seg)
        chars += len(seg.normalized_text)
    flush()
    return windows, errors


def _summary(obs: ValidatedObservation) -> str:
    bits = [obs.product_spoken or "?", obs.quantity_text or "quantité inconnue"]
    if obs.unit_spoken:
        bits.append(obs.unit_spoken)
    quote = " … ".join(e.quote for e in obs.evidence)
    return f"{' '.join(bits)}  [\"{quote}\"]"


def _fit(system, labels, ctx_labels, target_map, accepted, settings):
    """Largest carry list whose full prompt (upper-bound estimate) leaves the whole output budget free."""
    recent = accepted[-MAX_KNOWN:]
    dropped = len(accepted) - len(recent)
    while True:
        known = {f"k{i + 1}": o for i, o in enumerate(recent)}
        user = render_user([PromptSegment(k, s.normalized_text) for k, s in labels.items() if k in ctx_labels],
                           [PromptSegment(k, s.normalized_text) for k, s in target_map.items()],
                           [(k, _summary(o)) for k, o in known.items()])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        est = estimate_messages(messages)
        if est + settings.output_tokens <= settings.context_tokens:
            return messages, known, dropped, est
        if not recent:
            return messages, known, dropped, est  # caller reports overflow; nothing is truncated
        recent, dropped = recent[1:], dropped + 1


def extract(segments: list[TranscriptSegment], *, run_id: str, settings: Settings,
            backend: ChatBackend | None = None, expected_digest: str | None = None) -> ExtractionResult:
    """Extract observations from ordered transcript segments (see docs/contracts.md).

    Default backend: loopback Ollama, whose model manifest digest must equal the pinned digest
    (`assets.DIGEST`) before any transcript is sent. An injected `backend` (tests, other runtimes)
    is checked only when `expected_digest` is given.
    """
    ordinals = [s.ordinal for s in segments]
    if ordinals != sorted(set(ordinals)) or len({s.source_id for s in segments}) > 1:
        raise ValueError("segments must belong to one source and have strictly increasing ordinals")
    errors: list[str] = []
    usable = []
    for seg in segments:
        if seg.status == "error":
            errors.append(f"segment {seg.segment_id} not extracted: transcription error {seg.errors}")
        elif seg.status != "no_speech" and seg.normalized_text.strip():
            usable.append(seg)
    if not usable:
        return ExtractionResult(observations=[], status="error" if errors else "no_speech", errors=errors,
                                metrics={"windows": 0, "reason": "no extractable transcript text"})

    if backend is None:
        backend = OllamaBackend(settings.ollama_url, settings.ollama_model)  # loopback-validated
        expected_digest = expected_digest or DIGEST
    try:
        identity = backend.identify()
    except BackendError as exc:
        return ExtractionResult(observations=[], status="error", errors=errors + [f"extractor unavailable: {exc}"],
                                metrics={"windows": 0})
    if expected_digest is not None:
        found = str(identity.get("digest") or "").removeprefix("sha256:")
        if found != expected_digest.removeprefix("sha256:"):
            return ExtractionResult(
                observations=[], status="error", metrics={"windows": 0, "extractor": identity},
                errors=errors + [f"extractor digest mismatch: expected {expected_digest}, found {found or 'none'}; "
                                 "run `speechflow models prepare`. No transcript was sent."])
    schema = generation_json_schema()
    options = generation_options(settings)
    windows, plan_errors = plan_windows(usable, settings)
    errors += plan_errors
    segment_index = {s.segment_id: s for s in usable}

    accepted: list[ValidatedObservation] = []
    attempts: list[dict] = []
    failed_windows = 0
    totals = {"wall_s": 0.0, "prompt_tokens": 0, "output_tokens": 0, "load_ns": 0, "repairs": 0,
              "duplicates_skipped": 0, "context_only_dropped": 0, "estimate_misses": 0, "max_estimate_ratio": 0.0}
    t_start = time.perf_counter()
    previous_ok = False

    for win in windows:
        # Context is read-only text the previous window already extracted; never show unprocessed text as processed.
        context = win.context if previous_ok else []
        labels = {f"S{i + 1}": s for i, s in enumerate(context + win.targets)}
        ctx_labels = {k for k, s in labels.items() if s in context}
        target_map = {k: s for k, s in labels.items() if k not in ctx_labels}
        messages, known, dropped, est = _fit(SYSTEM_PROMPT, labels, ctx_labels, target_map, accepted, settings)
        base_rec = {"window": win.index, "run_id": run_id, "model": settings.ollama_model, "extractor": identity,
                    "prompt_version": PROMPT_VERSION, "schema_version": GENERATION_SCHEMA_VERSION,
                    "options": options, "segment_ids": [s.segment_id for s in win.targets],
                    "labels": {k: s.segment_id for k, s in labels.items()},
                    "known": {k: o.observation_id for k, o in known.items()},
                    "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                    "user_message": messages[1]["content"]}
        if dropped:
            errors.append(f"window {win.index}: {dropped} older observations not offered as correction targets "
                          "(prompt budget / carry limit)")
        problems: list[str] = []
        check = None
        if est + settings.output_tokens > settings.context_tokens:
            problems = [f"prompt upper bound {est} + output budget {settings.output_tokens} exceeds num_ctx "
                        f"{settings.context_tokens}; request not sent, nothing truncated"]
            attempts.append({**base_rec, "attempt": 0, "kind": "skipped_overflow", "estimate_tokens": est,
                             "errors": problems})
        else:
            for attempt_no in (1, 2):
                rec = {**base_rec, "attempt": attempt_no, "kind": "initial" if attempt_no == 1 else "repair",
                       "estimate_tokens": est, "repair_message": messages[-1]["content"] if attempt_no == 2 else None}
                try:
                    reply = backend.chat(messages, schema, options=options, timeout=settings.extraction_timeout)
                except BackendError as exc:
                    rec.update(error=str(exc), kind_error=exc.kind)
                    attempts.append(rec)
                    problems = [f"backend {exc.kind}: {exc}"]
                    break  # transport failures are not repaired by re-prompting
                rec.update(content=reply.content, done_reason=reply.done_reason, wall_s=reply.wall_s,
                           timings_ns=reply.timings_ns, prompt_tokens=reply.prompt_tokens,
                           output_tokens=reply.output_tokens, response_raw=reply.raw)
                totals["wall_s"] += reply.wall_s
                totals["prompt_tokens"] += reply.prompt_tokens or 0
                totals["output_tokens"] += reply.output_tokens or 0
                totals["load_ns"] += reply.timings_ns.get("load_duration") or 0
                if reply.prompt_tokens:
                    totals["max_estimate_ratio"] = max(totals["max_estimate_ratio"], reply.prompt_tokens / est)
                    if reply.prompt_tokens > est:
                        totals["estimate_misses"] += 1
                        errors.append(f"window {win.index}: token estimate {est} below actual prompt_tokens "
                                      f"{reply.prompt_tokens}; estimator must be recalibrated")
                problems, check = _validate_reply(reply, settings, run_id, labels, ctx_labels, known, accepted,
                                                  segment_index)
                rec["errors"] = problems
                attempts.append(rec)
                if not problems:
                    break
                if attempt_no == 1:
                    repaired = messages + [{"role": "assistant", "content": reply.content},
                                           {"role": "user", "content": repair_message(problems)}]
                    r_est = estimate_messages(repaired)
                    if r_est + settings.output_tokens > settings.context_tokens:
                        problems = problems + [
                            f"repair not sent: prompt with the prior answer (upper bound {r_est}) + output budget "
                            f"{settings.output_tokens} exceeds num_ctx {settings.context_tokens}"]
                        rec["errors"] = problems
                        attempts.append({**base_rec, "attempt": 2, "kind": "repair_skipped_overflow",
                                         "estimate_tokens": r_est, "errors": problems})
                        break
                    totals["repairs"] += 1
                    messages, est = repaired, r_est
        if problems:
            failed_windows += 1
            previous_ok = False
            errors.append(f"window {win.index} (segments {[s.segment_id for s in win.targets]}) failed "
                          f"(attempts used: {len([a for a in attempts if a['window'] == win.index and 'content' in a or a['window'] == win.index and 'error' in a])}): "
                          f"{problems[:6]}; raw replies retained in attempts")
            continue
        assert check is not None
        previous_ok = True
        totals["duplicates_skipped"] += check.skipped_duplicates
        totals["context_only_dropped"] += check.context_only_dropped
        errors += [f"window {win.index}: {e}" for e in check.reply_errors + check.replaced_notes]
        if check.replaced:
            accepted = [o for o in accepted if o.observation_id not in set(check.replaced)]
        accepted.extend(check.observations)

    review = any(o.status == "needs_review" for o in accepted)
    if failed_windows == len(windows):
        status = "error"
    elif failed_windows or errors or review:
        status = "needs_review"
    else:
        status = "ok"
    metrics = {"extractor": identity, "windows": len(windows), "failed_windows": failed_windows,
               "attempts": len(attempts), "repairs": totals["repairs"], "model_wall_s": round(totals["wall_s"], 4),
               "total_wall_s": round(time.perf_counter() - t_start, 4), "prompt_tokens": totals["prompt_tokens"],
               "output_tokens": totals["output_tokens"], "load_duration_ns": totals["load_ns"],
               "duplicates_skipped": totals["duplicates_skipped"],
               "context_only_dropped": totals["context_only_dropped"],
               "estimate_misses": totals["estimate_misses"], "max_actual_to_estimate_ratio": round(totals["max_estimate_ratio"], 3),
               "options": options, "prompt_version": PROMPT_VERSION, "schema_version": GENERATION_SCHEMA_VERSION}
    return ExtractionResult(observations=accepted, status=status, errors=errors, attempts=attempts, metrics=metrics)


def _validate_reply(reply, settings, run_id, labels, ctx_labels, known, accepted, segment_index):
    if reply.done_reason == "length":
        return ["output truncated at the output-token budget (done_reason=length)"], None
    if reply.prompt_tokens is not None and reply.prompt_tokens + (reply.output_tokens or 0) >= settings.context_tokens - 8:
        return [f"context window exhausted (prompt_tokens={reply.prompt_tokens}, output_tokens={reply.output_tokens}, "
                f"num_ctx={settings.context_tokens}); the prompt or output may have been truncated"], None
    try:
        parsed = LLMOutput.model_validate(json.loads(reply.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        detail = exc.errors(include_url=False)[:5] if isinstance(exc, ValidationError) else str(exc)
        return [f"reply is not valid per schema: {json.dumps(detail, default=str, ensure_ascii=False)[:600]}"], None
    check = check_reply(parsed, run_id=run_id, labels=labels, context_labels=ctx_labels, known=known,
                        already=accepted, segment_index=segment_index)
    return check.repairable, check
