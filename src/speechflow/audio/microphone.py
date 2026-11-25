"""Segmented microphone capture with a recoverable normalized audio spool."""

from collections import deque
from hashlib import file_digest
import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import resource
import signal
import sys
from threading import Thread, current_thread, main_thread
import time
from uuid import uuid4

import numpy as np

from ..config import Settings
from ..contracts import AudioResult, AudioSource
from . import _error_segment, _flag_reemissions, _infer_segment, _model_placement
from .capture_device import InputBuffer, MicrophoneError, input_device
from .decoding import SAMPLE_BYTES
from .hojo import SAMPLE_RATE, load_local_model, verify_local_bundle
from .resampling import StreamingResampler
from .segmentation import (
    SileroStream, VAD_FRAME_SAMPLES, VAD_MIN_SILENCE_MS, chunk_limits, make_segment,
    silero_provenance,
)

log = logging.getLogger(__name__)
ASR_QUEUE_BLOCKS = 8


class _LiveSegments:
    def __init__(self, source_id, settings, submit):
        self.source_id = source_id
        self.submit = submit
        self.core_limit, self.padding = chunk_limits(settings.max_chunk_seconds, settings.padding_ms)
        self.vad = SileroStream()
        self.guard = VAD_MIN_SILENCE_MS * SAMPLE_RATE // 1000 + VAD_FRAME_SAMPLES
        self.cursor = 0
        self.pending = deque()
        self.segments = []

    def feed(self, data, sample_count, *, final=False):
        ranges = self.vad.finish() if final else self.vad.feed(data)
        for start, end in ranges:
            start = max(start, self.cursor)
            for left in range(start, end, self.core_limit):
                right = min(left + self.core_limit, end)
                boundary = "duration_limit" if right < end else ("eof" if final else "vad_endpoint")
                self.pending.append((left, right, boundary))
            self.cursor = max(self.cursor, end)
        if not final and self.vad.active_start is not None:
            start = max(self.cursor, self.vad.active_start)
            confirmed = min(self.vad.processed - self.guard, sample_count - self.padding)
            while start + self.core_limit <= confirmed:
                end = start + self.core_limit
                self.pending.append((start, end, "duration_limit"))
                start = self.cursor = end
        while self.pending and (final or self.pending[0][1] + self.padding <= sample_count):
            start, end, boundary = self.pending.popleft()
            segment = make_segment(
                self.source_id, len(self.segments), start, end, sample_count, self.padding
            )
            self.segments.append(segment)
            self.submit(segment, boundary)


class _QueuedASR:
    def __init__(self, settings, pcm, buffers):
        self.settings = settings
        self.pcm = pcm
        self.buffers = buffers
        self.queue = Queue(maxsize=ASR_QUEUE_BLOCKS)
        self.records = {}
        self.timings = []
        self.backend, self.dtype = "not_loaded", "not_loaded"
        self.peak_depth = 0
        self.unscheduled = 0
        self.failed = None
        self.load_seconds = None
        self.verification_seconds = None
        self.thread = Thread(target=self._run, name="speechflow-mic-asr")

    def submit(self, segment, boundary):
        queued = time.monotonic()
        try:
            if self.failed:
                raise Full
            self.queue.put_nowait((segment, boundary, queued))
            self.peak_depth = max(self.peak_depth, self.queue.qsize())
        except Full:
            self.unscheduled += 1
            if self.unscheduled == 1:
                self.buffers.request_stop(
                    "asr_queue_saturated",
                    "The transcription queue could not accept another segment. Captured audio is retained.",
                )

    def finish(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except Full:
                continue
        self.thread.join()

    def _run(self):
        try:
            with self.pcm.open("rb") as source, self.pcm.with_name(
                "provisional-transcript.jsonl"
            ).open("w", encoding="utf-8", buffering=1) as provisional:
                self._transcribe(source, provisional)
        except Exception as error:
            self.failed = f"{type(error).__name__}: {error}"
            self.buffers.request_stop("asr_worker_error", self.failed)
            log.exception("Microphone ASR worker failed")

    def _transcribe(self, source, provisional):
        model = None
        load_error = None
        while (item := self.queue.get()) is not None:
            segment, boundary, queued = item
            dequeued = time.monotonic()
            if model is None and load_error is None:
                try:
                    verify_local_bundle(self.settings)
                    self.verification_seconds = time.monotonic() - dequeued
                    start = time.monotonic()
                    model = load_local_model(self.settings.hojo_dir, self.settings.backend)
                    self.load_seconds = time.monotonic() - start
                    self.backend, self.dtype = _model_placement(model)
                except Exception as error:
                    load_error = error
                    log.exception("Microphone Hojo model loading failed")
            asr_seconds = None
            if load_error is None:
                start = time.monotonic()
                try:
                    record = _infer_segment(
                        model, source, segment, self.pcm.parent, self.settings.max_chunk_seconds
                    )
                except Exception as error:
                    record = _error_segment(segment, error)
                    log.exception("Microphone Hojo inference failed for %s", segment.segment_id)
                asr_seconds = time.monotonic() - start
            else:
                record = _error_segment(segment, load_error)
            finished = time.monotonic()
            origin = self.buffers.origin_s
            timing = {
                "segment_id": segment.segment_id,
                "boundary": boundary,
                "endpointing_delay_s": None if origin is None else queued - (origin + segment.core_end / SAMPLE_RATE),
                "queue_delay_s": dequeued - queued,
                "asr_seconds": asr_seconds,
                "asr_finished_monotonic_s": finished,
            }
            provisional.write(json.dumps({"state": "provisional", "segment": record.model_dump(),
                                          "timing": timing}, ensure_ascii=False) + "\n")
            self.records[segment.ordinal] = record
            self.timings.append(timing)


def _session_record(path: Path, info: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(info, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _read_device(stream, buffers, consume):
    started = False
    try:
        with stream:
            started = True
            print("Listening for microphone input.", file=sys.stderr, flush=True)
            while buffers.reason is None and stream.active and not buffers.finished.is_set():
                try:
                    block = buffers.queue.get(timeout=0.05)
                except Empty:
                    continue
                consume(block)
        if buffers.reason is None:
            buffers.request_stop("device_stopped", "The input stream stopped unexpectedly. Check the microphone connection.")
    except KeyboardInterrupt:
        buffers.request_stop("interrupted")
    except buffers.sd.PortAudioError as error:
        buffers.request_stop(
            "device_error",
            f"Microphone stream failed: {error}. Check the device and macOS microphone permission.",
        )
    finally:
        if started:
            print("Microphone capture finished.", file=sys.stderr, flush=True)


def _record_pcm(stream, buffers, clips, pcm, rate):
    resampler = StreamingResampler(rate)
    sample_count = 0
    last_sync = time.monotonic()
    vad_available = True
    queue_delay_sum = queue_delay_max = 0.0
    dequeued_blocks = 0
    with pcm.open("ab", buffering=0) as output:
        def detect(data):
            nonlocal vad_available
            if vad_available:
                try:
                    clips.feed(data, sample_count)
                except (OSError, RuntimeError, ValueError) as error:
                    vad_available = False
                    buffers.request_stop("vad_error", f"Silero endpointing failed: {error}. Captured audio is retained.")
                    log.exception("Microphone Silero failed")

        def write_samples(samples):
            nonlocal sample_count, last_sync
            if not len(samples):
                return
            if sample_count + len(samples) > buffers.normalized_budget:
                raise OSError("Normalized microphone spool exceeded the configured sample budget")
            data = samples.astype("<f4", copy=False).tobytes()
            written = output.write(data)
            if written != len(data):
                valid = written // SAMPLE_BYTES * SAMPLE_BYTES
                output.truncate(sample_count * SAMPLE_BYTES + valid)
                sample_count += valid // SAMPLE_BYTES
                detect(data[:valid])
                raise OSError("Could not write the complete microphone block to the recoverable spool")
            sample_count += len(samples)
            detect(data)
            if time.monotonic() - last_sync >= 1:
                os.fsync(output.fileno())
                last_sync = time.monotonic()

        def consume(item):
            nonlocal queue_delay_sum, queue_delay_max, dequeued_blocks
            block, queued = item
            delay = time.monotonic() - queued
            dequeued_blocks += 1
            queue_delay_sum += delay
            queue_delay_max = max(queue_delay_max, delay)
            write_samples(resampler.feed(block.mean(axis=1, dtype=np.float32)))

        try:
            _read_device(stream, buffers, consume)
            capture_end = time.monotonic()
            for block in buffers.pending_blocks():
                consume(block)
            write_samples(resampler.finish())
        except (OSError, RuntimeError, ValueError) as error:
            capture_end = time.monotonic()
            buffers.request_stop("capture_processing_error", f"Microphone PCM or VAD failed: {error}")
            log.exception("Could not process microphone audio")
        try:
            if vad_available:
                clips.feed(b"", sample_count, final=True)
            os.fsync(output.fileno())
        except (OSError, RuntimeError, ValueError) as error:
            buffers.request_stop("capture_finalize_error", f"Could not finalize microphone audio: {error}")
    return sample_count, capture_end, {
        "processed_native_frames": resampler.input_end,
        "known_lost_normalized_samples": max(0, resampler.output_end - sample_count),
        "capture_queue_mean_delay_s": queue_delay_sum / dequeued_blocks if dequeued_blocks else None,
        "capture_queue_max_delay_s": queue_delay_max if dequeued_blocks else None,
    }


def capture_microphone(settings: Settings, spool: Path) -> AudioResult:
    """Capture local PCM, queue bounded utterances, and flush the final source."""
    import sounddevice as sd

    name, native_rate, channels = input_device(sd, settings.mic_device)
    buffers = InputBuffer(sd, native_rate, settings)
    session_id = f"mic_{uuid4().hex}"
    directory = spool.resolve() / session_id
    directory.mkdir(parents=True, mode=0o700)
    pcm = directory / "capture.f32le"
    pcm.touch()
    metadata = directory / "session.json"
    session_info = {
        "session_id": session_id, "state": "capturing", "device": name,
        "native_rate": native_rate, "native_channels": channels,
        "pcm_format": "float32le-mono", "pcm_file": pcm.name, "sample_rate": SAMPLE_RATE,
    }
    _session_record(metadata, session_info)
    asr = _QueuedASR(settings, pcm, buffers)
    clips = _LiveSegments(session_id, settings, asr.submit)
    try:
        stream = sd.InputStream(
            device=settings.mic_device, samplerate=native_rate, channels=channels, dtype="float32",
            blocksize=buffers.block_frames, callback=buffers.callback, finished_callback=buffers.finished.set,
        )
    except sd.PortAudioError as error:
        _session_record(metadata, {**session_info, "state": "device_error", "error": str(error)})
        raise MicrophoneError(
            f"Cannot open microphone {name!r}: {error}. Check microphone access in macOS Privacy & Security."
        ) from error

    previous_interrupt = None
    if current_thread() is main_thread():
        previous_interrupt = signal.signal(signal.SIGINT, lambda *_: buffers.request_stop("interrupted"))
    try:
        asr.thread.start()
        try:
            samples, capture_end, capture_metrics = _record_pcm(stream, buffers, clips, pcm, native_rate)
            _session_record(metadata, {
                **session_info, "state": "recorded", "stop_reason": buffers.reason,
                "sample_count": samples, "problems": buffers.problems,
                "segments": [s.model_dump() for s in clips.segments],
            })
        finally:
            asr.finish()
    finally:
        if previous_interrupt is not None:
            signal.signal(signal.SIGINT, previous_interrupt)

    if buffers.accepted == 0 and buffers.problems:
        raise MicrophoneError(f"{buffers.problems[0]} Capture record: {metadata}")
    with pcm.open("rb") as audio:
        digest = file_digest(audio, "sha256").hexdigest()
    source = AudioSource(
        source_id=session_id, kind="mic", digest=digest, name=name, native_rate=native_rate,
        native_channels=channels, sample_count=samples,
        channel_mix="mono" if channels == 1 else f"mean_of_{channels}_channels",
    )
    transcripts = [
        asr.records.get(segment.ordinal) or _error_segment(
            segment, MicrophoneError(asr.failed or "The utterance was retained in the PCM spool without transcription")
        )
        for segment in clips.segments
    ]
    _flag_reemissions(transcripts)
    errors = list(buffers.problems)
    errors.extend(f"{s.segment_id}: {error}" for s in transcripts for error in s.errors)
    if buffers.problems or any(s.status == "error" for s in transcripts):
        status = "error"
    elif not transcripts:
        status = "no_speech"
    elif errors:
        status = "needs_review"
    else:
        status = "ok"
    speech_end_sample = clips.segments[-1].core_end if clips.segments else None
    origin = buffers.origin_s
    speech_end_time = (
        origin + speech_end_sample / SAMPLE_RATE
        if origin is not None and speech_end_sample is not None else None
    )
    asr_seconds = sum(item["asr_seconds"] or 0 for item in asr.timings)
    metrics = {
        "capture_origin_monotonic_s": origin,
        "capture_origin_method": buffers.origin_method,
        "capture_end_monotonic_s": capture_end,
        "last_speech_end_sample": speech_end_sample,
        "last_speech_end_monotonic_s": speech_end_time,
        "speech_timeline_rate_hz": SAMPLE_RATE,
        "vad_min_silence_ms": VAD_MIN_SILENCE_MS,
        "silero": silero_provenance(),
        "stop_reason": buffers.reason,
        "capture_incomplete": bool(buffers.problems),
        "received_native_frames": buffers.received,
        "accepted_native_frames": buffers.accepted,
        "known_lost_native_frames": buffers.dropped + buffers.accepted - capture_metrics["processed_native_frames"],
        "ignored_native_frames_after_stop": buffers.ignored_after_stop,
        "device_overflow_events": buffers.overflows,
        "device_lost_frames_unknown": bool(buffers.overflows),
        "capture_block_frames": buffers.block_frames,
        "capture_queue_capacity_blocks": buffers.queue.maxsize,
        "capture_queue_peak_blocks_including_reserve": buffers.peak_blocks,
        "asr_queue_capacity_segments": asr.queue.maxsize,
        "asr_queue_peak_segments": asr.peak_depth,
        "untranscribed_queue_segments": asr.unscheduled,
        "normalized_spool_bytes": samples * SAMPLE_BYTES,
        "normalized_spooled_samples": samples,
        "model_load_seconds": asr.load_seconds,
        "model_asset_verification_seconds": asr.verification_seconds,
        "asr_seconds": asr_seconds,
        "asr_realtime_factor": asr_seconds / (samples / SAMPLE_RATE) if samples else None,
        "segments_timing": asr.timings,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "capture_session_dir": str(directory),
        **capture_metrics,
    }
    _session_record(metadata, {
        **session_info, "state": "finalized", "status": status, "source": source.model_dump(),
        "segments": [s.model_dump() for s in clips.segments], "metrics": metrics, "errors": errors,
    })
    return AudioResult(source=source, segments=transcripts, status=status, errors=errors,
                       backend=asr.backend, dtype=asr.dtype, metrics=metrics)
