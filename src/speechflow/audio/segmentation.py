"""Source-relative segmentation using the packaged Silero model."""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import BinaryIO

from ..contracts import AudioSegment, AudioSource, stable_id
from .decoding import SAMPLE_BYTES
from .hojo import SAMPLE_RATE

VAD_FRAME_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 350
_SILERO_IMPORT_LOCK = Lock()


@lru_cache(maxsize=1)
def silero_provenance() -> dict[str, str]:
    from importlib.metadata import version

    from ..assets import MANIFEST, silero_hash

    return {
        "version": version("silero-vad"),
        "artifact": MANIFEST["vad"]["path"],
        "sha256": silero_hash(),
    }


def _silero_detector():
    import torch

    # The pinned Silero package changes Torch's process-wide thread count at import.
    with _SILERO_IMPORT_LOCK:
        threads = torch.get_num_threads()
        try:
            from silero_vad import VADIterator, load_silero_vad
        finally:
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)

    return VADIterator(
        load_silero_vad(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )


class SileroStream:
    """One VAD state and a partial frame on the normalized sample clock."""

    def __init__(self):
        self.detector = _silero_detector()
        self.active_start: int | None = None
        self.processed = 0
        self.pending = b""

    def _frame(self, pcm: bytes) -> Iterator[tuple[int, int]]:
        import numpy as np
        import torch

        samples = len(pcm) // SAMPLE_BYTES
        self.processed += samples
        frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        frame[:samples] = np.frombuffer(pcm, dtype="<f4")
        event = self.detector(torch.from_numpy(frame))
        if event and "start" in event:
            self.active_start = max(0, min(self.processed, event["start"]))
        if event and "end" in event and self.active_start is not None:
            start, self.active_start = self.active_start, None
            end = min(self.processed, event["end"])
            if end > start:
                yield start, end

    def feed(self, pcm: bytes) -> Iterator[tuple[int, int]]:
        if len(pcm) % SAMPLE_BYTES:
            raise ValueError("Partial float32 sample in PCM")
        frame_bytes = VAD_FRAME_SAMPLES * SAMPLE_BYTES
        data = self.pending + pcm
        complete = len(data) // frame_bytes * frame_bytes
        self.pending = data[complete:]
        for offset in range(0, complete, frame_bytes):
            yield from self._frame(data[offset:offset + frame_bytes])

    def finish(self) -> Iterator[tuple[int, int]]:
        if self.pending:
            pending, self.pending = self.pending, b""
            yield from self._frame(pending)
        if self.active_start is not None and self.active_start < self.processed:
            yield self.active_start, self.processed
        self.active_start = None


def speech_ranges(pcm: BinaryIO, sample_count: int) -> Iterator[tuple[int, int]]:
    """Process exact 512-sample frames and flush speech active at EOF."""
    if sample_count == 0:
        return
    stream = SileroStream()
    while chunk := pcm.read(VAD_FRAME_SAMPLES * SAMPLE_BYTES):
        yield from stream.feed(chunk)
    yield from stream.finish()
    if stream.processed != sample_count:
        raise OSError(f"PCM spool contains {stream.processed} samples, expected {sample_count}")


def chunk_limits(max_seconds: float, padding_ms: int) -> tuple[int, int]:
    max_samples = int(max_seconds * SAMPLE_RATE)
    if not 1 <= max_samples <= 25 * SAMPLE_RATE:
        raise ValueError("max_chunk_seconds must allow at least one sample and at most 25 seconds")
    requested_pad = padding_ms * SAMPLE_RATE // 1000
    # Short chunk limits still leave room for speech on every iteration.
    pad_samples = min(requested_pad, (max_samples - 1) // 3)
    return max_samples - 2 * pad_samples, pad_samples


def make_segment(source_id, ordinal, core_start, core_end, sample_count, pad_samples):
    input_start = max(0, core_start - pad_samples)
    input_end = min(sample_count, core_end + pad_samples)
    return AudioSegment(
        segment_id=stable_id("segment", source_id, core_start, core_end, input_start, input_end),
        source_id=source_id, ordinal=ordinal, input_start=input_start, core_start=core_start,
        core_end=core_end, input_end=input_end,
    )


def segment_pcm(
    pcm: Path, source: AudioSource, max_seconds: float, padding_ms: int
) -> list[AudioSegment]:
    """Divide speech into disjoint cores, each with bounded source context."""
    core_limit, pad_samples = chunk_limits(max_seconds, padding_ms)
    segments = []
    with pcm.open("rb") as waveform:
        for speech_start, speech_end in speech_ranges(waveform, source.sample_count):
            for core_start in range(speech_start, speech_end, core_limit):
                core_end = min(core_start + core_limit, speech_end)
                segments.append(
                    make_segment(
                        source.source_id, len(segments), core_start, core_end,
                        source.sample_count, pad_samples,
                    )
                )
    return segments
