"""Source-relative segmentation using the packaged Silero model."""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from ..contracts import AudioSegment, AudioSource, stable_id
from .decoding import SAMPLE_BYTES
from .hojo import SAMPLE_RATE

VAD_FRAME_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 100


@lru_cache(maxsize=1)
def silero_provenance() -> dict[str, str]:
    from importlib.metadata import version

    from ..assets import MANIFEST, silero_hash

    return {
        "version": version("silero-vad"),
        "artifact": MANIFEST["vad"]["path"],
        "sha256": silero_hash(),
    }


def speech_ranges(pcm: BinaryIO, sample_count: int) -> Iterator[tuple[int, int]]:
    """Process exact 512-sample frames and flush speech active at EOF."""
    if sample_count == 0:
        return

    import numpy as np
    import torch
    from silero_vad import VADIterator, load_silero_vad

    detector = VADIterator(
        load_silero_vad(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )
    start = None
    consumed = 0
    while chunk := pcm.read(VAD_FRAME_SAMPLES * SAMPLE_BYTES):
        samples = len(chunk) // SAMPLE_BYTES
        consumed += samples
        frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
        frame[:samples] = np.frombuffer(chunk, dtype="<f4")
        if event := detector(torch.from_numpy(frame)):
            if "start" in event:
                start = max(0, min(sample_count, event["start"]))
            if "end" in event and start is not None:
                end = max(start, min(sample_count, event["end"]))
                if end > start:
                    yield start, end
                start = None
    if consumed != sample_count:
        raise OSError(f"PCM spool contains {consumed} samples, expected {sample_count}")
    if start is not None and start < sample_count:
        yield start, sample_count


def segment_pcm(
    pcm: Path, source: AudioSource, max_seconds: float, padding_ms: int
) -> list[AudioSegment]:
    """Divide speech into disjoint cores, each with bounded source context."""
    max_samples = int(max_seconds * SAMPLE_RATE)
    if not 1 <= max_samples <= 25 * SAMPLE_RATE:
        raise ValueError("max_chunk_seconds must allow at least one sample and at most 25 seconds")
    requested_pad = padding_ms * SAMPLE_RATE // 1000
    # Short chunk limits still leave room for speech on every iteration.
    pad_samples = min(requested_pad, (max_samples - 1) // 3)
    core_limit = max_samples - 2 * pad_samples

    segments = []
    with pcm.open("rb") as waveform:
        for speech_start, speech_end in speech_ranges(waveform, source.sample_count):
            for core_start in range(speech_start, speech_end, core_limit):
                core_end = min(core_start + core_limit, speech_end)
                input_start = max(0, core_start - pad_samples)
                input_end = min(source.sample_count, core_end + pad_samples)
                segments.append(
                    AudioSegment(
                        segment_id=stable_id(
                            "segment", source.source_id, core_start, core_end, input_start, input_end
                        ),
                        source_id=source.source_id,
                        ordinal=len(segments),
                        input_start=input_start,
                        core_start=core_start,
                        core_end=core_end,
                        input_end=input_end,
                    )
                )
    return segments
