"""Local file decoding, speech segmentation, and Hojo transcription."""

import logging
from pathlib import Path
import re
import resource
from tempfile import TemporaryDirectory
import time
import unicodedata

from ..config import Settings
from ..contracts import AudioResult, AudioSegment, AudioSource, TranscriptSegment
from .decoding import SAMPLE_BYTES, decode_file
from .hojo import SAMPLE_RATE, infer_wav, load_local_model, verify_local_bundle
from .segmentation import segment_pcm, silero_provenance

log = logging.getLogger(__name__)
WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")


def _error_segment(segment: AudioSegment, error: Exception) -> TranscriptSegment:
    return TranscriptSegment(
        **segment.model_dump(),
        raw={},
        normalized_text="",
        status="error",
        errors=[f"{type(error).__name__}: {error}"],
    )


def _infer_segment(model, source_pcm, segment: AudioSegment, work: Path, max_seconds: float):
    import numpy as np
    import soundfile

    length = segment.input_end - segment.input_start
    source_pcm.seek(segment.input_start * SAMPLE_BYTES)
    data = source_pcm.read(length * SAMPLE_BYTES)
    if len(data) != length * SAMPLE_BYTES:
        raise OSError(f"PCM spool ended inside segment {segment.segment_id}")
    wav = work / f"{segment.segment_id}.wav"
    try:
        soundfile.write(wav, np.frombuffer(data, dtype="<f4"), SAMPLE_RATE, subtype="FLOAT")
        raw = infer_wav(model, wav, max_seconds=max_seconds)
    finally:
        wav.unlink(missing_ok=True)

    text = " ".join(unicodedata.normalize("NFC", raw["text"]).split())
    return TranscriptSegment(
        **segment.model_dump(),
        raw=raw,
        normalized_text=text,
        status="ok" if text else "needs_review",
        errors=[] if text else ["Hojo returned empty text for a detected speech segment"],
    )


def _flag_reemissions(segments: list[TranscriptSegment]) -> None:
    for left, right in zip(segments, segments[1:]):
        shared = min(left.input_end, right.input_end) - max(left.input_start, right.input_start)
        if shared <= 0 or left.status == "error" or right.status == "error":
            continue
        suffix = WORD.findall(left.normalized_text.casefold().replace("’", "'"))
        prefix = WORD.findall(right.normalized_text.casefold().replace("’", "'"))
        for width in range(min(len(suffix), len(prefix)), 1, -1):
            if suffix[-width:] == prefix[:width]:
                phrase = " ".join(prefix[:width])
                for segment, neighbor in ((left, right), (right, left)):
                    segment.status = "needs_review"
                    segment.errors.append(
                        f"Possible repeated boundary text {phrase!r} in {shared} shared input "
                        f"samples with {neighbor.segment_id}. Both replies are preserved."
                    )
                break


def _transcribe_pcm(
    pcm: Path, source: AudioSource, settings: Settings, metrics: dict
) -> AudioResult:
    start = time.perf_counter()
    try:
        segments = segment_pcm(pcm, source, settings.max_chunk_seconds, settings.padding_ms)
        metrics["silero"] = silero_provenance()
    except Exception as error:
        log.exception("Silero failed for %s", source.source_id)
        return AudioResult(
            source=source, segments=[], status="error", backend="not_loaded", dtype="not_loaded",
            errors=[f"Silero segmentation failed: {type(error).__name__}: {error}"], metrics=metrics,
        )
    metrics["vad_seconds"] = time.perf_counter() - start
    metrics["asr_chunk_sample_counts"] = [s.input_end - s.input_start for s in segments]
    if not segments:
        return AudioResult(
            source=source, segments=[], status="no_speech", backend="not_loaded",
            dtype="not_loaded", metrics=metrics,
        )

    start = time.perf_counter()
    try:
        verify_local_bundle(settings)
        metrics["model_asset_verification_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        model = load_local_model(settings.hojo_dir, settings.backend)
    except Exception as error:
        log.exception("Hojo model loading failed")
        return AudioResult(
            source=source,
            segments=[_error_segment(segment, error) for segment in segments],
            status="error",
            backend="not_loaded",
            dtype="not_loaded",
            errors=[f"Hojo load failed on {settings.backend}: {type(error).__name__}: {error}"],
            metrics=metrics,
        )
    metrics["model_load_seconds"] = time.perf_counter() - start
    encoder = next(model.speech_encoder.parameters())
    decoder = model.decoder_model.get_input_embeddings().weight
    backend = model.device.type
    dtype = f"encoder={str(encoder.dtype).removeprefix('torch.')},decoder={str(decoder.dtype).removeprefix('torch.')}"

    transcripts = []
    start = time.perf_counter()
    with pcm.open("rb") as source_pcm:
        for segment in segments:
            try:
                transcript = _infer_segment(
                    model, source_pcm, segment, pcm.parent, settings.max_chunk_seconds
                )
            except Exception as error:
                log.exception("Hojo failed for %s", segment.segment_id)
                transcript = _error_segment(segment, error)
            transcripts.append(transcript)
    _flag_reemissions(transcripts)
    metrics["asr_seconds"] = time.perf_counter() - start
    metrics["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    duration = source.sample_count / SAMPLE_RATE
    metrics["asr_realtime_factor"] = metrics["asr_seconds"] / duration
    statuses = {segment.status for segment in transcripts}
    if "error" in statuses:
        status = "error"
    elif "needs_review" in statuses:
        status = "needs_review"
    else:
        status = "ok"
    return AudioResult(
        source=source, segments=transcripts, status=status, backend=backend, dtype=dtype,
        errors=[f"{s.segment_id}: {error}" for s in transcripts for error in s.errors],
        metrics=metrics,
    )


def transcribe_file(path: Path, settings: Settings) -> AudioResult:
    """Transcribe the first audio stream of a bounded file."""
    start = time.perf_counter()
    with TemporaryDirectory(prefix="speechflow-audio-") as temp:
        pcm = Path(temp) / "source.f32le"
        metrics: dict = {}
        source = decode_file(path, pcm, settings.max_spool_bytes, metrics=metrics)
        metrics.update({
            "decode_seconds": time.perf_counter() - start,
            "decoded_seconds": source.sample_count / SAMPLE_RATE,
            "normalized_spool_bytes": source.sample_count * SAMPLE_BYTES,
            "requested_backend": settings.backend,
        })
        result = _transcribe_pcm(pcm, source, settings, metrics)
    result.metrics["audio_total_seconds"] = time.perf_counter() - start
    return result


__all__ = ["transcribe_file"]
