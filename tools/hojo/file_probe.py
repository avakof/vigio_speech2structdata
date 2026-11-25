#!/usr/bin/env python3
"""Exercise local decoding and real Silero segmentation without loading Hojo."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from speechflow.audio import _flag_reemissions, transcribe_file
from speechflow.audio.decoding import AudioDecodeError, decode_file, ffmpeg_executable
from speechflow.audio.segmentation import segment_pcm, silero_provenance
from speechflow.config import Settings
from speechflow.contracts import AudioSource, TranscriptSegment


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def encode(source, destination, *options):
    command = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        *options, str(destination),
    ]
    subprocess.run(command, check=True)


def wav(path, frames, rate=16_000):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(frames)


def segment(path, pcm):
    started = time.perf_counter()
    source = decode_file(path, pcm, 32 * 1024 * 1024)
    segments = segment_pcm(pcm, source, 25, 200)
    for previous, current in zip(segments, segments[1:]):
        assert previous.core_end <= current.core_start
    for chunk in segments:
        assert 0 <= chunk.input_start <= chunk.core_start < chunk.core_end <= chunk.input_end
        assert chunk.input_end <= source.sample_count
        assert chunk.input_end - chunk.input_start <= 400_000
    emit(
        "segmentation", source=source.model_dump(), seconds=time.perf_counter() - started,
        segments=[item.model_dump() for item in segments],
    )
    return source, segments


def check_error(path, pcm, limit, expected):
    try:
        decode_file(path, pcm, limit)
    except AudioDecodeError as error:
        assert expected in str(error)
        emit("expected_decode_error", file=path.name, message=str(error))
    else:
        raise AssertionError(f"{path.name} unexpectedly decoded")


def check_continuous_boundaries(work):
    total = 65 * 16_000 + 83
    voice = (800, total)
    source = AudioSource(
        source_id="synthetic_boundary_probe", kind="file", digest="synthetic",
        name="synthetic_boundary_probe", native_rate=16000, native_channels=1,
        sample_count=total, channel_mix="mono",
    )
    pcm = work / "ranges-only.pcm"
    pcm.touch()
    with patch("speechflow.audio.segmentation.speech_ranges", return_value=iter([voice])):
        segments = segment_pcm(pcm, source, 25, 200)
    assert len(segments) == 3 and segments[0].core_start == voice[0]
    assert segments[-1].core_end == total and segments[-1].input_end == total
    for previous, current in zip(segments, segments[1:]):
        assert previous.core_end == current.core_start
        assert previous.input_end - current.input_start == 6400
    assert all(item.input_end - item.input_start <= 400_000 for item in segments)
    emit("mocked_vad_boundary_arithmetic", segments=[item.model_dump() for item in segments])
    replies = ["Il reste deux kilos.", "Deux kilos de lait.", "De lait. Trois tomates."]
    transcript = [
        TranscriptSegment(
            **segment.model_dump(), raw={"key": segment.segment_id, "text": reply},
            normalized_text=reply,
        ) for segment, reply in zip(segments, replies, strict=True)
    ]
    # The first two inputs overlap. The final pair uses adjacent, disjoint inputs.
    transcript[-1].input_start = transcript[-1].core_start
    transcript[1].input_end = transcript[1].core_end
    _flag_reemissions(transcript)
    assert [item.status for item in transcript] == ["needs_review", "needs_review", "ok"]
    assert [item.raw["text"] for item in transcript] == replies
    assert [item.normalized_text for item in transcript] == replies
    emit("mocked_asr_overlap_flags", segments=[item.model_dump() for item in transcript])
    single_word = [
        TranscriptSegment(
            **segment.model_dump(), raw={"key": segment.segment_id, "text": reply},
            normalized_text=reply,
        ) for segment, reply in zip(segments[:2], ["Il manque du lait.", "Lait de chèvre."], strict=True)
    ]
    _flag_reemissions(single_word)
    assert all(item.status == "ok" and item.raw["text"] == item.normalized_text for item in single_word)
    emit("mocked_single_word_overlap", segments=[item.model_dump() for item in single_word])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="*", help="Audio files to decode and segment")
    parser.add_argument("--boundaries-only", action="store_true")
    parser.add_argument("--segments-only", action="store_true")
    args = parser.parse_args()
    if args.boundaries_only:
        with TemporaryDirectory(prefix="speechflow-hojo-probe-") as temp:
            check_continuous_boundaries(Path(temp))
        return
    if args.segments_only:
        if not args.audio:
            parser.error("Supply at least one speech recording")
        emit("silero", **silero_provenance())
        with TemporaryDirectory(prefix="speechflow-hojo-probe-") as temp:
            for path in args.audio:
                _, segments = segment(path, Path(temp) / "probe.pcm")
                assert segments, f"No speech found in {path}"
        return
    if len(args.audio) != 1:
        parser.error("Supply one synthetic Jacques WAV, or choose --segments-only or --boundaries-only")
    audio = args.audio[0]
    with wave.open(str(audio), "rb") as speech:
        assert (speech.getframerate(), speech.getnchannels(), speech.getsampwidth()) == (16000, 1, 2)
        samples = speech.readframes(speech.getnframes())

    emit("silero", **silero_provenance())
    with TemporaryDirectory(prefix="speechflow-hojo-probe-") as temp:
        work = Path(temp)
        pcm = work / "probe.pcm"
        source, segments = segment(audio, pcm)
        assert (source.native_rate, source.native_channels) == (16000, 1) and segments

        for suffix in ("flac", "mp3", "m4a"):
            path = work / f"right-only.{suffix}"
            encode(audio, path, "-af", "pan=stereo|c0=0*c0|c1=c0", "-ar", "44100")
            source, segments = segment(path, pcm)
            assert (source.native_rate, source.native_channels) == (44100, 2) and segments

        silence = work / "silence.wav"
        wav(silence, bytes(16000 * 2))
        result = transcribe_file(silence, Settings(hojo_dir=work / "missing-hojo", offline=True))
        assert result.status == "no_speech" and not result.segments and result.backend == "not_loaded"
        emit("silence", status=result.status, metrics=result.metrics)

        corrupt = work / "corrupt.wav"
        corrupt.write_bytes(b"broken audio")
        check_error(corrupt, pcm, 1 << 20, "FFmpeg could not decode")
        check_error(audio, pcm, 1000, "max_spool_bytes")

        # Trim leading/trailing near-silence and replay the synthetic utterance.
        # This tests source clock and bounds, while boundary recognition requires Hojo.
        import numpy as np

        waveform = np.frombuffer(samples, "<i2")
        voiced = np.flatnonzero(np.abs(waveform.astype("int32")) > 1000)
        trimmed = waveform[voiced[0]:voiced[-1] + 1].tobytes()
        repetitions = 65 * 16000 * 2 // len(trimmed) + 1
        long = work / "long-synthetic.wav"
        wav(long, trimmed * repetitions)
        source, segments = segment(long, pcm)
        assert source.sample_count > 60 * 16000
        assert any(s.core_end > 35 * 16000 for s in segments)
        assert segments[-1].core_end > source.sample_count - 1600
        check_continuous_boundaries(work)
        emit("passed", case_count=9)


if __name__ == "__main__":
    main()
