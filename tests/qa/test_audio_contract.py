"""Independently exercise input diagnostics and padded window arithmetic."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from speechflow.audio import segmentation
from speechflow.audio.decoding import AudioDecodeError, decode_file
from speechflow.audio.hojo import HojoResultError, infer_wav, validate_frame_count
from speechflow.config import Settings
from speechflow.contracts import AudioSource


def test_unsupported_suffix_errors_before_decoder_or_model(tmp_path):
    path = tmp_path / "untrusted.ogg"
    path.write_bytes(b"not an audio file")
    spool = tmp_path / "pcm"
    with pytest.raises(AudioDecodeError, match="WAV, FLAC, MP3, or M4A"):
        decode_file(path, spool, 64000)
    assert not spool.exists()


def test_installed_cli_rejects_empty_audio_before_model_access(tmp_path):
    path = tmp_path / "empty.wav"
    path.touch()
    cli = Path(sys.executable).parent / "speechflow"
    command = [str(cli)] if cli.exists() else [sys.executable, "-m", "speechflow.cli"]
    result = subprocess.run([*command, "transcribe", str(path), "--offline"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 4
    assert "empty" in result.stderr


@pytest.mark.parametrize("padding", [0, 200, 2000])
def test_mock_speech_range_produces_gapless_bounded_cores_and_flushes_tail(tmp_path, monkeypatch, padding):
    pcm = tmp_path / "dummy.f32"
    pcm.touch()  # Only range planning is under test. Input to Silero is mocked here.
    end = 72 * 16000 + 423
    source = AudioSource(source_id="source", kind="file", digest="fixture",
                         name="fixture.wav", native_rate=48000, native_channels=2,
                         sample_count=end, channel_mix="test")
    monkeypatch.setattr(segmentation, "speech_ranges", lambda pipe, count: [(32, count)])
    bounded = segmentation.segment_pcm(pcm, source, 25, padding)
    assert len(bounded) >= 3
    assert bounded[0].core_start == 32
    assert bounded[-1].core_end == bounded[-1].input_end == end
    assert [s.ordinal for s in bounded] == list(range(len(bounded)))
    assert len({s.segment_id for s in bounded}) == len(bounded)
    for left, right in zip(bounded, bounded[1:]):
        assert left.core_end == right.core_start
        assert (left.input_end > right.input_start) == bool(padding)
    assert all(0 < s.input_end - s.input_start <= 400000 for s in bounded)
    assert bounded == segmentation.segment_pcm(pcm, source, 25, padding)


def test_padded_hojo_input_limit_includes_each_sample():
    validate_frame_count(400000, 25)
    for frames in (0, 400001, 480001):
        with pytest.raises(ValueError):
            validate_frame_count(frames, 25)
    assert Settings().max_chunk_seconds == 25


@pytest.mark.parametrize("records, error", [
    ([], "0 records"),
    ([{"key": "other", "text": "Farine."}], "invalid record"),
    ([{"key": "cut", "text": None}], "non-text result"),
    ([{"key": "cut", "text": "Farine."}] * 2, "2 records"),
])
def test_hojo_adapter_rejects_missing_duplicate_wrong_or_invalid_model_replies(
    monkeypatch, records, error
):
    soundfile = SimpleNamespace(info=lambda file: SimpleNamespace(
        format="WAV", channels=1, samplerate=16000, frames=16000))
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    model = SimpleNamespace(device=SimpleNamespace(type="cpu"), run_infer=lambda *a, **kw: records)
    with pytest.raises(HojoResultError, match=error):
        infer_wav(model, Path("cut.wav"))


def test_capture_queue_configuration_requires_a_positive_finite_bound():
    from pydantic import ValidationError

    assert Settings(mic_queue_seconds=120).mic_queue_seconds == 120
    for limit in (0, -0.01, 120.01, float("inf"), float("nan")):
        with pytest.raises(ValidationError, match="mic_queue_seconds"):
            Settings(mic_queue_seconds=limit)
