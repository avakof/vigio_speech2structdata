"""Microphone regressions with explicitly injected frames and no physical input."""

from hashlib import file_digest
import json
from math import gcd
from pathlib import Path
import signal
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest
from scipy.signal import resample_poly

from speechflow.audio import microphone
from speechflow.audio.capture_device import InputBuffer, MicrophoneError
from speechflow.audio.resampling import StreamingResampler
from speechflow.config import Settings
from speechflow.contracts import TranscriptSegment


class PortAudioError(Exception):
    pass


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


TIMESTAMP = SimpleNamespace(inputBufferAdcTime=3.0, currentTime=3.02)
OK_FLAGS = SimpleNamespace(input_overflow=False)


class ActiveFromFirstFrame:
    active_start = 0
    processed = 0

    def feed(self, payload):
        self.processed += len(payload) // 4
        return []

    def finish(self):
        return [(0, self.processed)]


def install_input(monkeypatch, frames, *, rate=48000, pause_at=None, proceed=None, deny=False):
    """Inject the same buffer and lifecycle interface as a PortAudio input stream."""
    streams = []

    class InjectedStream:
        def __init__(self, *, callback, finished_callback, blocksize, channels, **kwargs):
            if deny:
                raise PortAudioError("permission denied")
            assert channels == frames.shape[1] and kwargs["samplerate"] == rate
            self.callback = callback
            self.finished_callback = finished_callback
            self.blocksize = blocksize
            self.stopping = Event()
            self.active = False
            self.worker = Thread(target=self._inject)
            streams.append(self)

        def __enter__(self):
            self.active = True
            self.worker.start()
            return self

        def __exit__(self, *exc):
            self.stopping.set()
            if proceed is not None:
                proceed.set()
            self.worker.join(timeout=5)
            assert not self.worker.is_alive()

        def _inject(self):
            paused = False
            try:
                for start in range(0, len(frames), self.blocksize):
                    if self.stopping.is_set():
                        break
                    if pause_at is not None and start >= pause_at and not paused:
                        paused = True
                        proceed.wait(timeout=5)
                    block = frames[start:start + self.blocksize]
                    self.callback(block, len(block), TIMESTAMP, OK_FLAGS)
                    time.sleep(0.001)
            except (CallbackStop, CallbackAbort):
                pass
            finally:
                self.active = False
                self.finished_callback()

    sd = SimpleNamespace(
        PortAudioError=PortAudioError, CallbackStop=CallbackStop, CallbackAbort=CallbackAbort,
        query_devices=lambda *_args, **_kwargs: {
            "name": "Injected audio fixture", "default_samplerate": rate, "max_input_channels": frames.shape[1],
        },
        check_input_settings=lambda **_kwargs: None,
        InputStream=InjectedStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd, streams


@pytest.mark.parametrize("rate", [16000, 22050, 32000, 44100, 48000])
def test_streaming_resampler_matches_full_signal(rate):
    audio = np.random.default_rng(777).standard_normal(53191).astype(np.float32)
    resampler = StreamingResampler(rate)
    pieces = [resampler.feed(audio[i:i + 977]) for i in range(0, len(audio), 977)]
    actual = np.concatenate([*pieces, resampler.finish()])
    common = gcd(rate, 16000)
    expected = resample_poly(audio, 16000 // common, rate // common)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    assert not len(resampler.finish())


def test_capture_saturation_keeps_last_block(monkeypatch):
    sd, _ = install_input(monkeypatch, np.empty((0, 1), dtype=np.float32))
    buffers = InputBuffer(sd, 48000, Settings(mic_queue_seconds=0.02, mic_seconds=2))
    shape = (buffers.block_frames, 1)
    first = np.full(shape, 1, dtype=np.float32)
    second = np.full(shape, 2, dtype=np.float32)
    buffers.callback(first, len(first), TIMESTAMP, OK_FLAGS)
    with pytest.raises(CallbackStop):
        buffers.callback(second, len(second), TIMESTAMP, OK_FLAGS)
    drained = [block for block, _ in buffers.pending_blocks()]
    assert [block[0, 0] for block in drained] == [1, 2]
    assert buffers.reason == "capture_queue_saturated"
    assert buffers.received == buffers.accepted == 2 * len(first)
    assert buffers.dropped == 0 and buffers.peak_blocks == 2


def test_input_overflow_is_reported_without_guessing_loss(monkeypatch):
    sd, _ = install_input(monkeypatch, np.empty((0, 1), dtype=np.float32))
    buffers = InputBuffer(sd, 48000, Settings(mic_seconds=2))
    block = np.ones((buffers.block_frames, 1), dtype=np.float32)
    with pytest.raises(CallbackStop):
        buffers.callback(block, len(block), TIMESTAMP, SimpleNamespace(input_overflow=True))
    assert buffers.reason == "device_overflow" and buffers.overflows == 1
    assert buffers.dropped == 0 and len(list(buffers.pending_blocks())) == 1


def test_denied_permission_has_device_diagnostic(monkeypatch, tmp_path, capsys):
    install_input(monkeypatch, np.zeros((960, 1), dtype=np.float32), deny=True)
    monkeypatch.setattr(microphone, "SileroStream", lambda: None)
    with pytest.raises(MicrophoneError, match="permission denied.*Privacy"):
        microphone.capture_microphone(Settings(mic_seconds=0.2), tmp_path)
    meta = json.loads(next(tmp_path.glob("mic_*/session.json")).read_text())
    assert meta["state"] == "device_error" and capsys.readouterr().err == ""


@pytest.mark.parametrize("budget,seconds,frames,expected_status,expected_reason,expected_samples", [
    (1024 * 1024, 0.2, 9600, "no_speech", "duration", 3200),
    (320, 1, 960, "error", "spool_limit", 80),
    (1024 * 1024, 1, 2880, "error", "device_stopped", 960),
])
def test_real_silero_on_injected_silence_and_recovery(
    monkeypatch, tmp_path, capsys, budget, seconds, frames, expected_status, expected_reason, expected_samples
):
    install_input(monkeypatch, np.zeros((frames, 1), dtype=np.float32))
    monkeypatch.setattr(microphone, "load_local_model", lambda *_: pytest.fail("Silence loaded Hojo"))
    result = microphone.capture_microphone(Settings(mic_seconds=seconds, max_spool_bytes=budget), tmp_path)
    messages = capsys.readouterr()
    assert not messages.out and messages.err.splitlines() == [
        "Listening for microphone input.", "Microphone capture finished.",
    ]
    assert result.status == expected_status and result.metrics["stop_reason"] == expected_reason
    assert result.source.sample_count == expected_samples
    assert result.metrics["last_speech_end_monotonic_s"] is None
    assert result.metrics["known_lost_native_frames"] == 0
    audio = next(tmp_path.glob("mic_*/capture.f32le"))
    with audio.open("rb") as source:
        assert file_digest(source, "sha256").hexdigest() == result.source.digest
    assert audio.stat().st_size == 4 * expected_samples


def test_continuous_injected_voice_has_forced_chunks_and_exact_eof(monkeypatch):
    monkeypatch.setattr(microphone, "SileroStream", ActiveFromFirstFrame)
    emitted = []
    live = microphone._LiveSegments(
        "mic_fixture", Settings(max_chunk_seconds=1, padding_ms=75),
        lambda segment, reason: emitted.append((segment, reason)),
    )
    duration = 55 * 16000 + 183
    for start in range(0, duration, 691):
        count = min(691, duration - start)
        live.feed(bytes(count * 4), start + count)
    assert len(emitted) > 2 and emitted[0][1] == "duration_limit"
    live.feed(b"", duration, final=True)
    segments = [segment for segment, reason in emitted]
    assert segments[-1].core_end == segments[-1].input_end == duration
    assert all(s.input_end - s.input_start <= 16000 for s in segments)
    assert all(a.core_end == b.core_start and a.input_end > b.input_start
               for a, b in zip(segments, segments[1:]))


def test_synthetic_speech_transcribes_while_still_capturing(monkeypatch, tmp_path):
    fixture = Path.home() / ".cache/speechflow/hojo-jacques.wav"
    if not fixture.is_file():
        pytest.skip("Create the labeled Jacques WAV from the Hojo feasibility instructions")
    with wave.open(str(fixture), "rb") as wav:
        mono = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    expanded = np.pad(np.repeat(mono, 3), (0, 72000))
    frames = np.column_stack((np.zeros_like(expanded), expanded * 2))
    inferred = Event()
    _, streams = install_input(monkeypatch, frames, pause_at=48000 * 9 // 2, proceed=inferred)
    loads = []
    monkeypatch.setattr(microphone, "verify_local_bundle", lambda *_: None)
    monkeypatch.setattr(microphone, "load_local_model", lambda *_: loads.append(object()) or loads[-1])
    monkeypatch.setattr(microphone, "_model_placement", lambda *_: ("cpu", "injected"))

    def stub_infer(model, pcm, segment, work, max_seconds):
        assert streams[0].active and model is loads[0]
        pcm.seek(segment.input_start * 4)
        samples = np.frombuffer(pcm.read((segment.input_end - segment.input_start) * 4), dtype="<f4")
        assert abs(samples).max() > 0.01
        inferred.set()
        raw = {"key": segment.segment_id, "text": "il reste deux virgule cinq kilos de tomates"}
        return TranscriptSegment(**segment.model_dump(), raw=raw, normalized_text=raw["text"])

    monkeypatch.setattr(microphone, "_infer_segment", stub_infer)
    result = microphone.capture_microphone(
        Settings(mic_seconds=len(frames) / 48000), tmp_path,
    )
    assert result.status == "ok", result.errors
    assert len(loads) == len(result.segments) == 1
    assert result.source.native_rate == 48000 and result.source.native_channels == 2
    assert result.source.sample_count == (len(frames) + 2) // 3
    assert result.metrics["known_lost_native_frames"] == 0
    assert result.metrics["vad_min_silence_ms"] == 350
    assert result.metrics["last_speech_end_monotonic_s"] == (
        result.metrics["capture_origin_monotonic_s"] + result.segments[-1].core_end / 16000
    )
    provisional = json.loads(next(tmp_path.glob("mic_*/provisional-transcript.jsonl")).read_text())
    assert provisional["state"] == "provisional" and provisional["segment"]["raw"] == result.segments[0].raw


def test_asr_queue_saturation_preserves_captured_pcm(monkeypatch, tmp_path):
    unblocked = Event()
    install_input(monkeypatch, np.ones((32000, 1), dtype=np.float32), rate=16000, proceed=unblocked)
    monkeypatch.setattr(microphone, "SileroStream", ActiveFromFirstFrame)
    monkeypatch.setattr(microphone, "ASR_QUEUE_BLOCKS", 1)
    monkeypatch.setattr(microphone, "verify_local_bundle", lambda *_: None)
    loads = []
    monkeypatch.setattr(microphone, "load_local_model", lambda *_: loads.append(object()) or loads[-1])
    monkeypatch.setattr(microphone, "_model_placement", lambda *_: ("cpu", "injected"))

    def delayed_asr(model, pcm, segment, work, max_seconds):
        assert model is loads[0] and unblocked.wait(timeout=5)
        return TranscriptSegment(**segment.model_dump(), raw={"key": segment.segment_id, "text": "test"},
                                 normalized_text="test")

    monkeypatch.setattr(microphone, "_infer_segment", delayed_asr)
    result = microphone.capture_microphone(
        Settings(mic_seconds=2, max_chunk_seconds=0.05, padding_ms=0), tmp_path
    )
    assert result.status == "error" and result.metrics["stop_reason"] == "asr_queue_saturated"
    assert result.metrics["asr_queue_peak_segments"] == 1
    assert result.metrics["untranscribed_queue_segments"] > 0
    assert len(loads) == 1 and result.metrics["known_lost_native_frames"] == 0
    assert result.source.sample_count == result.metrics["accepted_native_frames"]
    assert next(tmp_path.glob("mic_*/capture.f32le")).stat().st_size == result.source.sample_count * 4
    assert any(s.status == "error" for s in result.segments)


def test_interrupt_drains_pending_audio_and_flushes_eof(monkeypatch, tmp_path):
    baseline = signal.getsignal(signal.SIGINT)
    sd, _ = install_input(monkeypatch, np.empty((0, 1), dtype=np.float32), rate=16000)
    monkeypatch.setattr(microphone, "SileroStream", ActiveFromFirstFrame)
    monkeypatch.setattr(microphone, "verify_local_bundle", lambda *_: None)
    monkeypatch.setattr(microphone, "load_local_model", lambda *_: object())
    monkeypatch.setattr(microphone, "_model_placement", lambda *_: ("cpu", "injected"))
    monkeypatch.setattr(
        microphone, "_infer_segment",
        lambda model, pcm, segment, work, max_seconds: TranscriptSegment(
            **segment.model_dump(), raw={"key": segment.segment_id, "text": "test"}, normalized_text="test"
        ),
    )

    class SignalStream:
        active = True

        def __init__(self, *, callback, finished_callback, blocksize, **_):
            self.callback, self.finished_callback, self.blocksize = callback, finished_callback, blocksize

        def __enter__(self):
            frame = np.ones((self.blocksize, 1), dtype=np.float32)
            self.callback(frame, len(frame), TIMESTAMP, OK_FLAGS)
            self.callback(frame, len(frame), TIMESTAMP, OK_FLAGS)
            signal.raise_signal(signal.SIGINT)
            return self

        def __exit__(self, *_):
            self.active = False
            self.finished_callback()

    monkeypatch.setattr(sd, "InputStream", SignalStream)
    result = microphone.capture_microphone(Settings(), tmp_path)
    assert result.status == "ok" and result.metrics["stop_reason"] == "interrupted"
    assert result.source.sample_count == 640
    assert len(result.segments) == 1 and result.segments[0].core_end == 640
    assert result.metrics["known_lost_native_frames"] == 0
    assert signal.getsignal(signal.SIGINT) is baseline
