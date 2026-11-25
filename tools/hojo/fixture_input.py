#!/usr/bin/env python3
"""Acceptance-only synthetic input on a paced PortAudio-shaped interface."""

import argparse
from hashlib import file_digest
import importlib.abc
import json
from math import gcd, isfinite
from pathlib import Path
import sys
from threading import Event, Thread
import time
from types import ModuleType, SimpleNamespace
import wave

import numpy as np
from scipy.signal import resample_poly

DEVICE = "SPEECHFLOW_QA_FILE_INPUT_ONLY"
RATE = 48_000


class PortAudioError(Exception):
    pass


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


def install(path: Path, *, leading_seconds: float = 1, trailing_seconds: float = 2) -> ModuleType:
    """Register the fixture module in the current ASR child's module cache."""
    existing = sys.modules.get("sounddevice")
    if existing is not None and not getattr(existing, "__speechflow_fixture_input__", False):
        raise RuntimeError("Fixture input must be installed before any native sounddevice import")
    if not all(isfinite(value) and value >= 0 for value in (leading_seconds, trailing_seconds)):
        raise ValueError("Fixture silence durations must be finite and nonnegative")

    path = Path(path).resolve(strict=True)
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2:
            raise ValueError("Fixture input requires signed 16-bit PCM WAV")
        rate, channels, sample_count = audio.getframerate(), audio.getnchannels(), audio.getnframes()
        pcm = audio.readframes(sample_count)
    native = np.frombuffer(pcm, dtype="<i2").reshape(sample_count, channels)
    mono = native.mean(axis=1, dtype=np.float32) / 32768
    factor = gcd(rate, RATE)
    resampled = resample_poly(mono, RATE // factor, rate // factor).astype(np.float32, copy=False)
    lead, trail = round(RATE * leading_seconds), round(RATE * trailing_seconds)
    prepared = np.pad(resampled, (lead, trail))
    evidence = {
        "label": "synthetic_fixture_input_no_hardware",
        "input_path": str(path), "input_sha256": digest, "input_rate_hz": rate,
        "input_channels": channels, "input_frames": sample_count,
        "callback_rate_hz": RATE, "callback_channels": 1, "resampled_speech_frames": len(resampled),
        "leading_silence_frames": lead, "trailing_silence_frames": trail, "prepared_frames": len(prepared),
        "emitted_callback_frames": 0, "extra_digital_silence_frames": 0,
        "timestamp_method": "synthetic ADC start and callback deadline on a paced host monotonic clock",
        "hardware_opened": False, "playback": False,
    }

    def query_devices(device=None, kind=None):
        if device != DEVICE or kind != "input":
            raise PortAudioError(f"Fixture input accepts only {DEVICE!r} as its input device")
        return {"name": DEVICE, "default_samplerate": RATE, "max_input_channels": 1}

    def check_input_settings(*, device, samplerate, channels, dtype):
        query_devices(device, "input")
        if (samplerate, channels, dtype) != (RATE, 1, "float32"):
            raise PortAudioError("Fixture input uses 48 kHz mono float32")

    class InputStream:
        def __init__(self, *, device, samplerate, channels, dtype, blocksize, callback, finished_callback):
            check_input_settings(device=device, samplerate=samplerate, channels=channels, dtype=dtype)
            if blocksize < 1:
                raise PortAudioError("Fixture input requires a positive callback block size")
            self.blocksize = blocksize
            self.callback = callback
            self.finished_callback = finished_callback
            self.stopping = Event()
            self.active = False
            self.worker = Thread(target=self._feed, name="speechflow-fixture-input")

        def __enter__(self):
            self.active = True
            self.worker.start()
            return self

        def __exit__(self, *_exc):
            self.stopping.set()
            self.worker.join()

        def _feed(self):
            origin = time.monotonic()
            emitted = 0
            evidence["fixture_origin_monotonic_s"] = origin
            try:
                while not self.stopping.is_set():
                    end = emitted + self.blocksize
                    deadline = origin + end / RATE
                    if self.stopping.wait(max(0, deadline - time.monotonic())):
                        break
                    block = np.zeros((self.blocksize, 1), dtype=np.float32)
                    available = prepared[emitted:end]
                    block[:len(available), 0] = available
                    stamp = SimpleNamespace(inputBufferAdcTime=origin + emitted / RATE, currentTime=deadline)
                    emitted = end
                    evidence["emitted_callback_frames"] = emitted
                    evidence["extra_digital_silence_frames"] = max(0, emitted - len(prepared))
                    self.callback(block, self.blocksize, stamp, SimpleNamespace(input_overflow=False))
            except (CallbackStop, CallbackAbort):
                pass
            finally:
                evidence["fixture_elapsed_monotonic_s"] = time.monotonic() - origin
                self.active = False
                self.finished_callback()

    substitute = ModuleType("sounddevice")
    substitute.__speechflow_fixture_input__ = True
    substitute.fixture_metadata = evidence
    substitute.PortAudioError = PortAudioError
    substitute.CallbackStop = CallbackStop
    substitute.CallbackAbort = CallbackAbort
    substitute.query_devices = query_devices
    substitute.check_input_settings = check_input_settings
    substitute.InputStream = InputStream
    sys.modules["sounddevice"] = substitute
    return substitute


def self_test(path):
    class RejectNativeSoundDevice(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"sounddevice", "_sounddevice"}:
                raise AssertionError(f"Attempted to resolve native module {fullname}")

    def reject_portaudio_library(event, args):
        if event == "ctypes.dlopen" and "portaudio" in str(args[0]).casefold():
            raise AssertionError("Attempted to load the native PortAudio library")

    sys.meta_path.insert(0, RejectNativeSoundDevice())
    sys.addaudithook(reject_portaudio_library)
    fixture = install(path)
    import sounddevice

    assert sounddevice is fixture and fixture.__speechflow_fixture_input__
    assert "_sounddevice" not in sys.modules
    assert sounddevice.query_devices(DEVICE, "input")["default_samplerate"] == RATE
    try:
        sounddevice.query_devices(None, "input")
    except PortAudioError:
        pass
    else:
        raise AssertionError("Fixture input accepted a real default device selector")

    finished = Event()
    received = []

    def callback(block, frames, stamp, flags):
        assert block.shape == (frames, 1) and not flags.input_overflow
        received.append((frames, stamp.inputBufferAdcTime, stamp.currentTime))
        if len(received) == 3:
            raise CallbackStop

    with sounddevice.InputStream(
        device=DEVICE, samplerate=RATE, channels=1, dtype="float32", blocksize=960,
        callback=callback, finished_callback=finished.set,
    ):
        assert finished.wait(timeout=2)
    assert len(received) == 3 and received[-1][2] - received[0][1] > 0.059
    assert fixture.fixture_metadata["emitted_callback_frames"] == 2880
    assert fixture.fixture_metadata["fixture_elapsed_monotonic_s"] > 0.059
    print(json.dumps({"self_test": "passed", "native_import_and_dlopen_guarded": True,
                      **fixture.fixture_metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()
    self_test(args.wav)
