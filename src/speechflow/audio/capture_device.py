"""PortAudio input and bounded callback handoff for microphone sessions."""

from math import ceil, isfinite
from queue import Empty, Full, Queue
from threading import Event
import time

from .decoding import SAMPLE_BYTES
from .hojo import SAMPLE_RATE


class MicrophoneError(RuntimeError):
    """An actionable capture failure."""


def input_device(sd, selected):
    try:
        device = sd.query_devices(selected, kind="input")
        rate = float(device["default_samplerate"])
        channels = int(device["max_input_channels"])
        if not isfinite(rate) or not rate.is_integer() or rate <= 0 or channels <= 0:
            raise ValueError("the device advertised an invalid input rate or channel count")
        sd.check_input_settings(device=selected, samplerate=int(rate), channels=channels, dtype="float32")
    except (ValueError, TypeError, OSError, sd.PortAudioError) as error:
        raise MicrophoneError(
            f"Cannot use microphone {selected!r}: {error}. "
            "Choose an available input with --device and check microphone access in macOS Privacy & Security."
        ) from error
    return device["name"], int(rate), channels


class InputBuffer:
    """A callback queue plus one reserved block for controlled saturation stop."""

    def __init__(self, sd, rate, settings):
        self.sd = sd
        self.rate = rate
        budget = (settings.max_spool_bytes // SAMPLE_BYTES) * rate // SAMPLE_RATE
        if budget < 1:
            raise MicrophoneError("max_spool_bytes is too small for one normalized microphone sample")
        requested = settings.mic_seconds * rate if settings.mic_seconds is not None else None
        if requested is not None and not isfinite(requested):
            raise MicrophoneError("Microphone duration must be finite")
        configured = ceil(requested) if requested is not None else None
        self.limit = min(budget, configured) if configured is not None else budget
        self.at_limit = "duration" if configured is not None and configured <= budget else "spool_limit"
        queue_frames = max(1, int(rate * settings.mic_queue_seconds))
        self.block_frames = min(max(1, rate // 50), queue_frames)
        self.queue = Queue(maxsize=max(1, queue_frames // self.block_frames))
        self.finished = Event()
        self.emergency = None
        self.reason = None
        self.problems: list[str] = []
        self.origin_s = None
        self.origin_method = None
        self.received = 0
        self.accepted = 0
        self.dropped = 0
        self.ignored_after_stop = 0
        self.peak_blocks = 0
        self.overflows = 0
        self.normalized_budget = settings.max_spool_bytes // SAMPLE_BYTES

    def request_stop(self, reason, problem=None):
        if self.reason is None:
            self.reason = reason
        if problem:
            self.problems.append(problem)

    def callback(self, indata, frames, timestamp, flags):
        """Copy at most one fixed-size block and enqueue without waiting or doing I/O."""
        self.received += frames
        if self.reason is not None:
            self.ignored_after_stop += frames
            raise self.sd.CallbackStop

        take = min(frames, self.block_frames, self.limit - self.accepted)
        now = time.monotonic()
        if self.origin_s is None:
            adc = timestamp.inputBufferAdcTime
            delay = timestamp.currentTime - adc
            if adc > 0 and isfinite(delay) and delay >= 0:
                self.origin_s = now - delay
                self.origin_method = "portaudio_adc_mapped_to_monotonic"
            else:
                self.origin_s = now - frames / self.rate
                self.origin_method = "callback_monotonic_minus_block_duration"

        try:
            block = indata[:take].copy(), now
            self.queue.put_nowait(block)
            self.peak_blocks = max(self.peak_blocks, self.queue.qsize())
        except Full:
            self.emergency = block
            self.peak_blocks = max(self.peak_blocks, self.queue.maxsize + 1)
            self.request_stop("capture_queue_saturated", "Capture queue saturated. The overflow block was retained.")
        except Exception as error:
            self.dropped += frames
            self.request_stop("callback_error", f"Could not retain microphone frames: {error}")
            raise self.sd.CallbackAbort from error

        self.accepted += take
        if flags.input_overflow:
            self.overflows += 1
            self.request_stop("device_overflow", "PortAudio reported input overflow. The lost frame count is unknown.")
        if frames > self.block_frames:
            self.dropped += frames - take
            self.request_stop("unexpected_block_size", "PortAudio supplied a block larger than the bounded input block.")
        else:
            self.ignored_after_stop += frames - take
        if self.accepted >= self.limit:
            detail = "The configured audio spool limit was reached." if self.at_limit == "spool_limit" else None
            self.request_stop(self.at_limit, detail)
        if self.reason is not None:
            raise self.sd.CallbackStop

    def pending_blocks(self):
        """Drain in capture order after the PortAudio stream has stopped."""
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                break
        if self.emergency is not None:
            yield self.emergency
            self.emergency = None
