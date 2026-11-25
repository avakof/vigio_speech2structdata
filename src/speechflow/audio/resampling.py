"""Incremental polyphase conversion to the normalized sample clock."""

from math import gcd

import numpy as np
from scipy.signal import firwin, resample_poly

from .hojo import SAMPLE_RATE


class StreamingResampler:
    def __init__(self, source_rate: int):
        factor = gcd(source_rate, SAMPLE_RATE)
        self.up = SAMPLE_RATE // factor
        self.down = source_rate // factor
        self.tail = np.empty(0, dtype=np.float32)
        self.input_start = 0
        self.input_end = 0
        self.output_end = 0
        if self.up != self.down:
            ratio = max(self.up, self.down)
            half_filter = 10 * ratio
            self.guard = (half_filter + self.up - 1) // self.up + 2
            self.window = firwin(2 * half_filter + 1, 1 / ratio, window=("kaiser", 5)).astype(np.float32)

    def feed(self, mono: np.ndarray) -> np.ndarray:
        self.input_end += len(mono)
        if self.up == self.down:
            self.output_end += len(mono)
            return mono
        self.tail = np.concatenate((self.tail, mono))
        # Hold the FIR's future input at the write edge until it is available.
        available = max(0, self.input_end - self.guard) * self.up // self.down
        return self._emit(available)

    def finish(self) -> np.ndarray:
        if self.up == self.down:
            return np.empty(0, dtype=np.float32)
        final_count = (self.input_end * self.up + self.down - 1) // self.down
        return self._emit(final_count)

    def _emit(self, available: int) -> np.ndarray:
        if available <= self.output_end:
            return np.empty(0, dtype=np.float32)
        filtered = resample_poly(self.tail, self.up, self.down, window=self.window)
        base = self.input_start * self.up // self.down
        result = filtered[self.output_end - base:available - base]
        if len(result) != available - self.output_end:
            raise RuntimeError("Microphone resampling produced an inconsistent sample count")
        self.output_end = available
        # Retain past filter context with the original decimation phase.
        past = self.output_end * self.down // self.up - self.guard
        next_start = max(self.input_start, past // self.down * self.down)
        self.tail = self.tail[next_start - self.input_start:].copy()
        self.input_start = next_start
        return result
