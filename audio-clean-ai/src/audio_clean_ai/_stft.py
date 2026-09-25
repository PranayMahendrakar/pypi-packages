"""A short-time Fourier transform built on :mod:`numpy.fft` alone, in blocks.

The transform is a periodic Hann window at 75% overlap, analysed with
``rfft`` and resynthesised by weighted overlap-add, so an unmodified spectrum
comes back as the original samples to floating-point precision. The signal is
padded by half a frame at the front and enough at the back to cover the last
sample, and the output is cut back to exactly the input length.

Everything works on a block of frames at a time, so a long recording never has
its whole spectrogram in memory at once.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

HOPS_PER_FRAME = 4
"""75% overlap: every sample is covered by four frames."""


def _fast_size(target: int) -> int:
    """The smallest ``2**a * 3**b * 5**c`` (with ``a >= 2``) that is ``>= target``.

    FFTs of such sizes are fast, and ``a >= 2`` keeps the frame an exact
    multiple of :data:`HOPS_PER_FRAME`.
    """
    best = None
    power5 = 1
    while power5 < 2 * target + 8:
        power3 = power5
        while power3 < 2 * target + 8:
            size = 4 * power3
            while size < target:
                size *= 2
            if best is None or size < best:
                best = size
            power3 *= 3
        power5 *= 5
    assert best is not None
    return best


def frame_sizes(sample_rate: int, frame_ms: float) -> Tuple[int, int]:
    """``(n_fft, hop)`` for a sample rate and a nominal frame length."""
    target = max(32, int(round(sample_rate * frame_ms / 1000.0)))
    n_fft = _fast_size(target)
    return n_fft, n_fft // HOPS_PER_FRAME


def hann(n_fft: int) -> np.ndarray:
    """A periodic Hann window, the one that overlap-adds cleanly at 75%."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)


def smooth(values: np.ndarray, half_width: int, axis: int) -> np.ndarray:
    """Triangular smoothing along one axis, with edge values held at the ends.

    Holding the edge (rather than padding with zeros) keeps the first and last
    frames of a recording from being dragged towards zero.
    """
    if half_width <= 0 or values.shape[axis] == 0:
        return values
    ramp = np.arange(1, half_width + 2, dtype=np.float64)
    weights = np.concatenate([ramp, ramp[-2::-1]])
    weights /= weights.sum()
    pad = [(0, 0)] * values.ndim
    pad[axis] = (half_width, half_width)
    padded = np.pad(values, pad, mode="edge")
    length = values.shape[axis]
    out = np.zeros_like(values, dtype=np.float64)
    index = [slice(None)] * values.ndim
    for offset, weight in enumerate(weights):
        index[axis] = slice(offset, offset + length)
        out += weight * padded[tuple(index)]
    return out


class Stft:
    """The framing of one channel: padding, frame count, and the transforms.

    Args:
        n_samples: Length of the channel, at least 1.
        n_fft: Frame length in samples, a multiple of :data:`HOPS_PER_FRAME`.
        sample_rate: Frames per second, for bin frequencies.
    """

    def __init__(self, n_samples: int, n_fft: int, sample_rate: int) -> None:
        if n_samples < 1:
            raise ValueError("an STFT needs at least one sample")
        self.n_samples = n_samples
        self.n_fft = n_fft
        self.hop = n_fft // HOPS_PER_FRAME
        self.sample_rate = sample_rate
        self.pad_left = n_fft // 2
        self.n_frames = 1 + int(math.ceil(n_samples / self.hop))
        self.padded_length = (self.n_frames - 1) * self.hop + n_fft
        self.window = hann(n_fft)
        self.n_bins = n_fft // 2 + 1
        self.frequencies = np.arange(self.n_bins) * (sample_rate / float(n_fft))
        # Scale |X|^2 so that summing a frame's bins gives that frame's mean
        # square: one-sided spectrum, so interior bins count twice.
        weights = np.full(self.n_bins, 2.0)
        weights[0] = 1.0
        weights[-1] = 1.0
        self.bin_weight = weights / (n_fft * float(np.sum(self.window ** 2)))

    def pad(self, samples: np.ndarray) -> np.ndarray:
        """Zero-pad one channel so every sample sits under four full frames."""
        padded = np.zeros(self.padded_length, dtype=np.float64)
        padded[self.pad_left : self.pad_left + self.n_samples] = samples
        return padded

    def full_frames(self) -> np.ndarray:
        """Boolean per frame: True where the frame lies wholly inside the signal."""
        starts = np.arange(self.n_frames) * self.hop
        return (starts >= self.pad_left) & (
            starts + self.n_fft <= self.pad_left + self.n_samples
        )

    def frame_seconds(self, first: int, last: int) -> Tuple[float, float]:
        """The stretch of the recording, in seconds, that frames first..last span."""
        start = (first * self.hop - self.pad_left) / float(self.sample_rate)
        end = (last * self.hop + self.n_fft - self.pad_left) / float(self.sample_rate)
        duration = self.n_samples / float(self.sample_rate)
        return max(0.0, start), min(duration, end)

    def analyse(self, padded: np.ndarray, first: int, stop: int) -> np.ndarray:
        """Complex spectra of frames ``first`` to ``stop - 1``, shape (frames, bins)."""
        start = first * self.hop
        end = (stop - 1) * self.hop + self.n_fft
        view = np.lib.stride_tricks.sliding_window_view(padded[start:end], self.n_fft)
        frames = view[:: self.hop] * self.window
        return np.fft.rfft(frames, axis=1)

    def power(self, spectra: np.ndarray) -> np.ndarray:
        """Per-bin power, scaled so each row sums to that frame's mean square."""
        return (spectra.real ** 2 + spectra.imag ** 2) * self.bin_weight

    def overlap_add(self, output: np.ndarray, spectra: np.ndarray, first: int) -> None:
        """Resynthesise frames starting at ``first`` and add them into ``output``.

        ``output`` is laid out in hops: shape ``(n_frames - 1 + HOPS_PER_FRAME,
        hop)``, so frame ``f`` covers hop rows ``f`` to ``f + 3`` and the whole
        block is added with four slice operations instead of a Python loop.
        """
        count = spectra.shape[0]
        frames = np.fft.irfft(spectra, n=self.n_fft, axis=1) * self.window
        pieces = frames.reshape(count, HOPS_PER_FRAME, self.hop)
        for part in range(HOPS_PER_FRAME):
            output[first + part : first + part + count] += pieces[:, part, :]

    def new_output(self) -> np.ndarray:
        """An empty hop-major accumulator for :meth:`overlap_add`."""
        return np.zeros(
            (self.n_frames - 1 + HOPS_PER_FRAME, self.hop), dtype=np.float64
        )

    def finish(self, output: np.ndarray) -> np.ndarray:
        """Divide out the window overlap and cut back to the input length."""
        squared = (self.window ** 2).reshape(HOPS_PER_FRAME, self.hop)
        norm = np.zeros_like(output)
        for part in range(HOPS_PER_FRAME):
            norm[part : part + self.n_frames] += squared[part]
        flat = output.reshape(-1)
        flat_norm = norm.reshape(-1)
        keep = slice(self.pad_left, self.pad_left + self.n_samples)
        # Every kept sample sits under at least two frames with non-zero window
        # weight, so the guard never actually triggers inside the signal.
        denominator = np.maximum(flat_norm[keep], 1e-12)
        return flat[keep] / denominator
