"""Frame-level analysis in numpy: frame energy, log mel-band energies, cepstra, deltas.

Frames are 25 ms long every 10 ms. Each frame gets its RMS level in dBFS (for
the energy detector) and its log mel-band energies in dB (for describing the
voice). The mel energies are decorrelated with a DCT (the MFCC recipe) before
they are averaged over a window, and the frame-by-frame models that place each
change of speaker also see their deltas. Long recordings are processed in
chunks so memory stays proportional to the number of frames, not to frames
times frame length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

FRAME_S = 0.025
HOP_S = 0.010
N_MELS = 40
F_MIN = 60.0
F_MAX = 8000.0
FLOOR_DB = -120.0
"""Digital silence is reported at this level instead of minus infinity."""

_POWER_EPS = 1e-12
_CHUNK_FRAMES = 8192


@dataclass
class Frames:
    """The frame-level description of a signal.

    Attributes:
        level_db: ``(n_frames,)`` RMS level of each frame in dBFS.
        logmel: ``(n_frames, n_mels)`` log mel-band energies in dB.
        frame_len: frame length in samples.
        hop: hop between frames in samples.
        sample_rate: samples per second.
    """

    level_db: np.ndarray
    logmel: np.ndarray
    frame_len: int
    hop: int
    sample_rate: int

    @property
    def n_frames(self) -> int:
        return int(self.level_db.shape[0])

    @property
    def hop_s(self) -> float:
        return self.hop / float(self.sample_rate)

    @property
    def offset_s(self) -> float:
        """Time of the start of frame 0's hop-long slot, which is centred on the frame."""
        return (self.frame_len - self.hop) / (2.0 * self.sample_rate)

    def frame_time(self, index: int) -> float:
        """Start time in seconds of frame ``index``'s slot (``index`` may equal ``n_frames``)."""
        return self.offset_s + index * self.hop_s

    def sample_span(self, first: int, last: int) -> Tuple[int, int]:
        """Sample range ``[start, end)`` covered by frames ``first .. last - 1``."""
        return first * self.hop, (last - 1) * self.hop + self.frame_len


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = N_MELS, f_min: float = F_MIN,
                   f_max: float = F_MAX) -> np.ndarray:
    """Triangular mel filters, ``(n_mels, n_fft // 2 + 1)``, each normalised to unit area."""
    top = min(float(f_max), sample_rate / 2.0)
    bottom = min(float(f_min), top / 4.0)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(bottom), _hz_to_mel(top), n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * (sample_rate / float(n_fft))
    bank = np.zeros((n_mels, freqs.size))
    for band in range(n_mels):
        low, centre, high = edges[band], edges[band + 1], edges[band + 2]
        rising = (freqs - low) / max(centre - low, 1e-9)
        falling = (high - freqs) / max(high - centre, 1e-9)
        bank[band] = np.maximum(0.0, np.minimum(rising, falling))
        if bank[band].sum() <= 0.0:
            # A band narrower than one FFT bin still gets the nearest bin.
            bank[band, int(np.argmin(np.abs(freqs - centre)))] = 1.0
    return bank / bank.sum(axis=1, keepdims=True)


def analyse(samples: np.ndarray, sample_rate: int, n_mels: int = N_MELS) -> Frames:
    """Frame ``samples`` and compute level and log mel-band energies for every frame.

    A signal shorter than one frame yields zero frames rather than an error.
    """
    frame_len = max(8, int(round(FRAME_S * sample_rate)))
    hop = max(1, int(round(HOP_S * sample_rate)))
    n = int(samples.shape[0])
    n_frames = 0 if n < frame_len else 1 + (n - frame_len) // hop
    level = np.full(n_frames, FLOOR_DB)
    logmel = np.full((n_frames, n_mels), FLOOR_DB)
    if n_frames == 0:
        return Frames(level, logmel, frame_len, hop, sample_rate)
    n_fft = 1 << (frame_len - 1).bit_length()
    window = np.hanning(frame_len)
    bank = mel_filterbank(sample_rate, n_fft, n_mels).T
    mel_power = np.zeros((n_frames, n_mels))
    view = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop]
    for start in range(0, n_frames, _CHUNK_FRAMES):
        block = np.array(view[start:start + _CHUNK_FRAMES], dtype=np.float64)
        block -= block.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(block * block, axis=1))
        with np.errstate(divide="ignore"):
            level[start:start + block.shape[0]] = np.maximum(20.0 * np.log10(rms), FLOOR_DB)
        spectrum = np.fft.rfft(block * window, n=n_fft, axis=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2) / float(frame_len)
        mel_power[start:start + block.shape[0]] = power @ bank
    logmel[:] = _shape_db(mel_power)
    return Frames(level, logmel, frame_len, hop, sample_rate)


def _shape_db(mel_power: np.ndarray) -> np.ndarray:
    """Log mel energies that describe the voice rather than the background.

    Each band's steady background (its 10th-percentile power over the recording)
    is subtracted before taking the log. Without this, background noise fills the
    spectral valleys of quieter passages, so the same voice spoken more softly
    would look like a different voice.
    """
    noise = np.percentile(mel_power, 10, axis=0)
    clean = np.maximum(mel_power - noise[None, :], 0.0)
    return np.maximum(10.0 * np.log10(clean + _POWER_EPS), FLOOR_DB)


def deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas over +/- ``width`` frames, edge-padded, same shape as ``features``."""
    if features.shape[0] == 0:
        return features.copy()
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    out = np.zeros_like(features, dtype=np.float64)
    count = features.shape[0]
    for step in range(1, width + 1):
        out += step * (padded[width + step:width + step + count] - padded[width - step:width - step + count])
    return out / (2.0 * sum(step * step for step in range(1, width + 1)))


N_CEPS = 20
"""Cepstral coefficients kept per frame (coefficient 0, the overall level, is dropped)."""


def cepstra(logmel: np.ndarray, n_ceps: int = N_CEPS) -> np.ndarray:
    """Decorrelate log mel-band energies with a DCT-II and keep coefficients ``1 .. n_ceps``.

    This is the familiar MFCC recipe. Coefficient 0 is the overall level and is
    dropped, so a louder or quieter take of the same voice looks the same.
    """
    n_mels = logmel.shape[1]
    keep = min(int(n_ceps), n_mels - 1)
    bands = np.arange(n_mels)
    basis = np.cos(np.pi / n_mels * (bands[None, :] + 0.5) * np.arange(1, keep + 1)[:, None])
    return logmel @ (basis.T * np.sqrt(2.0 / n_mels))


def window_descriptors(ceps: np.ndarray, active: np.ndarray,
                       windows: Sequence[Tuple[int, int]]) -> np.ndarray:
    """One vector per window: the mean cepstrum of its active frames.

    ``windows`` holds ``(first_frame, end_frame)`` pairs. Only frames marked
    ``active`` (above the energy threshold) are averaged; a window with fewer
    than three active frames falls back to all of its frames. Averaging over a
    second of speech blurs out which vowel was being said and keeps the shape of
    the voice: where its pitch harmonics and formants sit.
    """
    n_windows = len(windows)
    width = ceps.shape[1]
    if n_windows == 0:
        return np.zeros((0, width))
    weights = active.astype(np.float64)
    sums_active = np.vstack([np.zeros((1, width)), np.cumsum(ceps * weights[:, None], axis=0)])
    sums_all = np.vstack([np.zeros((1, width)), np.cumsum(ceps, axis=0)])
    count_active = np.concatenate([[0.0], np.cumsum(weights)])
    starts = np.array([w[0] for w in windows], dtype=np.int64)
    ends = np.array([w[1] for w in windows], dtype=np.int64)
    n_active = count_active[ends] - count_active[starts]
    use_active = n_active >= 3
    totals = np.where(use_active[:, None], sums_active[ends] - sums_active[starts],
                      sums_all[ends] - sums_all[starts])
    counts = np.where(use_active, n_active, (ends - starts).astype(np.float64))
    return totals / np.maximum(counts, 1.0)[:, None]


def frame_vectors(ceps: np.ndarray) -> np.ndarray:
    """Per-frame cepstra with their deltas, for modelling each voice frame by frame."""
    return np.hstack([ceps, deltas(ceps)])
