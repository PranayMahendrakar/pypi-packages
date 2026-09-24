"""Signal generators shared by the tests.

Every signal here is built from numpy alone and from a fixed seed, so the whole
suite downloads nothing, touches no network, and gives the same answer on every
machine and every run.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 16000
"""The sample rate most tests use, when the rate itself is not what is tested."""

FRAME_MS = 30.0
"""The default frame length, repeated here so tolerances can be derived from it."""


def coloured_noise(
    n: int, slope: float = -1.0, seed: int = 0, level: float = 0.02
) -> np.ndarray:
    """Noise with a power-law spectrum, scaled to ``level`` RMS.

    ``slope=0`` is white, ``-1`` is pink (a fan, a room), ``-2`` is brown
    (traffic, an air conditioner, a building's rumble), ``+1`` is blue (hiss).
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequency = np.arange(spectrum.size, dtype=np.float64)
    frequency[0] = 1.0  # DC would be a divide-by-zero, and carries no colour
    shaped = np.fft.irfft(spectrum * frequency ** (slope / 2.0), n)
    return shaped / (np.std(shaped) + 1e-12) * level


def speech_like(
    n: int, sr: int = SR, f0: float = 130.0, seed: int = 3, level: float = 0.30
) -> np.ndarray:
    """A burst that looks like voiced speech to a spectral detector.

    A harmonic stack on a wavering fundamental, shaped by three formant-ish
    bumps and modulated at 4 Hz, which is roughly a syllable rate. This is not
    speech - nothing generated from numpy is - but it carries the two properties
    the detector actually keys on: power concentrated into harmonics rather than
    spread evenly, and an envelope that moves.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    frequency = f0 * (1.0 + 0.04 * np.sin(2.0 * np.pi * 0.9 * t))
    phase = 2.0 * np.pi * np.cumsum(frequency) / sr
    signal = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 60):
        centre = harmonic * f0
        if centre > sr / 2.0:
            break
        gain = sum(
            weight * np.exp(-((centre - formant) ** 2) / (2.0 * 260.0 ** 2))
            for formant, weight in ((600.0, 1.0), (1150.0, 0.7), (2500.0, 0.35))
        )
        signal += gain * np.sin(harmonic * phase) / harmonic
    signal *= 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * t)
    signal += 0.01 * rng.standard_normal(n)
    return signal / (np.max(np.abs(signal)) + 1e-12) * level


def planted(
    start_s: float = 1.0,
    end_s: float = 2.0,
    total_s: float = 4.0,
    sr: int = SR,
    slope: float = -1.0,
    noise_level: float = 0.02,
    seed: int = 1,
    f0: float = 130.0,
) -> np.ndarray:
    """Known speech at a known time in known noise - the detector's real exam."""
    audio = coloured_noise(int(total_s * sr), slope=slope, seed=seed, level=noise_level)
    begin, finish = int(start_s * sr), int(end_s * sr)
    audio[begin:finish] += speech_like(finish - begin, sr=sr, f0=f0)
    return audio


@pytest.fixture
def burst() -> np.ndarray:
    """One second of speech from 1.0 s to 2.0 s in four seconds of pink noise."""
    return planted()
