"""Synthetic voices for the tests, generated here; nothing is downloaded.

A voice is a harmonic source (a pitch contour with slow intonation and
jitter) shaped by three formants that move from vowel to vowel at a syllable
rate, with syllable-shaped loudness and a little breath noise. Two voices
differ by pitch and by a formant scale factor, as a man's and a woman's do.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

SR = 16000

VOWELS = np.array(
    [
        (730.0, 1090.0, 2440.0),
        (270.0, 2290.0, 3010.0),
        (300.0, 870.0, 2240.0),
        (530.0, 1840.0, 2480.0),
        (640.0, 1190.0, 2390.0),
        (490.0, 1350.0, 1690.0),
    ]
)
BANDWIDTHS = np.array([90.0, 110.0, 170.0])
GAINS = np.array([1.0, 0.55, 0.3])

SPEAKERS: Dict[str, Tuple[float, float]] = {
    # name: (pitch in Hz, formant scale)
    "man": (110.0, 1.0),
    "woman": (215.0, 1.18),
    "man_twin": (116.0, 1.02),
    "child": (300.0, 1.35),
}


def _knots(rng: np.random.Generator, n: int, sr: int, spacing_s: float) -> np.ndarray:
    """A smooth random curve in [-1, 1], linear between knots ``spacing_s`` apart."""
    count = int(np.ceil(n / (spacing_s * sr))) + 2
    values = rng.uniform(-1.0, 1.0, count)
    x = np.arange(count) * spacing_s * sr
    return np.interp(np.arange(n), x, values)


def voice(duration_s: float, speaker: str = "man", *, sr: int = SR, seed: int = 0,
          level: float = 0.1) -> np.ndarray:
    """``duration_s`` seconds of one synthetic talker, RMS ``level``."""
    f0, scale = SPEAKERS[speaker]
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    if n == 0:
        return np.zeros(0)
    t = np.arange(n) / sr
    contour = f0 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.28))
                    + 0.04 * _knots(rng, n, sr, 0.2))
    phase = 2 * np.pi * np.cumsum(contour) / sr

    syllable_s = 0.25
    count = int(np.ceil(duration_s / syllable_s)) + 2
    choice = rng.integers(0, len(VOWELS), count)
    centres = (np.arange(count) + 0.5) * syllable_s * sr
    formants = np.stack(
        [np.interp(np.arange(n), centres, VOWELS[choice, i]) for i in range(3)], axis=1
    ) * scale

    out = np.zeros(n)
    top = 0.45 * sr
    for h in range(1, int(top / (0.85 * f0)) + 1):
        freq = h * contour
        amp = np.zeros(n)
        for i in range(3):
            amp += GAINS[i] / (1.0 + ((freq - formants[:, i]) / (BANDWIDTHS[i] * scale)) ** 2)
        amp = (amp + 0.01) / h ** 0.7
        amp[freq >= top] = 0.0
        out += amp * np.sin(h * phase)

    within = (t / syllable_s) % 1.0
    envelope = 0.35 + 0.65 * np.sin(np.pi * within) ** 0.6
    envelope *= 0.8 + 0.2 * _knots(rng, n, sr, 0.5)
    ramp = min(n, int(0.02 * sr))
    envelope[:ramp] *= np.linspace(0.0, 1.0, ramp)
    envelope[n - ramp:] *= np.linspace(1.0, 0.0, ramp)
    out = out * envelope + 0.03 * rng.standard_normal(n) * envelope * np.std(out)
    return out * (level / max(float(np.sqrt(np.mean(out ** 2))), 1e-12))


def conversation(turns: Sequence[Tuple[str, float, float]], *, sr: int = SR, seed: int = 0,
                 noise_rms: float = 0.0005) -> Tuple[np.ndarray, List[Tuple[float, float, str]]]:
    """Concatenate turns of ``(speaker, seconds, pause_after_s)``.

    Returns the signal and the truth as ``(start_s, end_s, speaker)`` per turn.
    """
    pieces: List[np.ndarray] = [np.zeros(int(0.3 * sr))]
    truth: List[Tuple[float, float, str]] = []
    cursor = pieces[0].size
    for index, (speaker, seconds, pause) in enumerate(turns):
        chunk = voice(seconds, speaker, sr=sr, seed=seed * 1000 + index)
        truth.append((cursor / sr, (cursor + chunk.size) / sr, speaker))
        pieces.append(chunk)
        cursor += chunk.size
        gap = np.zeros(int(round(pause * sr)))
        pieces.append(gap)
        cursor += gap.size
    signal = np.concatenate(pieces + [np.zeros(int(0.3 * sr))])
    rng = np.random.default_rng(seed + 99)
    return signal + noise_rms * rng.standard_normal(signal.size), truth


TWO_VOICES = [
    ("man", 3.0, 0.5),
    ("woman", 2.5, 0.6),
    ("man", 2.0, 0.4),
    ("woman", 3.0, 0.0),
    ("man", 2.5, 0.5),
    ("woman", 2.0, 0.3),
]
"""Six turns alternating a man and a woman; the fourth hands over with no pause."""

ONE_VOICE = [
    ("man", 3.0, 0.5),
    ("man", 2.5, 0.6),
    ("man", 2.0, 0.4),
    ("man", 3.0, 0.0),
    ("man", 2.5, 0.5),
]
