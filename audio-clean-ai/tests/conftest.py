"""Signal generators shared by the tests.

Every signal here is built from numpy alone and from a fixed seed, so the suite
downloads nothing, touches no network, and gives the same answer on every
machine and every run.
"""

from __future__ import annotations

import wave

import numpy as np

SR = 16000
"""The sample rate most tests use, when the rate itself is not what is tested."""


def voice_like(
    n: int,
    sr: int = SR,
    f0: float = 140.0,
    level: float = 0.3,
    gaps: bool = True,
    lead_s: float = 0.5,
) -> np.ndarray:
    """A signal that looks like voiced speech to a spectral gate.

    A harmonic stack on a slowly wavering fundamental, shaped by three
    formant-like bumps. With ``gaps`` it is chopped into 220 ms "syllables"
    with 120 ms pauses and a silent lead-in and tail of ``lead_s``; without, it
    runs from end to end with only a gentle 4 Hz swell, so it never pauses.
    Not speech - nothing generated from numpy is - but it has what the gate
    keys on: power packed into harmonics, in the speech band, that comes and goes.
    """
    t = np.arange(n, dtype=np.float64) / sr
    frequency = f0 * (1.0 + 0.03 * np.sin(2.0 * np.pi * 0.8 * t))
    phase = 2.0 * np.pi * np.cumsum(frequency) / sr
    signal = np.zeros(n)
    for harmonic in range(1, 40):
        centre = harmonic * f0
        if centre > min(sr / 2.0 - 200.0, 4000.0):
            break
        gain = sum(
            weight * np.exp(-((centre - formant) ** 2) / (2.0 * 250.0 ** 2))
            for formant, weight in ((600.0, 1.0), (1200.0, 0.6), (2500.0, 0.3))
        )
        signal += (gain + 0.02) * np.sin(harmonic * phase) / harmonic
    if gaps:
        envelope = np.zeros(n)
        syllable, pause, lead = int(0.22 * sr), int(0.12 * sr), int(lead_s * sr)
        position = lead
        while position + syllable < n - lead:
            envelope[position : position + syllable] = np.hanning(syllable) ** 0.5
            position += syllable + pause
        signal *= envelope
    else:
        signal *= 0.8 + 0.2 * np.sin(2.0 * np.pi * 4.0 * t)
    return signal / (np.max(np.abs(signal)) + 1e-12) * level


def white(n: int, level: float = 0.05, seed: int = 1) -> np.ndarray:
    """White noise with a standard deviation of ``level`` - hiss."""
    return level * np.random.default_rng(seed).standard_normal(n)


def hum(n: int, sr: int = SR, level: float = 0.05, mains: float = 50.0) -> np.ndarray:
    """Mains hum: a fundamental and two harmonics, perfectly steady."""
    t = np.arange(n, dtype=np.float64) / sr
    wave_ = (
        np.sin(2 * np.pi * mains * t)
        + 0.5 * np.sin(2 * np.pi * 2 * mains * t + 0.3)
        + 0.25 * np.sin(2 * np.pi * 3 * mains * t + 1.1)
    )
    return level * wave_ / np.max(np.abs(wave_))


def true_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SNR of ``estimate`` against the known clean ``reference``, in dB."""
    error = estimate - reference
    return 10.0 * np.log10(np.sum(reference ** 2) / np.sum(error ** 2))


def harmonic_ratio_db(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:
    """How much of the reference's strongest speech-band bins survive, in dB.

    Takes the 1% loudest FFT bins of the clean reference inside 300-3400 Hz -
    the harmonics - and compares the estimate's energy there with the
    reference's. 0 dB means the voice came through untouched.
    """
    ref = np.abs(np.fft.rfft(reference)) ** 2
    est = np.abs(np.fft.rfft(estimate)) ** 2
    frequency = np.fft.rfftfreq(reference.size, 1.0 / sr)
    band = (frequency >= 300.0) & (frequency <= 3400.0)
    strong = band & (ref > np.percentile(ref[band], 99))
    return float(10.0 * np.log10(est[strong].sum() / ref[strong].sum()))


def write_pcm(path: str, frames: np.ndarray, sr: int, width: int) -> None:
    """Write raw integer frames (already in the file's integer range) as a WAV."""
    data = np.asarray(frames)
    channels = 1 if data.ndim == 1 else data.shape[1]
    if width == 1:
        payload = data.astype(np.uint8).tobytes()
    elif width == 2:
        payload = data.astype("<i2").tobytes()
    else:
        raise ValueError(width)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(sr)
        handle.writeframes(payload)
