"""Test signals, all generated with numpy, and a WAV writer to put them in files.

Nothing here reads anything off the network or off disk that it did not just
write. Every signal is built from a seeded generator or from pure arithmetic, so
the whole suite is reproducible on any machine.
"""

from __future__ import annotations

import contextlib
import wave

import numpy as np
import pytest

RATE = 16000
"""The sample rate most fixtures use."""


def sine(seconds: float = 1.0, rate: int = RATE, freq: float = 440.0, amplitude: float = 0.3):
    """A plain sine wave: steady, tonal, and deliberately not speech."""
    t = np.arange(int(seconds * rate)) / float(rate)
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def band_limited_noise(
    seconds: float = 1.0,
    rate: int = RATE,
    top_hz: float = 3400.0,
    amplitude: float = 0.1,
    seed: int = 0,
):
    """Noise with everything above ``top_hz`` removed, built in the frequency domain.

    Used to prove the bandwidth measure finds the edge it was given, which is
    the whole point of that measure.
    """
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(rate))
    spectrum[freqs > float(top_hz)] = 0.0
    signal = np.fft.irfft(spectrum, n=n)
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak
    return amplitude * signal


def speech_like(
    seconds: float = 3.0,
    rate: int = RATE,
    rms: float = 0.1,
    seed: int = 0,
    modulation_hz: float = 3.5,
):
    """Noise shaped and modulated to behave the way speech behaves.

    Band energy in the right place, a spectral centre of gravity where a voice
    puts it, and a level that swings at syllable rate. It is not speech, but it
    passes for it on every measure this package takes, which is exactly what a
    test needs.
    """
    n = int(seconds * rate)
    t = np.arange(n) / float(rate)
    rng = np.random.default_rng(seed)
    hiss = rng.standard_normal(n + 4)
    shaped = np.convolve(hiss, np.ones(5) / 5.0, "valid")
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * modulation_hz * t)
    signal = shaped * envelope
    level = float(np.sqrt(np.mean(signal * signal)))
    if level > 0.0:
        signal = signal * (float(rms) / level)
    return signal


def clipped_sine(seconds: float = 1.0, rate: int = RATE, freq: float = 300.0, drive: float = 3.0):
    """A sine driven into the rails, so a known share of it sits at full scale."""
    return np.clip(sine(seconds, rate, freq, amplitude=1.0) * float(drive), -1.0, 1.0)


def clipped_voice(
    seconds: float = 4.0,
    rate: int = RATE,
    rms: float = 0.1,
    drive: float = 2.5,
    seed: int = 0,
):
    """Speech driven into the rails: good speech in every respect except clipping.

    A clipped sine fails four measures at once, which hides whether clipping
    alone is enough to condemn a take. This one still reads as speech, still has
    its full band and still stands well clear of the noise; only its peaks are
    destroyed.
    """
    voice = speech_like(seconds=seconds, rate=rate, rms=rms, seed=seed)
    return np.clip(voice * float(drive), -1.0, 1.0)


def noisy_voice(
    seconds: float = 4.0,
    rate: int = RATE,
    rms: float = 0.1,
    noise_rms: float = 0.04,
    seed: int = 0,
):
    """Speech with a hiss under it close enough that words will be lost in it."""
    voice = speech_like(seconds=seconds, rate=rate, rms=rms, seed=seed)
    rng = np.random.default_rng(seed + 101)
    hiss = rng.standard_normal(voice.size)
    hiss = hiss / float(np.sqrt(np.mean(hiss * hiss))) * float(noise_rms)
    mixed = voice + hiss
    level = float(np.sqrt(np.mean(mixed * mixed)))
    return mixed * (float(rms) / level) if level > 0.0 else mixed


def telephone_band(
    seconds: float = 4.0,
    rate: int = 48000,
    top_hz: float = 3400.0,
    rms: float = 0.1,
    seed: int = 0,
):
    """Telephone-band speech carried in a wideband file, the fault bandwidth exists for.

    The header says 48 kHz and the sound stops at 3.4 kHz: every consonant that
    lived above that is already gone, and no later pass brings it back.
    """
    voice = speech_like(seconds=seconds, rate=rate, rms=rms, seed=seed)
    spectrum = np.fft.rfft(voice)
    freqs = np.fft.rfftfreq(voice.size, d=1.0 / float(rate))
    spectrum[freqs > float(top_hz)] = 0.0
    band = np.fft.irfft(spectrum, n=voice.size)
    level = float(np.sqrt(np.mean(band * band)))
    return band * (float(rms) / level) if level > 0.0 else band


def silence(seconds: float = 1.0, rate: int = RATE):
    """Digital silence: every sample exactly zero."""
    return np.zeros(int(seconds * rate), dtype=np.float64)


def write_wav(path, samples, rate: int = RATE, width: int = 2, channels: int = 1) -> str:
    """Write float samples to a PCM WAV file at 8, 16, 24 or 32 bit.

    Args:
        path: where to write.
        samples: float samples on a -1.0 to 1.0 scale. A 2-D array is written
            channel-per-column.
        rate: samples per second.
        width: bytes per sample, 1, 2, 3 or 4.
        channels: how many channels to declare; interleaving is done here.

    Returns:
        The path written, as a string.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 2:
        channels = int(values.shape[1])
        values = values.reshape(-1)
    clipped = np.clip(values, -1.0, 1.0)

    if width == 1:
        raw = np.clip(np.round(clipped * 127.0) + 128.0, 0, 255).astype(np.uint8).tobytes()
    elif width == 2:
        raw = np.clip(np.round(clipped * 32767.0), -32768, 32767).astype("<i2").tobytes()
    elif width == 3:
        full = 1 << 23
        ints = np.clip(np.round(clipped * (full - 1)), -full, full - 1).astype(np.int32)
        unsigned = np.where(ints < 0, ints + (1 << 24), ints).astype(np.uint32)
        octets = np.empty((unsigned.size, 3), dtype=np.uint8)
        octets[:, 0] = unsigned & 0xFF
        octets[:, 1] = (unsigned >> 8) & 0xFF
        octets[:, 2] = (unsigned >> 16) & 0xFF
        raw = octets.tobytes()
    elif width == 4:
        full = 1 << 31
        raw = np.clip(np.round(clipped * (full - 1)), -full, full - 1).astype("<i4").tobytes()
    else:  # pragma: no cover - guarded by the callers
        raise ValueError("unsupported width {}".format(width))

    text = str(path)
    with contextlib.closing(wave.open(text, "wb")) as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(int(width))
        handle.setframerate(int(rate))
        handle.writeframes(raw)
    return text


@pytest.fixture
def clean_speech():
    """Three seconds of well-recorded speech-like audio at 16 kHz."""
    return speech_like()


@pytest.fixture
def wav_dir(tmp_path):
    """A directory to write test WAV files into."""
    folder = tmp_path / "audio"
    folder.mkdir()
    return folder
