"""Signals and WAV files the tests build for themselves. Nothing is downloaded."""

from __future__ import annotations

import struct
import wave

import numpy as np

SR = 16000


def hum(sr=SR, seconds=4.0, f0=120.0, noise=0.004, seed=0, harmonics=3, wobble=0.0):
    """Steady machine hum: a fundamental, falling harmonics, a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * sr))) / sr
    x = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        x += (0.2 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    if wobble:
        x *= 1.0 + wobble * np.sin(2 * np.pi * 0.7 * t)
    return x + noise * rng.standard_normal(t.size)


def add_knock(x, sr=SR, at=2.0, amp=0.5, seed=100):
    """A 30 ms decaying broadband impact starting at ``at`` seconds."""
    y = x.copy()
    rng = np.random.default_rng(seed)
    n = int(0.03 * sr)
    i = int(round(at * sr))
    y[i : i + n] += amp * np.exp(-np.arange(n) / (0.004 * sr)) * rng.standard_normal(n)
    return y


def add_tone(x, sr=SR, start=2.0, amp=0.05, freq=3100.0, ramp_s=0.02):
    """A tone that fades in over ``ramp_s`` at ``start`` and stays to the end."""
    y = x.copy()
    t = np.arange(y.size) / sr
    envelope = np.clip((t - start) / ramp_s, 0.0, 1.0)
    return y + amp * envelope * np.sin(2 * np.pi * freq * t)


def scale_span(x, sr=SR, at=2.0, seconds=0.5, factor=0.0):
    """Multiply ``seconds`` of audio starting at ``at`` by ``factor``."""
    y = x.copy()
    i = int(round(at * sr))
    y[i : i + int(round(seconds * sr))] *= factor
    return y


def write_pcm(path, samples, sr=SR, bits=16):
    """Write float samples in [-1, 1] as integer PCM with the standard ``wave`` module.

    ``samples`` is ``(n,)`` or ``(n, channels)``; values beyond full scale are
    clipped to the rail, exactly as a recorder would.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    full = 2 ** (bits - 1)
    ints = np.clip(np.round(data * full), -full, full - 1).astype(np.int64)
    if bits == 8:
        raw = (ints + 128).astype(np.uint8).tobytes()
    elif bits == 16:
        raw = ints.astype("<i2").tobytes()
    elif bits == 24:
        as32 = ints.astype("<i4").reshape(-1, 1).view(np.uint8).reshape(-1, 4)
        raw = as32[:, :3].tobytes()
    elif bits == 32:
        raw = ints.astype("<i4").tobytes()
    else:
        raise ValueError(bits)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(bits // 8)
        handle.setframerate(sr)
        handle.writeframes(raw)
    return path


def write_float_wav(path, samples, sr=SR, extensible=False):
    """Write 32-bit IEEE float WAV by hand (``wave`` cannot write it)."""
    data = np.asarray(samples, dtype="<f4")
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    payload = data.tobytes()
    if extensible:
        guid = struct.pack("<H", 3) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, sr, sr * 4 * channels, 4 * channels, 32)
        fmt += struct.pack("<HHI", 22, 32, 0) + guid
    else:
        fmt = struct.pack("<HHIIHH", 3, channels, sr, sr * 4 * channels, 4 * channels, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    with open(path, "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path
