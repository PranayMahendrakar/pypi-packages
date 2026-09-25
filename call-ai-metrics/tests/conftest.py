"""Synthetic calls for the tests. Everything is generated here; nothing is downloaded."""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

SR = 8000

Script = Sequence[Tuple[float, float, int]]


def speech_like(n: int, sr: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    """A voiced, syllable-modulated tone burst that behaves like speech to an energy detector."""
    t = np.arange(n) / sr
    wobble = 1.0 + 0.04 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0 * wobble) / sr
    voice = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 9))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.3 * t + rng.uniform(0, 2 * np.pi))
    ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.03) if n > 1 else np.ones(n)
    burst = voice * syllables * ramp
    peak = np.max(np.abs(burst)) if n else 1.0
    return 0.3 * burst / (peak or 1.0)


def make_call(
    script: Script,
    duration: float,
    *,
    sr: int = SR,
    n_channels: int = 2,
    noise_db: float = -65.0,
    seed: int = 0,
    f0s: Sequence[float] = (120.0, 210.0, 160.0),
) -> np.ndarray:
    """Render ``(start_s, end_s, channel)`` speech into a (samples, channels) float array."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    audio = np.zeros((n, n_channels))
    for start, end, channel in script:
        a, b = int(round(start * sr)), int(round(end * sr))
        audio[a:b, channel] += speech_like(b - a, sr, f0s[channel], rng)
    if noise_db is not None:
        audio += 10 ** (noise_db / 20.0) * rng.standard_normal(audio.shape)
    return audio


def add_bleed(audio: np.ndarray, source: int, target: int, level_db: float, delay_samples: int = 40) -> np.ndarray:
    """Return a copy with ``source`` leaking into ``target`` at ``level_db``, slightly delayed."""
    out = audio.copy()
    leak = np.zeros(audio.shape[0])
    leak[delay_samples:] = audio[: audio.shape[0] - delay_samples, source]
    # A little low-pass smearing so the leak is not a perfect copy.
    leak = np.convolve(leak, [0.25, 0.5, 0.25], mode="same")
    out[:, target] += 10 ** (level_db / 20.0) * leak
    return out


def ulaw_encode(x: np.ndarray) -> np.ndarray:
    """ITU-T G.711 mu-law encoder (the classic Sun g711.c algorithm)."""
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    mag = np.minimum(np.abs(pcm), 32635) + 0x84
    top = (mag >> 7) & 0xFF
    exponent = np.floor(np.log2(np.maximum(top, 1))).astype(np.int32)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def alaw_encode(x: np.ndarray) -> np.ndarray:
    """ITU-T G.711 A-law encoder (the classic Sun g711.c algorithm)."""
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int32) >> 3
    mask = np.where(pcm >= 0, 0xD5, 0x55)
    mag = np.where(pcm >= 0, pcm, -pcm - 1)
    seg_end = [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF]
    seg = np.full_like(mag, 8)
    for index in range(7, -1, -1):
        seg = np.where(mag <= seg_end[index], index, seg)
    shifted = np.where(seg < 2, mag >> 1, mag >> np.maximum(seg, 1))
    aval = (np.minimum(seg, 7) << 4) | (shifted & 0x0F)
    aval = np.where(seg >= 8, 0x7F, aval)
    return (aval ^ mask) & 0xFF


def write_wav(
    path: str,
    audio: np.ndarray,
    sr: int = SR,
    *,
    encoding: str = "pcm16",
    extensible: bool = False,
    extra_chunk: bool = False,
) -> None:
    """Write (samples, channels) float audio to a WAV file in the requested encoding."""
    data = np.atleast_2d(audio.T).T if audio.ndim == 1 else audio
    channels = data.shape[1]
    clipped = np.clip(data, -1.0, 1.0)
    if encoding == "pcm16":
        code, width, raw = 1, 2, np.round(clipped * 32767).astype("<i2").tobytes()
    elif encoding == "pcm8":
        code, width, raw = 1, 1, np.clip(np.round(clipped * 128 + 128), 0, 255).astype(np.uint8).tobytes()
    elif encoding == "pcm24":
        ints = np.round(clipped * 8388607).astype(np.int32).reshape(-1)
        packed = np.stack([ints & 0xFF, (ints >> 8) & 0xFF, (ints >> 16) & 0xFF], axis=1)
        code, width, raw = 1, 3, packed.astype(np.uint8).tobytes()
    elif encoding == "pcm32":
        code, width, raw = 1, 4, np.round(clipped * 2147483647).astype("<i4").tobytes()
    elif encoding == "float32":
        code, width, raw = 3, 4, data.astype("<f4").tobytes()
    elif encoding == "ulaw":
        code, width, raw = 7, 1, ulaw_encode(clipped).astype(np.uint8).tobytes()
    elif encoding == "alaw":
        code, width, raw = 6, 1, alaw_encode(clipped).astype(np.uint8).tobytes()
    else:
        raise ValueError(encoding)
    align = width * channels
    if extensible:
        guid_tail = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack(
            "<HHIIHHHHIH", 0xFFFE, channels, sr, sr * align, align, width * 8, 22, width * 8, 0, code
        ) + guid_tail
    else:
        fmt = struct.pack("<HHIIHH", code, channels, sr, sr * align, align, width * 8)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if extra_chunk:
        body = b"call-ai-metrics test\x00"
        chunks += b"LIST" + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")
    chunks += b"data" + struct.pack("<I", len(raw)) + raw
    with open(path, "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


# A realistic little call: agent (channel 0) and customer (channel 1).
# Includes one backchannel from the customer (0.3 s, inside agent speech) and
# one real interruption by the customer (1.5 s of overlap).
CALL_SCRIPT = [
    (1.0, 5.0, 0),
    (5.5, 8.0, 1),
    (8.6, 14.0, 0),
    (10.0, 10.3, 1),  # "mm-hm" inside the agent's speech
    (14.5, 17.0, 1),
    (17.6, 22.0, 0),
    (20.5, 25.0, 1),  # customer cuts in: 1.5 s of overlap
    (25.6, 29.0, 0),
]
CALL_DURATION = 30.0

# The same call without any overlap at all: nothing to find.
CLEAN_SCRIPT = [
    (1.0, 5.0, 0),
    (5.5, 8.0, 1),
    (8.6, 14.0, 0),
    (14.5, 17.0, 1),
    (17.6, 22.0, 0),
    (22.6, 25.0, 1),
    (25.6, 29.0, 0),
]


def true_talk(script: Script, channel: int) -> float:
    """Talk time of one channel in a script (spans on one channel never overlap here)."""
    return sum(end - start for start, end, ch in script if ch == channel)


@pytest.fixture
def call_audio() -> np.ndarray:
    return make_call(CALL_SCRIPT, CALL_DURATION, seed=1)


@pytest.fixture
def clean_audio() -> np.ndarray:
    return make_call(CLEAN_SCRIPT, CALL_DURATION, seed=2)


@pytest.fixture
def quickstart_segments() -> List[Tuple[float, float, str]]:
    return [
        (0.0, 6.5, "agent"),
        (6.9, 9.0, "customer"),
        (8.7, 20.0, "agent"),
        (12.0, 12.4, "customer"),
        (21.5, 24.0, "customer"),
        (23.0, 41.0, "agent"),
    ]


def assert_close(value: Optional[float], expected: float, tol: float) -> None:
    assert value is not None
    assert abs(value - expected) <= tol, "{} is not within {} of {}".format(value, tol, expected)
