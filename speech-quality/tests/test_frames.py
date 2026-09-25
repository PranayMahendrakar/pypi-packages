"""Framing: the per-frame numbers, and how much memory taking them is allowed to cost.

The level of each frame comes from a running sum of squares, which is linear in
the samples. The peak of each frame used to come from indexing a
``sliding_window_view`` with the frame starts, which copies every frame in full:
frames x frame_length rather than samples, twice the audio at the default 2:1
overlap and 2.8 GB for an hour of 48 kHz interview. That is the one place in the
package that scaled with anything but the recording, and these tests hold it
down.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from conftest import RATE, speech_like
from speech_quality._frames import (
    PEAK_BLOCK_SAMPLES,
    _peak_block_frames,
    frame_peaks,
    frame_signal,
)
from speech_quality.thresholds import DEFAULT_THRESHOLDS


def naive_peaks(signal, starts, length):
    """What the peak of each frame is, worked out the obvious slow way."""
    return np.array([np.abs(signal[start : start + length]).max() for start in starts])


@pytest.mark.parametrize(
    "length,hop", [(1, 1), (2, 1), (7, 3), (64, 16), (512, 1), (1024, 1024), (4999, 1)]
)
def test_frame_peaks_matches_a_naive_per_frame_maximum(length, hop):
    """Blocking the work changes the cost, not the answer, at any geometry."""
    signal = np.random.default_rng(0).standard_normal(5000)
    starts = np.arange(0, signal.size - length + 1, hop, dtype=np.int64)

    assert np.array_equal(frame_peaks(signal, starts, length), naive_peaks(signal, starts, length))


def test_frame_peaks_reads_both_rails():
    """Peak is the largest excursion either way, so a deep negative still counts."""
    signal = np.array([0.1, -0.9, 0.2, 0.3, 0.8, -0.05])

    peaks = frame_peaks(signal, np.array([0, 2, 4], dtype=np.int64), 2)

    assert peaks.tolist() == [0.9, 0.3, 0.8]


@pytest.mark.parametrize(
    "length", [1, 16, 1000, PEAK_BLOCK_SAMPLES // 2, PEAK_BLOCK_SAMPLES, PEAK_BLOCK_SAMPLES * 3]
)
def test_a_peak_block_never_holds_more_than_the_ceiling(length):
    """However long the frames are, one block stays inside the budget - or is one frame."""
    frames_per_block = _peak_block_frames(length)

    assert frames_per_block >= 1
    assert frames_per_block * length <= max(length, PEAK_BLOCK_SAMPLES)


def numpy_allocations_reach_tracemalloc():
    """Whether this numpy reports its own allocations, which the memory test needs."""
    tracemalloc.start()
    try:
        block = np.zeros(1 << 20)  # 8 MB
        seen = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    del block
    return seen > 4 * (1 << 20)


def test_framing_never_materialises_every_frame():
    """A minute of 48 kHz audio must not need eight copies of itself to frame."""
    if not numpy_allocations_reach_tracemalloc():  # pragma: no cover - numpy dependent
        pytest.skip("this numpy does not report its allocations to tracemalloc")

    rate = 48000
    signal = np.sin(np.arange(60 * rate) / 97.0)  # 60 s, 23 MB of float64
    # Long frames stepped short: frames x frame_length is eight times the audio,
    # which is what materialising them all would have to hold.
    limits = DEFAULT_THRESHOLDS.replace(frame_seconds=0.128, hop_seconds=0.016)

    tracemalloc.start()
    try:
        frames = frame_signal(signal, rate, limits)
        peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    dense_bytes = frames.count * frames.length * 8
    assert dense_bytes > 6 * signal.nbytes, "this test means nothing without the overlap"
    # Measured at 1.7x the signal here, against 10x for the version that copied
    # every frame; 3x leaves room for numpy to allocate differently without
    # letting the old cost back in.
    assert peak_bytes < 3 * signal.nbytes


def test_frame_levels_still_describe_the_recording():
    """The blocked peak path feeds the same Frames every measure reads."""
    voice = speech_like(seconds=2.0)

    frames = frame_signal(voice, RATE, DEFAULT_THRESHOLDS)

    assert frames.count > 1
    assert frames.peak.shape == frames.rms.shape == frames.starts.shape
    assert np.all(frames.peak >= frames.rms - 1e-12)
    assert frames.peak.max() == pytest.approx(np.abs(voice).max())


def test_a_recording_shorter_than_one_frame_is_a_single_frame():
    """The short path has its own peak, and it is still the right one."""
    signal = np.array([0.0, -0.4, 0.25])

    frames = frame_signal(signal, RATE, DEFAULT_THRESHOLDS)

    assert frames.count == 1
    assert frames.peak[0] == pytest.approx(0.4)
