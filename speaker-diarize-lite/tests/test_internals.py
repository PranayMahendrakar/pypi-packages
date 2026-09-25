"""The building blocks: features, the energy detector, and the clustering."""

from __future__ import annotations

import numpy as np
import pytest

from speaker_diarize_lite import _cluster as C
from speaker_diarize_lite import _features as F
from speaker_diarize_lite import _vad as V


def test_mel_filterbank_rows_are_normalised_and_cover_low_rates():
    for rate, n_fft in ((8000, 256), (16000, 512), (44100, 2048), (2000, 64)):
        bank = F.mel_filterbank(rate, n_fft)
        assert bank.shape == (F.N_MELS, n_fft // 2 + 1)
        assert np.allclose(bank.sum(axis=1), 1.0)
        assert (bank >= 0).all()


def test_analyse_short_and_normal_signals():
    empty = F.analyse(np.zeros(10), 16000)
    assert empty.n_frames == 0
    tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    frames = F.analyse(tone, 16000)
    assert frames.n_frames == 1 + (16000 - 400) // 160
    assert np.allclose(frames.level_db[5:-5], 20 * np.log10(0.5 / np.sqrt(2)), atol=0.2)
    silent = F.analyse(np.zeros(16000), 16000)
    assert (silent.level_db == F.FLOOR_DB).all()


def test_cepstra_ignore_overall_level():
    rng = np.random.default_rng(0)
    logmel = rng.normal(-40, 5, size=(50, F.N_MELS))
    assert np.allclose(F.cepstra(logmel), F.cepstra(logmel + 17.0))
    assert F.cepstra(logmel).shape == (50, F.N_CEPS)


def test_deltas_of_a_ramp_are_its_slope():
    ramp = np.arange(20, dtype=float)[:, None] * np.ones((1, 3)) * 2.0
    assert np.allclose(F.deltas(ramp)[3:-3], 2.0)
    assert F.deltas(np.zeros((0, 3))).shape == (0, 3)


def test_window_descriptors_use_active_frames():
    ceps = np.zeros((10, 2))
    ceps[5:] = 4.0
    active = np.array([False] * 5 + [True] * 5)
    out = F.window_descriptors(ceps, active, [(0, 10), (0, 4)])
    assert np.allclose(out[0], 4.0)  # only the active half is averaged
    assert np.allclose(out[1], 0.0)  # too few active frames: all frames are used
    assert F.window_descriptors(ceps, active, []).shape == (0, 2)


def test_energy_detector_finds_the_loud_part_and_bridges_short_gaps():
    level = np.full(300, -80.0)
    level[50:120] = -20.0
    level[140:200] = -22.0  # 0.2 s gap: bridged
    level[260:265] = -20.0  # 50 ms click: dropped
    found = V.detect_activity(level, 0.01)
    assert not found.silent
    assert found.regions == [(50, 200)]
    assert V.detect_activity(np.full(100, -90.0), 0.01).silent
    assert V.detect_activity(np.zeros(0), 0.01).silent


def test_energy_detector_flat_recording():
    found = V.detect_activity(np.full(200, -25.0), 0.01)
    assert found.flat and found.regions == [(0, 200)]


def test_runs_helper():
    assert V.runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert V.runs(np.zeros(0, dtype=bool)) == []


def _blobs(n=60, gap=10.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, size=(n, 3))
    b = rng.normal(0, 1, size=(n, 3)) + np.array([gap, 0, 0])
    return np.vstack([a, b])


def test_kmeans_is_deterministic_and_finds_blobs():
    x = _blobs()
    first, centres = C.kmeans(x, 2, random_state=1)
    second, _ = C.kmeans(x, 2, random_state=1)
    assert np.array_equal(first, second)
    assert len(set(first[:60])) == 1 and len(set(first[60:])) == 1 and first[0] != first[-1]
    assert centres.shape == (2, 3)


def test_average_linkage_and_cut():
    x = _blobs(20)
    merges = C.average_linkage(x)
    assert len(merges) == 39
    labels = C.cut(merges, 40, 2)
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[-1]
    assert (C.cut(merges, 40, 1) == 0).all()
    assert C.average_linkage(x[:1]) == []


def test_separation_scale():
    x = _blobs(200, gap=6.0)
    assert C.separation(x[:200], x[200:]) == pytest.approx(6.0, rel=0.15)
    same = np.ones((5, 2))
    assert C.separation(same, same + 1.0) == float("inf")
    assert C.separation(same, same) == 0.0
    assert C.separation(same[:0], same) == 0.0


def test_one_blob_is_one_speaker_and_two_blobs_are_two():
    one = np.random.default_rng(3).normal(size=(80, 4))
    assert C.cluster_windows(one, num_speakers=None, max_speakers=8, min_windows=3).k == 1
    two = C.cluster_windows(_blobs(40, gap=12.0), num_speakers=None, max_speakers=8, min_windows=3)
    assert two.k == 2 and two.evidence == 2 and two.separation >= C.SEPARATION_THRESHOLD


def test_many_windows_take_the_summary_path_deterministically():
    x = _blobs(300, gap=12.0)
    first = C.cluster_windows(x, num_speakers=None, max_speakers=8, min_windows=3, random_state=5)
    second = C.cluster_windows(x, num_speakers=None, max_speakers=8, min_windows=3, random_state=5)
    assert first.k == 2 and np.array_equal(first.labels, second.labels)


def test_forced_counts_and_tiny_inputs():
    x = _blobs(30, gap=12.0)
    forced = C.cluster_windows(x, num_speakers=4, max_speakers=8, min_windows=3)
    assert forced.k == 4 and forced.evidence == 2
    assert C.cluster_windows(x[:1], num_speakers=3, max_speakers=8, min_windows=2).k == 1
    assert C.cluster_windows(x[:0], num_speakers=None, max_speakers=8, min_windows=2).k == 0
    assert C.cluster_windows(x[:3], num_speakers=5, max_speakers=8, min_windows=2).k == 3


def test_a_bridge_between_two_voices_is_not_a_third():
    labels = np.array([0, 0, 0, 0, 2, 2, 1, 1, 1, 1])
    regions = np.zeros(10, dtype=int)
    assert C.is_blend(labels, regions, 2, max_run=3)
    assert not C.is_blend(labels, regions, 0, max_run=3)  # starts the stretch
    assert not C.is_blend(labels, None, 2, max_run=3)
    back_and_forth = np.array([0, 0, 2, 2, 0, 0])
    assert not C.is_blend(back_and_forth, np.zeros(6, dtype=int), 2, max_run=3)
    long_run = np.array([0, 2, 2, 2, 2, 1])
    assert not C.is_blend(long_run, np.zeros(6, dtype=int), 2, max_run=3)


def test_heard_seconds_counts_overlaps_once():
    spans = np.array([[0.0, 1.0], [0.25, 1.25], [3.0, 4.0]])
    assert C.heard_seconds(spans, np.array([True, True, True])) == pytest.approx(2.25)
    assert C.heard_seconds(spans, np.array([False, False, False])) == 0.0


def test_projection_and_standardising_edge_cases():
    assert C.project(np.zeros((1, 3))).shape == (1, 1)
    assert np.allclose(C.project(np.ones((5, 3))), 0.0)
    z = C.standardise(np.column_stack([np.arange(5.0), np.ones(5)]))
    assert np.allclose(z[:, 1], 0.0) and np.isclose(z[:, 0].std(), 1.0)
    assert np.allclose(np.linalg.norm(C.unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]])), axis=1), [1.0, 0.0])
