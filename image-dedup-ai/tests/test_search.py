"""Pair search: the multi-index shortcut must return exactly what the exhaustive search does."""
from __future__ import annotations

import numpy as np
import pytest

from image_dedup_ai import _search


def planted(n: int, seed: int = 0, bits: int = 64):
    """Random hashes with near copies planted at known distances."""
    rng = np.random.default_rng(seed)
    nbytes = bits // 8
    packed = rng.integers(0, 256, (n, nbytes), dtype=np.uint8)
    truth = {}
    for k in range(60):
        a, b = 2 * k, 2 * k + 1
        flips = k % 9
        row = np.unpackbits(packed[a])
        row[rng.choice(bits, flips, replace=False)] ^= 1
        packed[b] = np.packbits(row)
        truth[(a, b)] = flips
    return packed, truth


def as_set(result):
    i, j, d = result
    return set(zip(i.tolist(), j.tolist(), d.tolist()))


def test_popcount_native_and_table_agree():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2**63, (50, 3), dtype=np.uint64) ^ rng.integers(0, 2**63, (50, 3), dtype=np.uint64)
    table = _search.popcount(x, native=False)
    expected = np.array([[bin(int(v)).count("1") for v in row] for row in x])
    assert (table == expected).all()
    assert (_search.popcount(x) == expected).all()


@pytest.mark.parametrize("max_dist", [0, 2, 6, 9])
def test_mih_matches_brute_force_and_finds_every_planted_pair(max_dist):
    packed, truth = planted(3000)
    brute = as_set(_search.similar_pairs(packed, 64, max_dist, strategy="brute"))
    mih = as_set(_search.similar_pairs(packed, 64, max_dist, strategy="mih"))
    assert mih == brute
    expected = {(a, b, d) for (a, b), d in truth.items() if d <= max_dist}
    assert expected <= brute
    # clean random hashes almost never land this close: nothing beyond the planted pairs
    assert brute == expected


def test_auto_uses_the_shortcut_on_large_sets_and_agrees():
    packed, truth = planted(_search.BRUTE_FORCE_MAX + 500, seed=3)
    assert _search._mih_worthwhile(packed.shape[0], 64, 6)
    assert not _search._mih_worthwhile(packed.shape[0], 64, 40), "loose thresholds fall back to exhaustive"
    auto = as_set(_search.similar_pairs(packed, 64, 6))
    assert auto == {(a, b, d) for (a, b), d in truth.items() if d <= 6}


def test_longer_hashes_and_edge_inputs():
    packed, truth = planted(800, seed=5, bits=256)
    brute = as_set(_search.similar_pairs(packed, 256, 8, strategy="brute"))
    assert as_set(_search.similar_pairs(packed, 256, 8, strategy="mih")) == brute
    assert {(a, b, d) for (a, b), d in truth.items()} <= brute
    one = np.zeros((1, 8), dtype=np.uint8)
    assert as_set(_search.similar_pairs(one, 64, 5)) == set()
    assert as_set(_search.similar_pairs(np.zeros((0, 8), dtype=np.uint8), 64, 5)) == set()
    with pytest.raises(ValueError):
        _search.similar_pairs(packed, 256, 3, strategy="fast")


def test_distances_and_components():
    words = _search.to_words(np.array([[0] * 8, [255] * 8, [1] + [0] * 7], dtype=np.uint8))
    assert _search.distances_to(words, words[0]).tolist() == [0, 64, 1]
    labels = _search.components(5, np.array([0, 3]), np.array([1, 4]))
    assert labels.tolist() == [0, 0, 2, 3, 3]


def test_cosine_pairs():
    v = np.array([[1, 0], [0.999, 0.02], [0, 1], [-1, 0]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    i, j, s = _search.cosine_pairs(v, 0.99)
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1)] and s[0] > 0.99
    assert _search.cosine_pairs(v[:1], 0.5)[0].size == 0
