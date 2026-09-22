from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import near_dupes  # noqa: E402
from near_dupes import DuplicateFinder, find_duplicates  # noqa: E402
from near_dupes import _images  # noqa: E402


def make_images():
    y, x = np.mgrid[0:64, 0:64]
    base = (x * 3 + 20).astype(np.uint8)
    base[20:40, 10:30] = 5  # a dark rectangle on a horizontal gradient
    brighter = np.clip(base.astype(int) + 30, 0, 255).astype(np.uint8)
    other = (y * 3 + 20).astype(np.uint8)  # vertical gradient: very different
    noise = np.random.default_rng(0).integers(0, 255, size=(64, 64), dtype=np.uint8)
    img_a = Image.fromarray(base).convert("RGB")
    img_b = Image.fromarray(brighter).convert("RGB")
    img_c = img_a.resize((120, 90), Image.BICUBIC)  # resized copy
    img_d = Image.fromarray(other).convert("RGB")
    img_e = Image.fromarray(noise).convert("RGB")
    return [img_a, img_b, img_c, img_d, img_e]


def test_dhash_basics():
    a, b, c, d, e = make_images()
    ha, hb, hc, hd = (_images.dhash(im) for im in (a, b, c, d))
    assert 0 <= ha < 2**64
    assert ha == hb  # brightness shift keeps the gradient direction bits
    assert _images.hamming_similarity(ha, hc) >= 0.85
    assert _images.hamming_similarity(ha, hd) < 0.7
    assert _images.dhash(np.asarray(a)) == ha


def test_pil_images_in_memory():
    imgs = make_images()
    r = find_duplicates(imgs, threshold=0.85)
    assert r.kind == "images"
    assert r.groups == [[0, 1, 2]]
    assert r.representatives == [0]
    kept = r.dedupe(imgs)
    assert kept == [imgs[0], imgs[3], imgs[4]]
    scores = {(i, j): s for i, j, s in r.pairs}
    assert scores[(0, 1)] == 1.0


def test_image_paths_auto_detected(tmp_path):
    imgs = make_images()
    paths = []
    for k, im in enumerate(imgs):
        p = tmp_path / f"img{k}.png"
        im.save(p)
        paths.append(str(p))
    r = find_duplicates(paths)
    assert r.kind == "images" and r.groups == [[0, 1, 2]]
    assert r.dedupe(paths) == [paths[0], paths[3], paths[4]]
    as_path_objects = [Path(p) for p in paths]
    assert find_duplicates(as_path_objects).groups == [[0, 1, 2]]
    assert near_dupes.dedupe(paths) == [paths[0], paths[3], paths[4]]


def test_threshold_one_only_identical_hashes():
    imgs = make_images()
    r = find_duplicates(imgs, threshold=1.0)
    assert r.groups == [[0, 1]]


def test_explicit_kind_and_empty():
    assert find_duplicates([], kind="images").n_items == 0
    r = DuplicateFinder(kind="images").find(make_images()[:1])
    assert r.groups == [] and r.keep_indices == [0]


def test_hamming_pairs_banding_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 600
    hashes = rng.integers(0, 2**63, size=n, dtype=np.int64).astype(np.uint64)
    for k in range(0, n, 20):  # plant close pairs a few bits apart
        flips = rng.integers(0, 64, size=int(rng.integers(0, 6)))
        v = int(hashes[k])
        for f in flips:
            v ^= 1 << int(f)
        hashes[k + 1] = v
    for max_dist in (0, 3, 9, 12):
        brute, bd = _images.hamming_pairs(hashes, max_dist, brute_force_max=10_000)
        banded, nd = _images.hamming_pairs(hashes, max_dist, brute_force_max=0)
        assert brute.tolist() == banded.tolist()
        assert bd.tolist() == nd.tolist()
        assert all(d <= max_dist for d in bd.tolist())
    empty, _ = _images.hamming_pairs(hashes[:1], 5)
    assert empty.shape == (0, 2)


def test_bad_image_type():
    with pytest.raises(TypeError):
        _images.dhash(12345)
