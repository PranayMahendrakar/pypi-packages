"""The embed hook: how to get past what a perceptual hash cannot see."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import image_dedup_ai
from image_dedup_ai import Index
from conftest import make_photo, tidy

CALLS = []


def colour_histogram(img: Image.Image) -> np.ndarray:
    """A toy embedding that is blind to flips and rotations (a real one would be a model)."""
    CALLS.append(img.size)
    assert img.mode == "RGB"
    small = np.asarray(img.resize((64, 64)), dtype=np.float64)
    return np.concatenate([np.histogram(small[..., c], bins=16, range=(0, 256))[0] for c in range(3)])


def other_embedding(img: Image.Image) -> np.ndarray:
    return colour_histogram(img)[::-1].copy()


@pytest.fixture
def mirrored(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    photo = make_photo(1)
    photo.save(folder / "original.png")
    photo.transpose(Image.FLIP_LEFT_RIGHT).save(folder / "mirrored.png")
    photo.rotate(90, expand=True).save(folder / "rotated.png")
    for s in range(10, 16):
        make_photo(s).save(folder / f"distinct_{s}.png")
    return folder


def test_embed_catches_flips_and_rotations_that_phash_misses(mirrored):
    by_hash = image_dedup_ai.find_duplicates(mirrored)
    assert by_hash.groups == []
    result = image_dedup_ai.find_duplicates(mirrored, embed=colour_histogram, method="embed", threshold=0.99)
    assert [sorted(os.path.basename(p) for p in g) for g in result.groups] == [["mirrored.png", "original.png", "rotated.png"]]
    assert result.method == "embed" and result.bits == 0
    assert "embed" in result.summary()
    assert all(0.99 <= m.similarity <= 1.0 for m in result.details[0])


def test_embeddings_persist_and_are_not_recomputed(mirrored, tmp_path):
    db = tmp_path / "e.idx"
    CALLS.clear()
    with Index(db, embed=colour_histogram) as idx:
        n = idx.add(mirrored)
        assert idx.last_add.embedded == n == len(CALLS) == 9
        assert CALLS[0] == (320, 240), "embed receives the full-size upright image"
    CALLS.clear()
    with Index(db, embed=colour_histogram) as idx:
        idx.add(mirrored)
        assert CALLS == [] and idx.last_add.embedded == 0
        assert len(idx.find_duplicates(method="embed", threshold=0.99).groups) == 1
    with Index(db) as idx:  # stored vectors are usable without the function
        assert len(idx.find_duplicates(method="embed", threshold=0.99).groups) == 1
        with pytest.raises(ValueError, match="embed"):
            idx.near(make_photo(1), method="embed")
        (best,) = idx.near(mirrored / "original.png", method="embed", k=1)
        assert os.path.basename(best.path) in ("mirrored.png", "rotated.png") and best.similarity > 0.99


def test_adding_embed_later_embeds_existing_images_without_rehashing(mirrored, tmp_path):
    db = tmp_path / "e.idx"
    with Index(db) as idx:
        idx.add(mirrored)
        with pytest.raises(ValueError, match="no embeddings"):
            idx.find_duplicates(method="embed")
    with Index(db, embed=colour_histogram) as idx:
        assert idx.add(mirrored) == 9
        assert idx.last_add.hashed == 0 and idx.last_add.embedded == 9
        near = idx.near(make_photo(1).transpose(Image.FLIP_TOP_BOTTOM), method="embed", k=3)
        assert {os.path.basename(m.path) for m in near} == {"original.png", "mirrored.png", "rotated.png"}


def test_changing_the_embed_function_drops_stale_vectors(mirrored, tmp_path):
    db = tmp_path / "e.idx"
    with Index(db, embed=colour_histogram) as idx:
        idx.add(mirrored)
    with Index(db, embed=other_embedding) as idx:
        idx.add(mirrored)
        assert idx.last_add.embedded == 9


def test_partial_embeddings_are_noted(mirrored, tmp_path):
    with Index(embed=colour_histogram) as idx:
        idx.add(mirrored / "original.png")
        idx.embed = None
        idx.add(mirrored / "mirrored.png")
        result = idx.find_duplicates(method="embed")
        assert any("no embedding" in n for n in result.notes)


def test_a_failing_embed_skips_the_image_and_says_why(mirrored):
    def picky(img):
        if img.size[0] > img.size[1]:
            raise RuntimeError("model only takes portrait images")
        return colour_histogram(img)

    with Index(embed=picky) as idx:
        assert idx.add(mirrored) == 1  # only the rotated (portrait) copy
        reasons = dict(idx.last_add.skipped)
        assert "model only takes portrait" in reasons[tidy(mirrored / "original.png")]
        result = idx.find_duplicates(method="embed")
        assert tidy(mirrored / "original.png") in dict(result.skipped)

    with Index(embed=lambda img: [float("nan")] * 3) as idx:
        assert idx.add({"a": make_photo(1)}) == 0
        assert "NaN" in idx.last_add.skipped[0][1]
    with Index(embed=lambda img: []) as idx:
        assert idx.add(mirrored / "original.png") == 0
        assert "empty" in idx.last_add.skipped[0][1]


def test_near_by_embedding_needs_embeddings(mirrored):
    with Index() as idx:
        idx.add(mirrored / "original.png")
        with pytest.raises(ValueError, match="no embeddings"):
            idx.near(mirrored / "original.png", method="embed")
    with Index(embed=colour_histogram) as idx:
        assert idx.near(make_photo(1), method="embed") == []
