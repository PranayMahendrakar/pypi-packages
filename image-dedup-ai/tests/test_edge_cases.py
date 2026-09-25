"""The edge cases the package promises to handle. Each plants what should be found
and checks that clean input around it raises no false alarm."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
from PIL import Image

import image_dedup_ai
from image_dedup_ai import Index
from conftest import file_state, make_photo, tidy


def test_identical_file_under_two_names_groups_together(photo_folder):
    result = image_dedup_ai.find_duplicates(photo_folder["root"])
    original, renamed = tidy(photo_folder["original"]), tidy(photo_folder["renamed"])
    group = next(g for g in result.groups if original in g)
    assert renamed in group
    assert group[0] == original, "same size and resolution: the shorter path is kept"
    member = next(m for d in result.details for m in d if m.path == renamed)
    assert member.identical and member.similarity == 1.0 and not member.kept
    assert (original, renamed, 1.0) in result.pairs or (renamed, original, 1.0) in result.pairs
    assert "Of the copies, 1 is identical in content and 1 is a near-duplicate" in result.summary()


def test_half_size_jpeg_q60_is_still_found_by_phash(photo_folder):
    result = image_dedup_ai.find_duplicates(photo_folder["root"], method="phash", threshold=0.9)
    original, half = tidy(photo_folder["original"]), tidy(photo_folder["half_jpeg"])
    group = next(g for g in result.groups if original in g)
    assert half in group
    member = next(m for d in result.details for m in d if m.path == half)
    assert not member.identical and 0.9 <= member.similarity < 1.0 + 1e-9
    assert (member.width, member.height) == (160, 120)
    assert result.keep[result.groups.index(group)] == original, "the larger resolution is kept"


def test_flipped_or_rotated_copy_is_not_claimed_by_phash(photo_folder):
    result = image_dedup_ai.find_duplicates(photo_folder["root"], method="phash", threshold=0.9)
    grouped = {p for g in result.groups for p in g}
    assert tidy(photo_folder["flipped"]) not in grouped
    assert tidy(photo_folder["rotated"]) not in grouped
    assert any("rotated, flipped" in n for n in result.notes), "the limit is stated in the result"


def test_distinct_photos_raise_no_false_alarm(photo_folder):
    result = image_dedup_ai.find_duplicates(photo_folder["root"])
    grouped = {p for g in result.groups for p in g}
    for key, path in photo_folder.items():
        if key.startswith("distinct_"):
            assert tidy(path) not in grouped
    # exactly one group: original + renamed copy + half-size JPEG
    assert len(result.groups) == 1 and len(result.groups[0]) == 3
    assert result.wasted_bytes == photo_folder["renamed"].stat().st_size + photo_folder["half_jpeg"].stat().st_size


def test_unreadable_and_non_image_files_are_skipped_and_reported(photo_folder):
    result = image_dedup_ai.find_duplicates(photo_folder["root"])
    skipped = dict(result.skipped)
    for key in ("broken", "truncated", "empty"):
        assert tidy(photo_folder[key]) in skipped, key
    assert skipped[tidy(photo_folder["empty"])] == "empty file"
    assert "not an image" in skipped[tidy(photo_folder["broken"])]
    assert tidy(photo_folder["notes"]) not in skipped  # ignored by extension, and counted
    assert any("without an image extension" in n for n in result.notes)
    text = result.summary()
    assert "3 files skipped" in text and "broken.jpg" in text
    # an explicitly named non-image file is tried, then skipped with a reason
    with Index() as idx:
        assert idx.add([photo_folder["notes"], photo_folder["original"]]) == 1
        reasons = dict(idx.last_add.skipped)
        assert "not an image" in reasons[tidy(photo_folder["notes"])]
        assert idx.count() == 1


def test_reopening_the_index_does_not_rehash_unchanged_files(photo_folder, tmp_path, monkeypatch):
    db = tmp_path / "photos.idx"
    with Index(db) as idx:
        first = idx.add(photo_folder["root"])
        assert idx.last_add.hashed == first == idx.count()

    calls = []
    real = image_dedup_ai._hashing.fingerprint_file

    def counting(path, hash_size):
        calls.append(path)
        return real(path, hash_size)

    monkeypatch.setattr(image_dedup_ai._hashing, "fingerprint_file", counting)
    with Index(db) as idx:
        assert idx.add(photo_folder["root"]) == first
        assert calls == [], "unchanged files, including known-broken ones, are not read again"
        assert idx.last_add.hashed == 0 and idx.last_add.unchanged == first
        assert len(idx.last_add.skipped) == 3, "known-broken files are still reported"
        again = idx.find_duplicates()
    assert len(again.groups) == 1

    # an edited file (new size and mtime) is rehashed, and only that file
    edited = photo_folder["distinct_10"]
    make_photo(99).save(edited)
    st = edited.stat()
    os.utime(edited, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with Index(db) as idx:
        idx.add(photo_folder["root"])
        assert calls == [tidy(edited)]
        assert idx.last_add.hashed == 1


def test_deleted_files_are_dropped_on_the_next_scan(photo_folder, tmp_path):
    db = tmp_path / "photos.idx"
    with Index(db) as idx:
        n = idx.add(photo_folder["root"])
    photo_folder["renamed"].unlink()
    with Index(db) as idx:
        assert idx.add(photo_folder["root"]) == n - 1
        assert idx.last_add.removed == 1 and idx.count() == n - 1
        assert all(tidy(photo_folder["renamed"]) not in g for g in idx.find_duplicates().groups)


def test_empty_folder_returns_an_empty_result(tmp_path):
    empty = tmp_path / "nothing here"
    empty.mkdir()
    result = image_dedup_ai.find_duplicates(empty)
    assert result.groups == [] and result.pairs == [] and result.keep == [] and result.drop == []
    assert result.wasted_bytes == 0 and result.n_images == 0 and result.skipped == []
    assert "nothing to compare" in result.summary()
    json.dumps(result.to_dict())
    with Index() as idx:
        assert idx.add(empty) == 0 and idx.count() == 0
        assert idx.near(make_photo(1)) == []
        assert idx.find_duplicates(method="dhash").groups == []


def test_thousand_small_images_index_in_a_few_seconds(tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    rng = np.random.default_rng(7)
    for i in range(1000):
        arr = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").resize((48, 48), Image.NEAREST).save(folder / f"img_{i:04d}.png")
    # plant five exact copies among the thousand
    for i in range(5):
        (folder / f"img_{i * 100:04d} - Copy.png").write_bytes((folder / f"img_{i * 100:04d}.png").read_bytes())
    start = time.perf_counter()
    with Index(tmp_path / "many.idx") as idx:
        assert idx.add(folder) == 1005
        result = idx.find_duplicates(threshold=0.95)
    elapsed = time.perf_counter() - start
    assert elapsed < 6.0, f"indexing 1000 small images took {elapsed:.1f}s"
    assert sorted(len(g) for g in result.groups) == [2, 2, 2, 2, 2]
    assert all(os.path.basename(g[0]).startswith("img_") for g in result.groups)
    start = time.perf_counter()
    with Index(tmp_path / "many.idx") as idx:
        idx.add(folder)
        assert idx.last_add.unchanged == 1005
    assert time.perf_counter() - start < 3.0, "a re-check of unchanged files should be near-instant"


def test_caller_files_are_never_modified(photo_folder, tmp_path):
    before = file_state(photo_folder["root"])
    with Index(tmp_path / "x.idx") as idx:
        idx.add(photo_folder["root"])
        idx.find_duplicates()
        idx.near(photo_folder["original"])
        idx.remove(photo_folder["original"])
    image_dedup_ai.find_duplicates(photo_folder["root"], method="dhash")
    assert file_state(photo_folder["root"]) == before


def test_caller_pil_images_are_never_modified():
    rgba = make_photo(3).convert("RGBA")
    rgba.putpixel((0, 0), (1, 2, 3, 0))
    palette = make_photo(4).convert("P")
    grey16 = Image.fromarray(np.asarray(make_photo(5).convert("L"), dtype=np.uint16) * 257)
    images = [make_photo(1), rgba, palette, grey16]
    snapshot = [(im.mode, im.size, im.tobytes(), dict(im.info)) for im in images]
    with Index() as idx:
        idx.add(images)
        idx.find_duplicates()
        idx.near(images[0])
    image_dedup_ai.find_duplicates({"a": images[0], "b": images[1]}, embed=lambda im: np.asarray(im.convert("L").resize((4, 4)), float).ravel())
    assert [(im.mode, im.size, im.tobytes(), dict(im.info)) for im in images] == snapshot


def test_index_that_is_not_ours_is_left_untouched(tmp_path):
    photo = tmp_path / "photo.jpg"
    make_photo(1).save(photo)
    data = photo.read_bytes()
    with pytest.raises(ValueError, match="not an image-dedup-ai index"):
        Index(photo)
    assert photo.read_bytes() == data

    import sqlite3

    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE customers (name TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="not an image-dedup-ai index"):
        Index(other)
    conn = sqlite3.connect(other)
    assert [r[0] for r in conn.execute("SELECT name FROM sqlite_master")] == ["customers"]
    conn.close()


def test_a_folder_is_not_an_index_path(tmp_path):
    with pytest.raises(ValueError, match="is a folder"):
        Index(tmp_path)


def test_undecodable_file_name_is_reported_not_raised(tmp_path):
    folder = tmp_path / "names"
    folder.mkdir()
    make_photo(1).save(folder / "fine.png")
    bad = os.path.join(str(folder), "bad\udcff.png")
    try:
        with open(bad, "wb") as fh:
            fh.write((folder / "fine.png").read_bytes())
    except (OSError, UnicodeError, ValueError):
        pytest.skip("this file system cannot hold a non-Unicode file name")
    result = image_dedup_ai.find_duplicates(folder)
    assert result.n_images == 1
    (entry,) = result.skipped
    assert "not valid Unicode" in entry[1]
    json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")
    result.summary().encode("utf-8")
