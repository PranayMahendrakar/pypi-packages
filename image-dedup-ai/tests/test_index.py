"""The public API: Index, find_duplicates, DedupeResult, Match."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

import image_dedup_ai
from image_dedup_ai import DedupeResult, Index, Match
from conftest import make_photo, tidy


def write_set(folder, seeds=(1, 2, 3)):
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in seeds:
        p = folder / f"photo_{s}.png"
        make_photo(s).save(p)
        out[s] = p
    return out


def test_index_lifecycle(tmp_path):
    db = tmp_path / "nested" / "photos.idx"
    idx = Index(db)
    assert os.path.exists(db) and idx.count() == 0 and len(idx) == 0
    assert "0 images" in repr(idx)
    paths = write_set(tmp_path / "a")
    assert idx.add(tmp_path / "a") == 3 and idx.count() == 3
    idx.close()
    idx.close()  # twice is fine
    assert "closed" in repr(idx)
    with pytest.raises(ValueError, match="closed"):
        idx.count()
    with Index(db) as again:
        assert again.count() == 3
        assert again.remove(paths[1]) == 1 and again.count() == 2
        assert again.remove(paths[1]) == 0
        assert again.remove(tmp_path / "a") == 2 and again.count() == 0


def test_hash_size_is_fixed_per_index_file(tmp_path):
    db = tmp_path / "p.idx"
    with Index(db, hash_size=16) as idx:
        idx.add(write_set(tmp_path / "a", seeds=(1,))[1])
    with pytest.raises(ValueError, match="hash_size=16"):
        Index(db)
    with Index(db, hash_size=16) as idx:
        assert idx.count() == 1 and idx.bits == 256
    with pytest.raises(ValueError):
        Index(hash_size=2)


def test_longer_hash_still_finds_the_resized_copy(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2, 3, 4))
    make_photo(1).resize((200, 150)).save(folder / "small.jpg", quality=70)
    result = image_dedup_ai.find_duplicates(folder, hash_size=16, threshold=0.88)
    assert result.bits == 256
    assert [sorted(os.path.basename(p) for p in g) for g in result.groups] == [["photo_1.png", "small.jpg"]]


def test_pil_images_by_name_and_unnamed(tmp_path):
    photo = make_photo(1)
    with Index(tmp_path / "mem.idx") as idx:
        assert idx.add({"cat.png": photo, "cat_small": photo.resize((160, 120))}) == 2
        assert idx.add(make_photo(2)) == 1
        assert idx.add([make_photo(3), make_photo(4)]) == 2
        result = idx.find_duplicates()
    assert result.groups == [["cat.png", "cat_small"]]
    assert result.wasted_bytes == 0
    assert "freeing" not in result.summary(), "in-memory images have no file size to free"
    with Index(tmp_path / "mem.idx") as idx:
        idx.add(make_photo(5))
        names = [m.path for m in idx.near(make_photo(5), k=10)]
    assert {"<image 1>", "<image 2>", "<image 3>"} <= set(names)
    assert "<image 4>" in names, "numbering continues after reopening"
    with Index(tmp_path / "mem.idx") as idx:
        assert idx.remove("<image 2>") == 1


def test_find_duplicates_validates_its_arguments(tmp_path):
    with Index() as idx:
        for bad in (0, -0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                idx.find_duplicates(threshold=bad)
        with pytest.raises(TypeError):
            idx.find_duplicates(threshold="0.9")
        with pytest.raises(ValueError, match="method"):
            idx.find_duplicates(method="sha1")
        with pytest.raises(ValueError):
            idx.near(make_photo(1), k=0)
        with pytest.raises(ValueError):
            idx.add(tmp_path, workers=0)
        with pytest.raises(TypeError):
            idx.add(42)
        with pytest.raises(TypeError):
            idx.add([tmp_path, 3.5])
        with pytest.raises(TypeError):
            idx.add({"a": "not an image"})
        with pytest.raises(FileNotFoundError):
            idx.add(tmp_path / "missing")
        with pytest.raises(TypeError):
            Index(embed="not callable")


def test_missing_path_inside_a_list_is_reported_not_raised(tmp_path):
    paths = write_set(tmp_path / "a", seeds=(1,))
    with Index() as idx:
        assert idx.add([paths[1], tmp_path / "gone.jpg"]) == 1
        assert dict(idx.last_add.skipped)[tidy(tmp_path / "gone.jpg")] == "no such file or folder"


def test_threshold_one_means_identical_hashes_only(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2))
    shutil.copyfile(folder / "photo_1.png", folder / "photo_1 copy.png")
    make_photo(2).resize((150, 110)).save(folder / "photo_2 small.jpg", quality=40)
    strict = image_dedup_ai.find_duplicates(folder, threshold=1.0)
    names = [[os.path.basename(p) for p in g] for g in strict.groups]
    assert ["photo_1.png", "photo_1 copy.png"] in names
    loose = image_dedup_ai.find_duplicates(folder, threshold=0.85)
    assert len(loose.groups) == 2


def test_keep_prefers_resolution_then_file_size(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    photo = make_photo(7)
    photo.save(folder / "a_big.jpg", quality=50)
    photo.save(folder / "b_same_size_better_quality.jpg", quality=95)
    photo.resize((160, 120)).save(folder / "c_small_but_huge.png")
    result = image_dedup_ai.find_duplicates(folder)
    assert len(result.groups) == 1
    assert os.path.basename(result.keep[0]) == "b_same_size_better_quality.jpg"
    assert set(result.drop) == {tidy(folder / "a_big.jpg"), tidy(folder / "c_small_but_huge.png")}
    assert result.wasted_bytes == (folder / "a_big.jpg").stat().st_size + (folder / "c_small_but_huge.png").stat().st_size


def test_near_ranks_neighbours_and_leaves_out_the_query(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2, 3, 4))
    make_photo(1).resize((160, 120)).save(folder / "small.jpg", quality=60)
    with Index() as idx:
        idx.add(folder)
        matches = idx.near(folder / "photo_1.png", k=3)
        assert len(matches) == 3 and all(isinstance(m, Match) for m in matches)
        assert matches[0].path == tidy(folder / "small.jpg") and matches[0].similarity >= 0.9
        assert tidy(folder / "photo_1.png") not in [m.path for m in matches]
        assert [m.similarity for m in matches] == sorted((m.similarity for m in matches), reverse=True)
        path, sim = matches[1]
        assert sim < 0.9
        outside = tmp_path / "outside.png"
        make_photo(3).save(outside)
        assert idx.near(outside, k=1)[0].path == tidy(folder / "photo_3.png")
        assert idx.count() == 5, "a query is not added"
        assert idx.near(make_photo(2), k=1, method="dhash")[0].path == tidy(folder / "photo_2.png")
        assert len(idx.near(make_photo(2), k=50)) == 5
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"junk")
        with pytest.raises(ValueError, match="cannot read"):
            idx.near(broken)
        with pytest.raises(FileNotFoundError):
            idx.near(tmp_path / "nope.png")
        with pytest.raises(TypeError):
            idx.near(12)


def test_blank_images_only_match_identical_blanks(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    Image.new("RGB", (80, 60), (0, 0, 0)).save(folder / "black.png")
    Image.new("RGB", (80, 60), (255, 255, 255)).save(folder / "white.png")
    Image.new("RGB", (80, 60), (128, 128, 128)).save(folder / "grey.png")
    shutil.copyfile(folder / "black.png", folder / "black copy.png")
    make_photo(1).save(folder / "photo.png")
    result = image_dedup_ai.find_duplicates(folder, method="dhash", threshold=0.5)
    assert [[os.path.basename(p) for p in g] for g in result.groups] == [["black.png", "black copy.png"]]
    assert any("near-blank" in n for n in result.notes)
    with Index() as idx:
        idx.add(folder)
        blank = idx.near(folder / "black.png", k=5)
        assert blank[0] == Match(tidy(folder / "black copy.png"), 1.0)
        assert all(m.similarity == 0.0 for m in blank[1:]), "other blanks and the photo are not similar"
        assert idx.near(folder / "photo.png", k=5)[-1].similarity == 0.0
        # a hash says nothing about a blank image, so a blank query is never "similar" by hash alone
        assert all(m.similarity == 0.0 for m in idx.near(Image.new("RGB", (80, 60), (9, 9, 9)), k=5))


def test_dhash_and_ahash_find_the_resized_copy(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2, 3))
    make_photo(2).resize((160, 120)).save(folder / "small.jpg", quality=60)
    for method in ("dhash", "ahash"):
        # the weaker hashes need a stricter threshold to stay clear of look-alikes
        result = image_dedup_ai.find_duplicates(folder, method=method, threshold=0.95)
        grouped = [sorted(os.path.basename(p) for p in g) for g in result.groups]
        assert grouped == [["photo_2.png", "small.jpg"]], method
        assert result.method == method


def test_dhash_is_weaker_than_phash_on_smooth_images(tmp_path):
    """photo_2 and photo_3 share a dominant brightness ramp: dhash at 0.9 calls them
    duplicates, phash does not. This is why phash is the default."""
    folder = tmp_path / "p"
    write_set(folder, seeds=(2, 3))
    assert len(image_dedup_ai.find_duplicates(folder, method="dhash", threshold=0.9).groups) == 1
    assert image_dedup_ai.find_duplicates(folder, method="phash", threshold=0.9).groups == []


def test_non_recursive_scan_stays_in_the_folder(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1,))
    write_set(folder / "deeper", seeds=(2,))
    with Index() as idx:
        assert idx.add(folder, recursive=False) == 1
        assert idx.add(folder) == 2


def test_workers_do_not_change_the_result(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=range(1, 13))
    make_photo(5).resize((100, 75)).save(folder / "tiny5.jpg", quality=60)
    one = image_dedup_ai.find_duplicates(folder, workers=1)
    many = image_dedup_ai.find_duplicates(folder, workers=4)
    assert one.to_dict() == many.to_dict() and len(one.groups) == 1


def test_result_explains_itself(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=range(1, 15))
    for s in range(1, 13):
        shutil.copyfile(folder / f"photo_{s}.png", folder / f"photo_{s} (1).png")
    (folder / "junk.gif").write_bytes(b"GIF89a")
    result = image_dedup_ai.find_duplicates(folder)
    assert isinstance(result, DedupeResult)
    assert result.n_groups == 12 and result.n_duplicates == 12 and len(result.keep) == 12
    assert all(os.path.basename(k).count("(1)") == 0 for k in result.keep)
    text = result.summary()
    assert "12 duplicate groups" in text and "... and 2 more groups" in text and "identical content" in text
    assert "1 file skipped" in text and "junk.gif" in text
    assert text.isascii(), "summary text is plain ASCII when the paths are"
    d = result.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["n_groups"] == 12 and d["wasted_bytes"] == result.wasted_bytes and d["drop"] == result.drop
    assert d["details"][0][0]["kept"] and not d["details"][0][1]["kept"]
    assert "DedupeResult(" in repr(result) and "groups=12" in repr(result)
    short = result.summary(max_groups=1)
    assert "... and 11 more groups" in short


def test_pairs_link_every_member_of_every_group(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2, 3))
    shutil.copyfile(folder / "photo_1.png", folder / "photo_1 - Copy.png")
    make_photo(1).resize((160, 120)).save(folder / "photo_1 small.jpg", quality=60)
    result = image_dedup_ai.find_duplicates(folder)
    (group,) = result.groups
    linked = {p for a, b, _ in result.pairs for p in (a, b)}
    assert linked == set(group)
    for a, b, s in result.pairs:
        assert a < b and 0.9 <= s <= 1.0
    assert (tidy(folder / "photo_1 - Copy.png"), tidy(folder / "photo_1.png"), 1.0) in result.pairs


def test_unicode_file_names(tmp_path):
    folder = tmp_path / "фото 写真"
    write_set(folder, seeds=(1, 2))
    shutil.copyfile(folder / "photo_1.png", folder / "café ☕ copie.png")
    result = image_dedup_ai.find_duplicates(folder)
    assert tidy(folder / "café ☕ copie.png") in result.groups[0]
    assert "café ☕ copie.png" in result.summary()
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_loose_threshold_is_flagged(tmp_path):
    write_set(tmp_path / "p", seeds=(1, 2))
    result = image_dedup_ai.find_duplicates(tmp_path / "p", threshold=0.5)
    assert any("loose" in n for n in result.notes)


def test_hash_image_matches_index_similarity(tmp_path):
    photo = make_photo(9)
    small = photo.resize((160, 120))
    with Index() as idx:
        idx.add({"a": photo})
        (match,) = idx.near(small, k=1)
    ha = np.unpackbits(np.frombuffer(bytes.fromhex(image_dedup_ai.hash_image(photo)), np.uint8))
    hb = np.unpackbits(np.frombuffer(bytes.fromhex(image_dedup_ai.hash_image(small)), np.uint8))
    assert match.similarity == pytest.approx(1 - (ha ^ hb).sum() / 64)


def test_add_report(tmp_path):
    folder = tmp_path / "p"
    write_set(folder, seeds=(1, 2))
    (folder / "readme.md").write_text("x", encoding="utf-8")
    (folder / "bad.png").write_bytes(b"x")
    with Index() as idx:
        assert idx.add(folder) == 2
        report = idx.last_add
    assert (report.hashed, report.unchanged, report.indexed, len(report.skipped), len(report.ignored)) == (2, 0, 2, 1, 1)
    text = report.summary()
    assert text.startswith("indexed 2 images: 2 hashed, 0 unchanged") and "1 non-image file ignored" in text
    assert json.loads(json.dumps(report.to_dict()))["skipped"][0]["reason"] == "not an image file Pillow can read"


def test_single_image_and_mixed_inputs(tmp_path):
    folder = tmp_path / "p"
    paths = write_set(folder, seeds=(1, 2))
    lone = tmp_path / "lone.png"
    make_photo(3).save(lone)
    single = image_dedup_ai.find_duplicates(lone)
    assert single.n_images == 1 and single.groups == [] and "No duplicates found." in single.summary()
    with Index() as idx:
        n = idx.add([folder, str(lone), make_photo(1).resize((200, 150)), paths[2]])
        assert n == 4, "folder (2) + file + PIL image; the repeated path is counted once"
        result = idx.find_duplicates()
    assert [sorted(os.path.basename(p) for p in g) for g in result.groups] == [["<image 1>", "photo_1.png"]]
    assert result.keep == [tidy(paths[1])]
