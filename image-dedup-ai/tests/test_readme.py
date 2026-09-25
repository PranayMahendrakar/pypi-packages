"""The README quickstart block, run exactly as written, and the claims the README makes."""
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import ImageEnhance

import image_dedup_ai
from conftest import make_photo

README = Path(__file__).resolve().parents[1] / "README.md"


def quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1, "the Quickstart section holds exactly one code block"
    return blocks[0]


def test_readme_sections_in_order():
    text = README.read_text(encoding="utf-8")
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert heads == ["Install", "Quickstart", "What it does", "API", "CLI", "License"]
    first_paragraph = text.split("\n\n")[1]
    assert "heuristic" in first_paragraph and "rotated" in first_paragraph and "flipped" in first_paragraph


def test_readme_quickstart_runs(capsys):
    code = quickstart_block()
    assert 3 <= len([line for line in code.splitlines() if line.strip()]) <= 6
    namespace = {}
    exec(compile(code, "README-quickstart", "exec"), namespace)
    out = capsys.readouterr().out
    assert "Compared 3 images" in out and "1 duplicate group" in out
    assert "keep photo (512x512)" in out and "copy photo_half_size (256x256)" in out
    result = namespace["result"]
    assert isinstance(result, image_dedup_ai.DedupeResult)
    assert result.groups == [["photo", "photo_half_size"]] and result.keep == ["photo"]


def sim(a, b):
    x = np.unpackbits(np.frombuffer(bytes.fromhex(a), np.uint8)) ^ np.unpackbits(np.frombuffer(bytes.fromhex(b), np.uint8))
    return 1 - x.sum() / 64


def test_claim_small_trims_survive_and_real_crops_do_not():
    for seed in range(4):
        photo = make_photo(seed)
        w, h = photo.size
        base = image_dedup_ai.hash_image(photo)

        def crop(f):
            return image_dedup_ai.hash_image(photo.crop((int(w * f), int(h * f), int(w * (1 - f)), int(h * (1 - f)))))

        assert sim(base, crop(0.02)) >= 0.9
        assert sim(base, crop(0.10)) < 0.9
        assert sim(base, image_dedup_ai.hash_image(ImageEnhance.Contrast(photo).enhance(0.8))) >= 0.9


def test_claim_grouping_a_large_index_is_fast(tmp_path):
    """20,000 synthetic entries (hashes written straight into the index) with 150 planted copies."""
    db = tmp_path / "big.idx"
    image_dedup_ai.Index(db).close()
    rng = np.random.default_rng(0)
    n = 20000
    hashes = rng.integers(0, 256, (n, 8), dtype=np.uint8)
    for k in range(150):
        bits = np.unpackbits(hashes[2 * k])
        bits[rng.choice(64, k % 7, replace=False)] ^= 1  # 0-6 bits apart: all within 0.9
        hashes[2 * k + 1] = np.packbits(bits)
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (f"/fake/img_{i:06d}.jpg", 1000, 1, 64, 64, "JPEG", f"{i:032x}", 0, 128.0,
             hashes[i].tobytes(), hashes[i].tobytes(), hashes[i].tobytes(), None)
            for i in range(n)
        ],
    )
    conn.commit()
    conn.close()
    start = time.perf_counter()
    with image_dedup_ai.Index(db) as idx:
        result = idx.find_duplicates()
    assert time.perf_counter() - start < 5.0
    assert result.n_groups == 150 and all(len(g) == 2 for g in result.groups)
    expected = {(f"/fake/img_{2 * k:06d}.jpg", f"/fake/img_{2 * k + 1:06d}.jpg") for k in range(150)}
    assert {tuple(sorted(g)) for g in result.groups} == expected
