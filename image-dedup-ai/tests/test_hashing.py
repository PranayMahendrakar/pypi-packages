"""The hashes themselves: DCT, robustness, honest limits, awkward image modes."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import image_dedup_ai
from image_dedup_ai import _hashing
from conftest import make_photo


def similarity(a: str, b: str) -> float:
    x = np.unpackbits(np.frombuffer(bytes.fromhex(a), np.uint8)) ^ np.unpackbits(np.frombuffer(bytes.fromhex(b), np.uint8))
    return 1.0 - x.sum() / x.size


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def test_dct_matrix_is_orthonormal_and_matches_the_definition():
    for n in (4, 8, 32):
        c = _hashing.dct_matrix(n)
        assert np.allclose(c @ c.T, np.eye(n), atol=1e-12)
    x = np.random.default_rng(0).normal(size=(8, 8))
    naive = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            au = np.sqrt((1 if u == 0 else 2) / 8)
            av = np.sqrt((1 if v == 0 else 2) / 8)
            total = 0.0
            for i in range(8):
                for j in range(8):
                    total += x[i, j] * np.cos(np.pi * (2 * i + 1) * u / 16) * np.cos(np.pi * (2 * j + 1) * v / 16)
            naive[u, v] = au * av * total
    assert np.allclose(_hashing.dct2(x), naive)
    flat = _hashing.dct2(np.full((16, 16), 3.0))
    assert flat[0, 0] == pytest.approx(48.0) and np.abs(flat.ravel()[1:]).max() < 1e-9


def test_hash_image_shapes_and_determinism(tmp_path):
    photo = make_photo(1)
    for method in image_dedup_ai.METHODS:
        h = image_dedup_ai.hash_image(photo, method)
        assert len(h) == 16 and h == image_dedup_ai.hash_image(photo, method)
    assert len(image_dedup_ai.hash_image(photo, hash_size=16)) == 64
    path = tmp_path / "p.png"
    photo.save(path)
    assert image_dedup_ai.hash_image(path) == image_dedup_ai.hash_image(photo)
    with pytest.raises(ValueError):
        image_dedup_ai.hash_image(photo, "md5")
    with pytest.raises(ValueError):
        image_dedup_ai.hash_image(photo, hash_size=3)
    with pytest.raises(TypeError):
        image_dedup_ai.hash_image(photo, hash_size="8")


@pytest.mark.parametrize("seed", range(6))
def test_phash_survives_half_size_and_jpeg_q60(seed):
    photo = make_photo(seed)
    small = jpeg(photo.resize((160, 120), Image.LANCZOS), 60)
    assert similarity(image_dedup_ai.hash_image(photo), image_dedup_ai.hash_image(small)) >= 0.9


@pytest.mark.parametrize("seed", range(6))
def test_phash_does_not_claim_flips_or_rotations(seed):
    photo = make_photo(seed)
    base = image_dedup_ai.hash_image(photo)
    for variant in (
        photo.transpose(Image.FLIP_LEFT_RIGHT),
        photo.transpose(Image.FLIP_TOP_BOTTOM),
        photo.rotate(90, expand=True),
        photo.rotate(180),
    ):
        assert similarity(base, image_dedup_ai.hash_image(variant)) < 0.9


def test_distinct_photos_are_far_apart_under_phash():
    hashes = [image_dedup_ai.hash_image(make_photo(s)) for s in range(20, 40)]
    worst = max(similarity(hashes[i], hashes[j]) for i in range(20) for j in range(i + 1, 20))
    assert worst < 0.85


def test_exif_orientation_is_applied(tmp_path):
    upright = make_photo(3)
    plain = tmp_path / "upright.jpg"
    upright.save(plain, quality=95)
    stored = upright.transpose(Image.ROTATE_90)  # what a sideways camera writes
    exif = Image.Exif()
    exif[0x0112] = 6
    tagged = tmp_path / "tagged.jpg"
    stored.save(tagged, quality=95, exif=exif.tobytes())
    a = _hashing.fingerprint_file(str(plain), 8)
    b = _hashing.fingerprint_file(str(tagged), 8)
    assert (b.width, b.height) == (320, 240)
    assert similarity(a.hashes["phash"].hex(), b.hashes["phash"].hex()) >= 0.95


def test_sixteen_bit_greyscale_is_scaled_not_clipped():
    grey8 = make_photo(4).convert("L")
    grey16 = Image.fromarray(np.asarray(grey8, dtype=np.uint16) * 257)
    assert grey16.mode.startswith("I")
    fp16 = _hashing.fingerprint_image(grey16, 8)
    fp8 = _hashing.fingerprint_image(grey8, 8)
    assert not fp16.flat
    assert abs(fp16.tone - fp8.tone) < 1.0
    assert similarity(fp16.hashes["phash"].hex(), fp8.hashes["phash"].hex()) >= 0.95
    as_int = Image.fromarray(np.asarray(grey8, dtype=np.int32) * 257)
    assert as_int.mode == "I"
    assert not _hashing.fingerprint_image(as_int, 8).flat
    floats = Image.fromarray(np.asarray(grey8, dtype=np.float32) / 255.0 * 1000.0 - 200.0)
    assert floats.mode == "F" and not _hashing.fingerprint_image(floats, 8).flat


def test_transparency_is_flattened_onto_white():
    photo = make_photo(5)
    rgba = photo.convert("RGBA")
    arr = np.asarray(rgba).copy()
    arr[:60, :, 3] = 0
    arr[:60, :, :3] = np.random.default_rng(0).integers(0, 256, (60, arr.shape[1], 3))  # garbage under alpha 0
    garbage = Image.fromarray(arr, "RGBA")
    arr2 = arr.copy()
    arr2[:60, :, :3] = 255
    white = Image.fromarray(arr2, "RGBA").convert("RGB")
    assert image_dedup_ai.hash_image(garbage) == image_dedup_ai.hash_image(white)


def test_blank_images_are_flat_and_photos_are_not():
    assert _hashing.fingerprint_image(Image.new("L", (40, 40), 0), 8).flat
    assert _hashing.fingerprint_image(Image.new("RGB", (40, 40), (250, 250, 250)), 8).flat
    assert not _hashing.fingerprint_image(make_photo(1), 8).flat
    assert not _hashing.fingerprint_image(Image.linear_gradient("L"), 8).flat


def test_unreadable_bytes_raise_value_error_with_a_reason():
    with pytest.raises(ValueError, match="empty file"):
        _hashing.fingerprint_bytes(b"", 8)
    with pytest.raises(ValueError, match="not an image"):
        _hashing.fingerprint_bytes(b"GIF89a but not really" * 3, 8)
    buf = io.BytesIO()
    make_photo(2).save(buf, "PNG")
    with pytest.raises(ValueError, match="unreadable image"):
        _hashing.fingerprint_bytes(buf.getvalue()[:2000], 8)
    with pytest.raises(TypeError):
        _hashing.fingerprint_image("not an image", 8)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _hashing.fingerprint_image(Image.new("RGB", (0, 5)), 8)


def test_palette_and_cmyk_and_bilevel_images_hash():
    photo = make_photo(6)
    base = image_dedup_ai.hash_image(photo)
    for variant in (photo.convert("P"), photo.convert("CMYK"), photo.convert("L"), photo.convert("LA")):
        assert similarity(base, image_dedup_ai.hash_image(variant)) >= 0.9, variant.mode
    assert len(image_dedup_ai.hash_image(photo.convert("1"))) == 16
    p1, p2 = photo.convert("P"), photo.convert("P")
    p2.putpalette([255 - v for v in p1.getpalette()])
    assert _hashing.pixel_digest(p1) != _hashing.pixel_digest(p2), "palette is part of the content"
