"""Shared test helpers. Every image is generated here; nothing is downloaded."""
from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from PIL import Image


def make_photo(seed: int, size: Tuple[int, int] = (320, 240)) -> Image.Image:
    """A deterministic photo-like RGB image: soft coloured blobs, a colour ramp and grain."""
    return Image.fromarray(_photo_pixels(seed, size).copy(), "RGB")


@lru_cache(maxsize=None)
def _photo_pixels(seed: int, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))
    for _ in range(8):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.05, 0.22) * w
        colour = rng.uniform(0, 255, 3)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s)))
        img += blob[..., None] * colour
    img += (xx / w)[..., None] * rng.uniform(-60, 60, 3)
    img += rng.normal(0, 6, img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    out.setflags(write=False)
    return out


def file_state(folder: Path) -> Dict[str, Tuple[str, int, int]]:
    """{relative path: (sha256, size, mtime_ns)} for every file under ``folder``."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(folder))] = (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns)
    return out


@pytest.fixture(scope="session")
def _photo_folder_template(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("template") / "photos"
    _build_photo_folder(root)
    return root


@pytest.fixture
def photo_folder(tmp_path: Path, _photo_folder_template: Path) -> Dict[str, Path]:
    """A fresh copy of a folder with planted duplicates, honest non-duplicates, junk and distinct photos."""
    root = tmp_path / "photos"
    shutil.copytree(_photo_folder_template, root)
    return _build_photo_folder(root, write=False)


def _build_photo_folder(root: Path, write: bool = True) -> Dict[str, Path]:
    if write:
        (root / "sub").mkdir(parents=True)
    paths = {
        "original": root / "original.png",
        "renamed": root / "sub" / "original - Copy.png",
        "half_jpeg": root / "half_q60.jpg",
        "flipped": root / "flipped.png",
        "rotated": root / "rotated.png",
        "broken": root / "broken.jpg",
        "truncated": root / "truncated.jpg",
        "empty": root / "empty.png",
        "notes": root / "notes.txt",
    }
    for seed in range(10, 16):
        paths[f"distinct_{seed}"] = root / f"distinct_{seed}.png"
    if write:
        original = make_photo(1)
        original.save(paths["original"])
        shutil.copyfile(paths["original"], paths["renamed"])
        original.resize((160, 120), Image.LANCZOS).save(paths["half_jpeg"], quality=60)
        original.transpose(Image.FLIP_LEFT_RIGHT).save(paths["flipped"])
        original.rotate(90, expand=True).save(paths["rotated"])
        paths["broken"].write_bytes(b"this is not a JPEG, it only has the name of one")
        full = root / "_full.jpg"
        make_photo(2).save(full, quality=90)
        data = full.read_bytes()
        paths["truncated"].write_bytes(data[: len(data) // 3])
        full.unlink()
        paths["empty"].write_bytes(b"")
        paths["notes"].write_text("shopping list", encoding="utf-8")
        for seed in range(10, 16):
            make_photo(seed).save(paths[f"distinct_{seed}"])
    paths["root"] = root
    return paths


def tidy(p: os.PathLike) -> str:
    return os.path.abspath(os.fspath(p))
