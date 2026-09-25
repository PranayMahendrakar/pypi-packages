"""Page factories. Every pixel in this suite is generated here - nothing is read
from disk, nothing is downloaded, and every helper is deterministic given a seed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

#: Luminance of clean paper on the generated pages.
PAPER = 246
#: Luminance of the ink on them.
INK = 50


def text_page(
    height: int = 620,
    width: int = 480,
    *,
    paper: int = PAPER,
    ink: int = INK,
    pitch: int = 20,
    bar: int = 7,
    left: int = 50,
    right: int = 430,
    top: int = 70,
    lines: Optional[int] = None,
) -> np.ndarray:
    """A sheet of paper with text-like bars on it, as a ``(h, w)`` uint8 array.

    Bars rather than glyphs, because what every measurement in this package
    actually looks at is the horizontal band a line of text makes in a
    projection profile. Each bar is broken into words, so a row through one is
    never as solid as a scanner's black margin - which is exactly the difference
    the border trim relies on.
    """
    page = np.full((height, width), paper, dtype=np.uint8)
    rows = range(top, height - top, pitch)
    if lines is not None:
        rows = list(rows)[:lines]
    for index, row in enumerate(rows):
        column = left
        word = 0
        while column < right:
            span = 20 + ((index * 7 + word * 13) % 28)
            page[row: row + bar, column: min(column + span, right)] = ink
            column += span + 9
            word += 1
    return page


def ink_extent(page: np.ndarray, cut: int = 200) -> int:
    """How many rows of ``page`` hold any ink, top of the first to foot of the last.

    Measured from the pixels themselves rather than from font metrics, so it is
    the same number whichever way the page was drawn.
    """
    rows = np.flatnonzero((page < cut).any(axis=1))
    return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)


def scalable_font(size: int):
    """A TrueType face at ``size`` pixels, or ``None`` when none can be loaded.

    Pillow 10.1 and later carry a face of their own, which is what this asks for
    first; older Pillow falls back to whatever the machine has. Nothing is
    downloaded, and a machine with neither gets ``None`` rather than an error.
    """
    try:
        font = ImageFont.load_default(size=size)
    except (TypeError, OSError):
        font = None
    if font is not None and getattr(font, "size", None):
        return font
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


#: A line with ascenders, descenders and plenty of plain x-height in between,
#: which is the shape that tells a line's core apart from its full extent.
GLYPH_LINE = "The quick brown fox jumps over lazy dogs; pgjqy bdfhklt."


def glyph_page(
    em: int = 30,
    pitch: Optional[int] = None,
    *,
    lines: int = 12,
    paper: int = PAPER,
    ink: int = INK,
) -> Tuple[np.ndarray, int]:
    """A page of really rendered glyphs, with the true ink extent of one line.

    Returns ``(page, extent)``, where ``extent`` is the ascender top to
    descender foot of a single line of this text, measured off a render of that
    line on its own. Raises :class:`RuntimeError` when no face can be loaded;
    callers skip on it.
    """
    font = scalable_font(em)
    if font is None:
        raise RuntimeError("no scalable font on this machine")
    step = pitch if pitch is not None else int(round(em * 1.7))
    width = int(font.getlength(GLYPH_LINE)) + 140
    height = 4 * em + lines * step

    alone = Image.new("L", (width, 4 * em + 40), paper)
    ImageDraw.Draw(alone).text((70, 2 * em), GLYPH_LINE, fill=ink, font=font)
    extent = ink_extent(np.array(alone, dtype=np.uint8))

    sheet = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(sheet)
    for index in range(lines):
        draw.text((70, 2 * em + index * step), GLYPH_LINE, fill=ink, font=font)
    return np.array(sheet, dtype=np.uint8), extent


def ascender_page(
    height: int = 700,
    width: int = 620,
    *,
    core: int = 12,
    ascender: int = 8,
    descender: int = 6,
    stem: int = 6,
    pitch: int = 44,
    paper: int = PAPER,
    ink: int = INK,
) -> Tuple[np.ndarray, int]:
    """Glyph-shaped lines drawn by hand, and the extent they were drawn to.

    The same shape as rendered text and none of its dependence on a font being
    installed: a solid x-height band, a few tall strokes standing above it and a
    few tails hanging below. Those strokes are a small share of each row, so the
    rows they occupy sum to a small fraction of an x-height row in a projection
    profile - measured here and on rendered text alike they come to roughly a
    tenth of one, which is exactly the thing a cut taken from the whole profile
    swing cannot see. Returns ``(page, extent)``.
    """
    page = np.full((height, width), paper, dtype=np.uint8)
    left, right = 60, width - 60
    top = 70
    for index, row in enumerate(range(top, height - top - descender, pitch)):
        column = left
        word = 0
        while column < right:
            span = 20 + ((index * 7 + word * 13) % 28)
            stop = min(column + span, right)
            page[row: row + core, column:stop] = ink
            if word % 2 == 0:
                page[row - ascender: row, column: min(column + stem, stop)] = ink
            else:
                page[row + core: row + core + descender,
                     max(column, stop - stem): stop] = ink
            column += span + 9
            word += 1
    return page, ascender + core + descender


def rotate(page: np.ndarray, degrees: float, paper: int = PAPER) -> np.ndarray:
    """Turn a page counter-clockwise by ``degrees``, growing the canvas."""
    turned = Image.fromarray(page, mode="L").rotate(
        float(degrees), resample=Image.BICUBIC, expand=True, fillcolor=int(paper)
    )
    return np.array(turned, dtype=np.uint8)


def add_grain(page: np.ndarray, sigma: float = 7.0, seed: int = 0) -> np.ndarray:
    """Gaussian grain, the kind a cheap scanner adds."""
    rng = np.random.default_rng(seed)
    noisy = page.astype(np.float64) + rng.normal(0.0, sigma, page.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_specks(page: np.ndarray, share: float = 0.004, seed: int = 1) -> np.ndarray:
    """Salt and pepper, the kind a dusty platen adds."""
    rng = np.random.default_rng(seed)
    out = page.copy()
    picked = rng.random(page.shape) < share
    values = rng.integers(0, 2, page.shape) * 255
    out[picked] = values[picked].astype(np.uint8)
    return out


def light_unevenly(page: np.ndarray, drop: float = -70.0) -> np.ndarray:
    """Darken the page steadily from the left edge to the right one."""
    ramp = np.linspace(0.0, drop, page.shape[1])[None, :]
    return np.clip(page.astype(np.float64) + ramp, 0, 255).astype(np.uint8)


def low_contrast_uneven_page(
    height: int = 900, width: int = 700, gap: float = 55.0
) -> np.ndarray:
    """A page whose ink at the bright end is paler than its paper at the dark end.

    No single grey level can split this page in two, which is the whole case for
    thresholding against the local mean instead.
    """
    paper = np.linspace(250.0, 70.0, width)[None, :] * np.ones((height, 1))
    page = paper.copy()
    for row in range(80, height - 70, 30):
        page[row: row + 10, 70: width - 70] = paper[row: row + 10, 70: width - 70] - gap
    return np.clip(page, 0, 255).astype(np.uint8)


def add_scanner_border(
    page: np.ndarray, left: int = 26, right: int = 18, top: int = 9, value: int = 8
) -> np.ndarray:
    """Black bands down the sides, the way a scanner lid that did not close leaves them."""
    out = page.copy()
    if left:
        out[:, :left] = value
    if right:
        out[:, -right:] = value
    if top:
        out[:top, :] = value
    return out


def blank_page(
    height: int = 800, width: int = 600, paper: int = 250, sigma: float = 0.0, seed: int = 2
) -> np.ndarray:
    """An empty sheet, optionally grainy."""
    page = np.full((height, width), paper, dtype=np.uint8)
    return add_grain(page, sigma, seed) if sigma else page


def photograph(height: int = 600, width: int = 800, seed: int = 3) -> np.ndarray:
    """Something that is plainly not a document: broad tones, no paper, no lines."""
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width]
    frame = (
        120.0
        + 90.0 * np.sin(columns / 37.0)
        + 60.0 * np.cos(rows / 23.0)
        + rng.normal(0.0, 18.0, (height, width))
    )
    return np.clip(frame, 0, 255).astype(np.uint8)


def colour_page(height: int = 900, width: int = 700) -> np.ndarray:
    """A text page in RGB, on slightly warm paper with blue-black ink."""
    grey = text_page(height, width)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    ink = grey < 150
    rgb[..., 0] = np.where(ink, 30, 250)
    rgb[..., 1] = np.where(ink, 34, 244)
    rgb[..., 2] = np.where(ink, 70, 230)
    return rgb


def count_bands(binary: np.ndarray, cut: float = 0.15) -> int:
    """How many horizontal bands of ink a binarised page has."""
    inked = (binary == 0).mean(axis=1)
    count, inside = 0, False
    for value in inked:
        if value > cut and not inside:
            count += 1
            inside = True
        elif value <= cut:
            inside = False
    return count


def size_of(page: np.ndarray) -> Tuple[int, int]:
    """``(width, height)`` of an array page, the way Pillow says it."""
    return (page.shape[1], page.shape[0])


@pytest.fixture
def page() -> np.ndarray:
    """A clean, straight page of text."""
    return text_page()


@pytest.fixture
def bad_scan() -> np.ndarray:
    """One page with every problem this package exists for, at once."""
    turned = rotate(text_page(), -2.4)
    return add_scanner_border(add_grain(light_unevenly(turned), 7.0, seed=4))


@pytest.fixture
def page_file(tmp_path, page) -> str:
    """A clean page written to a PNG."""
    destination = tmp_path / "scan.png"
    Image.fromarray(page, mode="L").save(destination)
    return str(destination)
