"""Synthetic pages for the test suite. Everything is generated here; nothing is downloaded.

The text pages are drawn glyph by glyph rather than as solid bars, with
ascenders and descenders in roughly Latin proportions, so the profile they give
has the same shape real type gives: a dense x-height body with sparser strokes
reaching above and below it. That is what the orientation test and the
text-height measure actually respond to.

:func:`typed_page` sets real TrueType type instead, through Pillow: a system
face when one is installed (Arial, DejaVu Sans, Liberation Sans, Helvetica),
otherwise the scalable face Pillow itself ships. Line geometry and the
upside-down test are checked on those pages as well, because hand-drawn glyphs
only ever exercise what they were drawn to exercise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PAPER = 243
INK = 28

_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def text_page(
    width: int = 1275,
    height: int = 1650,
    x_height: int = 12,
    pitch: Optional[int] = None,
    margin: int = 110,
    paper: int = PAPER,
    ink: int = INK,
    seed: int = 0,
    blur: float = 0.6,
) -> np.ndarray:
    """A page of word-shaped ink in rows, as an ``(h, w)`` uint8 array.

    ``x_height`` is the height of the letter body. Ascenders reach 0.75 of an
    x-height above it and descenders drop 0.4 below it, so an inked line is
    about 2.15 x-heights tall from ascender top to descender foot.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), paper, dtype=np.uint8)
    pitch = pitch or int(round(x_height * 3.4))
    stroke = max(2, int(round(x_height / 6.0)))
    ascender = int(round(x_height * 0.75))
    descender = int(round(x_height * 0.40))
    baseline = margin + ascender + x_height
    while baseline + descender < height - margin:
        x = margin
        while True:
            letters = int(rng.integers(2, 9))
            char_w = int(round(x_height * 0.62))
            word_w = letters * (char_w + stroke)
            if x + word_w > width - margin:
                break
            for _ in range(letters):
                top = baseline - x_height
                page[top:baseline, x:x + stroke] = ink
                page[top:baseline, x + char_w - stroke:x + char_w] = ink
                if rng.random() < 0.6:
                    page[top:top + stroke, x:x + char_w] = ink
                if rng.random() < 0.3:
                    page[baseline - stroke:baseline, x:x + char_w] = ink
                roll = rng.random()
                if roll < 0.32:
                    page[top - ascender:top, x:x + stroke] = ink
                elif roll < 0.45:
                    page[baseline:baseline + descender,
                         x + char_w - stroke:x + char_w] = ink
                x += char_w + stroke
            x += int(round(x_height * 0.9))
        baseline += pitch
    if blur > 0:
        page = np.asarray(
            Image.fromarray(page).filter(ImageFilter.GaussianBlur(blur)),
            dtype=np.uint8,
        ).copy()
    return page


def inked_line_height(x_height: int) -> int:
    """The ascender-top to descender-foot height :func:`text_page` draws."""
    return int(round(x_height * 0.75)) + x_height + int(round(x_height * 0.40))


def rotate(page: np.ndarray, degrees: float, paper: int = PAPER) -> np.ndarray:
    """``page`` turned ``degrees`` counter-clockwise, like ``PIL.Image.rotate``."""
    image = Image.fromarray(page)
    turned = image.rotate(degrees, resample=_BICUBIC, fillcolor=paper)
    return np.asarray(turned, dtype=np.uint8).copy()


def blank_page(width: int = 1275, height: int = 1650, seed: int = 0) -> np.ndarray:
    """Clean paper with the faint grain a real scanner leaves on it."""
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 1.5, size=(height, width))
    return np.clip(PAPER + grain, 0, 255).astype(np.uint8)


def photograph(
    width: int = 900, height: int = 675, seed: int = 0, colour: bool = True
) -> np.ndarray:
    """A smooth, colourful scene with fine texture: nothing like ink on paper."""
    rng = np.random.default_rng(seed)
    channels = []
    for _ in range(3 if colour else 1):
        coarse = rng.uniform(20, 235, size=(6, 8)).astype(np.uint8)
        smooth = Image.fromarray(coarse).resize((width, height), _BICUBIC)
        plane = np.asarray(smooth, dtype=np.float64)
        plane += rng.normal(0.0, 9.0, size=plane.shape)
        channels.append(np.clip(plane, 0, 255))
    if colour:
        return np.stack(channels, axis=2).astype(np.uint8)
    return channels[0].astype(np.uint8)


def shadow_left(page: np.ndarray, depth: float = 0.45, reach: float = 0.3) -> np.ndarray:
    """``page`` with a soft lighting shadow falling off towards the left edge.

    Lighting multiplies, so paper and ink darken together. At the left edge the
    light is ``1 - depth`` of full; it recovers smoothly to full at ``reach`` of
    the page width.
    """
    height, width = page.shape[:2]
    x = np.arange(width, dtype=np.float64) / max(1.0, reach * width)
    ramp = np.clip(x, 0.0, 1.0)
    light = 1.0 - depth * (0.5 + 0.5 * np.cos(np.pi * ramp))
    shaped = light[None, :] if page.ndim == 2 else light[None, :, None]
    return np.clip(page.astype(np.float64) * shaped, 0, 255).astype(np.uint8)


def black_border_left(page: np.ndarray, width_px: int = 70, level: int = 12) -> np.ndarray:
    """``page`` with a genuinely black scanner border down the left side."""
    out = page.copy()
    out[:, :width_px] = level
    return out


def show_through(page: np.ndarray, strength: float = 0.5, seed: int = 3) -> np.ndarray:
    """``page`` with the mirrored, blurred print of another page bleeding through."""
    height, width = page.shape
    back = text_page(width, height, x_height=13, seed=seed, blur=0.0)[:, ::-1]
    ghost = Image.fromarray(back).filter(ImageFilter.GaussianBlur(3.0))
    darkening = (PAPER - np.asarray(ghost, dtype=np.float64)) / float(PAPER)
    darkening = np.clip(darkening, 0.0, 1.0) * strength
    return np.clip(page.astype(np.float64) * (1.0 - darkening), 0, 255).astype(np.uint8)


def blurred(page: np.ndarray, radius: float = 4.0) -> np.ndarray:
    """``page`` out of focus."""
    return np.asarray(
        Image.fromarray(page).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8
    ).copy()


def colour_scan(page: np.ndarray, tint=(1.0, 0.985, 0.95)) -> np.ndarray:
    """``page`` as an RGB scan of slightly warm paper."""
    rgb = np.stack([page.astype(np.float64) * t for t in tint], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


#: Plain running prose, written for these tests.
PROSE = (
    "When the committee met on Thursday it agreed that the annual report should "
    "be published before the end of the month. The chair noted that the figures "
    "for the third quarter had been revised twice, and asked the finance team to "
    "check every total against the ledger before the draft went to the printer. "
    "Several members raised the question of whether the new office would open in "
    "time, since the builders had reported delays with the electrical work and "
    "the heating system. It was decided to hold a short meeting next week to "
    "review the schedule and to invite the architect to explain what had changed."
).split()
#: The same sentence over and over, so that every line is set the same way and
#: the letters line up in columns down the page.
PANGRAM = ("The quick brown fox jumps over the lazy dog. " * 60).split()

_FACES = ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
          "Helvetica.ttc", "FreeSans.ttf")


def truetype(size: int):
    """A scalable TrueType face at ``size`` px, or ``None`` if there is none."""
    for name in _FACES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        face = ImageFont.load_default(size=size)       # Pillow 10.1 and later
    except TypeError:                                   # pragma: no cover - old Pillow
        return None
    return face if isinstance(face, ImageFont.FreeTypeFont) else None


def typed_page(
    width: int = 1275,
    height: int = 1650,
    size: int = 30,
    lines: Optional[int] = None,
    margin: Optional[int] = None,
    paper: int = 245,
    ink: int = 20,
    spacing: float = 1.5,
    words=PROSE,
) -> np.ndarray:
    """A page of real type set line by line, as an ``(h, w)`` uint8 array.

    Raises:
        RuntimeError: when no scalable face is available; callers skip.
    """
    face = truetype(size)
    if face is None:
        raise RuntimeError("no TrueType face available")
    margin = int(width * 0.1) if margin is None else margin
    image = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(image)
    top, count, position = margin, 0, 0
    pitch = int(round(size * spacing))
    while top + pitch < height - margin and (lines is None or count < lines):
        line = []
        while True:
            word = words[position % len(words)]
            if line and draw.textlength(" ".join(line + [word]), font=face) > width - 2 * margin:
                break
            line.append(word)
            position += 1
        draw.text((margin, top), " ".join(line), fill=ink, font=face)
        top += pitch
        count += 1
    return np.asarray(image, dtype=np.uint8).copy()


def shadow_side(page: np.ndarray, side: str, depth: float, reach: float = 0.5) -> np.ndarray:
    """``page`` with :func:`shadow_left` falling on ``side`` instead."""
    turns = {"left": 0, "top": 1, "right": 2, "bottom": 3}[side]
    turned = np.ascontiguousarray(np.rot90(page, turns))
    return np.ascontiguousarray(np.rot90(shadow_left(turned, depth, reach), -turns))


def ghost_of(page: np.ndarray, back: np.ndarray, strength: float = 0.5,
             radius: float = 2.0) -> np.ndarray:
    """``page`` darkened by ``back`` mirrored and blurred, as if seen through the sheet."""
    blurred_back = Image.fromarray(np.ascontiguousarray(back[:, ::-1])).filter(
        ImageFilter.GaussianBlur(radius))
    back_level = float(np.percentile(back, 99))
    darkening = np.clip((back_level - np.asarray(blurred_back, dtype=np.float64))
                        / back_level, 0.0, 1.0) * strength
    return np.clip(page.astype(np.float64) * (1.0 - darkening), 0, 255).astype(np.uint8)
