"""The pixel operations, one function per step, each one pure.

Nothing here decides whether a step should run. Each function is handed the
numbers :mod:`ocr_cleaner._analysis` measured and does exactly one thing to the
plane, returning the new plane and a sentence saying what it did. The pipeline
in :mod:`ocr_cleaner._core` does the deciding, which keeps the "why" and the
"how" in separate files and makes every step testable on its own.

Speed matters here, because a 4000 x 3000 scan is an ordinary input. The two
window operations both go through Pillow's C filters rather than numpy: a box
blur is a running sum, so a 121-pixel window costs the same as a 3-pixel one,
and it beat an exact int64 integral image by eight times on a 12 megapixel page
while differing from it by less than one grey level.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from . import _images

logger = logging.getLogger(__name__)

#: A skew smaller than this is left alone. Rotating by a fiftieth of a degree
#: costs an interpolation pass over every pixel and buys nothing an OCR engine
#: can use, so a page this straight is reported straight rather than rotated.
MIN_SKEW_TO_CORRECT = 0.05

#: How dark a pixel must be, below the paper level, to read as border rather
#: than as ink.
BORDER_DARK_MARGIN = 55.0
#: Share of a row or column that must be that dark before it is a border and not
#: a line of text. A scanner edge runs at 1.0, because it is a band of black with
#: nothing in it; even a line of heavy display type leaves paper between the
#: words and comes in well under this.
BORDER_DARK_SHARE = 0.80
#: Most of each side the border trim is ever allowed to eat.
BORDER_MAX_TRIM = 0.25
#: Extra pixels trimmed past the dark run, for the soft edge beside it.
BORDER_FEATHER = 2

#: A scanner band is BLACK, not merely darker than the paper. A pixel counts towards a
#: band only when it is below this share of the paper level as well as below the
#: relative margin. Measured on a lit page with paper at 235: a scanner band sits near
#: 8, while paper under a heavy lighting shadow bottoms out near 106 - 45% of the
#: paper level - so 0.35 separates them with room on both sides. Text keeps its own
#: darkness under a shadow, so a shadowed column of text stays well short of the 80%
#: dark share a band needs.
BORDER_BLACK_SHARE = 0.35

#: Text at least this tall, in pixels, has strokes thick enough to survive a
#: 5 x 5 median. Below it the window stays at 3 x 3, which is also the only size
#: that keeps a 12 megapixel page inside a couple of seconds.
MEDIAN_LARGE_TEXT_PX = 40.0
#: Longest side of the patch the noise estimate is measured on.
NOISE_PATCH_SIDE = 768
#: Estimated noise, in grey levels, below which a page is already clean.
NOISE_FLOOR = 1.5
#: Share of paper pixels that must be specks before impulse noise counts.
SPECK_SHARE_FLOOR = 0.0005
#: How far from the local median a pixel must sit to count as a speck.
SPECK_LEVELS = 40.0

#: Local mean window as a multiple of the text height. The window has to hold a
#: whole character and some of the paper around it; much wider and it stops
#: tracking the lighting it is there to follow.
ADAPTIVE_WINDOW_FACTOR = 3.0
#: Window used when no text was found to size one from.
ADAPTIVE_WINDOW_DEFAULT = 31
#: Hard limits on that window, in pixels.
ADAPTIVE_WINDOW_MIN = 15
ADAPTIVE_WINDOW_MAX = 151
#: How far under the local mean a pixel must fall to be called ink, as a share
#: of the page's own ink-to-paper contrast.
ADAPTIVE_OFFSET_SHARE = 0.12
#: Floor for that offset, so a very flat page does not binarise its own noise.
ADAPTIVE_MIN_OFFSET = 6.0

#: The three thresholding modes.
THRESHOLD_MODES = ("adaptive", "otsu", "none")
#: Grey level at which an upscaled binary image is snapped back to black/white.
BINARY_MIDPOINT = 128


def ink_level(plane: np.ndarray, paper: float) -> float:
    """Average luminance of the ink on the page, or the paper level if there is none."""
    ink = plane[plane < paper - BORDER_DARK_MARGIN]
    if ink.size == 0:
        return float(paper)
    return float(ink.mean())


def contrast_of(plane: np.ndarray, paper: float) -> float:
    """How far the ink sits below the paper, in grey levels. Zero on blank paper."""
    return max(0.0, float(paper) - ink_level(plane, paper))


# --------------------------------------------------------------------------- #
# deskew
# --------------------------------------------------------------------------- #

def deskew(image: Image.Image, degrees: float, paper: float) -> Tuple[Image.Image, str]:
    """Rotate ``image`` back to horizontal, filling the corners with paper.

    ``degrees`` is the skew as :func:`ocr_cleaner.estimate_skew` reports it, so
    the correction applied is ``-degrees``. The canvas grows to hold the turned
    page rather than clipping its corners, and the new corners are filled with
    the page's own paper colour - white would print a bright frame around a grey
    scan and give the next step a false paper level to work from.
    """
    fill = int(round(max(0.0, min(255.0, paper))))
    turned = image.rotate(
        -float(degrees),
        resample=_images.BICUBIC,
        expand=True,
        fillcolor=fill,
    )
    return turned, (
        "rotated {0:+.2f} degrees, corners filled with the paper level "
        "({1})".format(-float(degrees), fill)
    )


# --------------------------------------------------------------------------- #
# border
# --------------------------------------------------------------------------- #

def _trim_one_side(darkness: np.ndarray, limit: int, gap: int = 0) -> int:
    """How many leading entries of ``darkness`` are border rather than page.

    ``gap`` is how many pale lines may sit between the edge and the band before
    the scan gives up. It is zero unless the page has been rotated, in which
    case it is exactly the width of the paper-coloured wedge the rotation left
    outside the page - so the scan can reach a band the turn pushed inward, and
    can never reach anything further in than the turn itself accounts for.
    """
    last_dark = -1
    clean_run = 0
    for index in range(min(limit, int(darkness.size))):
        if darkness[index] >= BORDER_DARK_SHARE:
            last_dark = index
            clean_run = 0
        else:
            clean_run += 1
            if clean_run > gap:
                break
    if last_dark < 0:
        return 0
    return min(limit, last_dark + 1 + BORDER_FEATHER)


def find_content_box(
    plane: np.ndarray, paper: float, gap: Tuple[int, int] = (0, 0)
) -> Optional[Tuple[int, int, int, int]]:
    """The page inside the scan, as ``(left, top, right, bottom)``, or ``None``.

    A scanner lid that did not close, or a page smaller than the glass, leaves a
    black band down one or more sides. The band is solid: every pixel across it
    is dark, where even a line of heavy display type leaves paper between the
    words. The trim walks in from each side while the outermost line is still
    almost entirely dark, and stops at the first line that is not (or, on a page
    that has been rotated, at the first run of pale lines wider than the wedge
    that rotation left), so it can never walk past the margin into the text.

    Args:
        plane: the luminance plane to look at.
        paper: the paper level on that plane.
        gap: ``(rows, columns)`` of pale lines the scan may cross to reach a
            band, as :func:`_trim_one_side` explains. Zero on a page that has
            not been rotated.

    Returns:
        The box, or ``None`` when there is no border to remove or when removing
        what was found would eat so much of the page that the measurement is
        more likely wrong than the page is small.
    """
    height, width = plane.shape
    if height < 8 or width < 8:
        return None
    # "Darker than the paper" alone made a soft lighting shadow read as a scanner band.
    # Under a shadow the paper itself falls below a single page-wide paper level, so a
    # line of text there looked almost wholly dark, and the trim walked in and silently
    # cut away up to a quarter of the page, text and all. A real band is BLACK, so a
    # pixel has to clear an absolute bar as well as the relative one. This keeps working
    # on a skewed page, where one line legitimately crosses band and paper both.
    dark = plane < min(paper - BORDER_DARK_MARGIN, BORDER_BLACK_SHARE * paper)
    rows = dark.mean(axis=1)
    columns = dark.mean(axis=0)
    row_limit = int(height * BORDER_MAX_TRIM)
    column_limit = int(width * BORDER_MAX_TRIM)
    row_gap, column_gap = int(gap[0]), int(gap[1])
    top = _trim_one_side(rows, row_limit, row_gap)
    bottom = height - _trim_one_side(rows[::-1], row_limit, row_gap)
    left = _trim_one_side(columns, column_limit, column_gap)
    right = width - _trim_one_side(columns[::-1], column_limit, column_gap)
    if (left, top, right, bottom) == (0, 0, width, height):
        return None
    if right - left < width * 0.4 or bottom - top < height * 0.4:
        logger.debug("border trim rejected: it would have kept too little of the page")
        return None
    if float(dark[top:bottom, left:right].mean()) >= BORDER_DARK_SHARE:
        # A border is a dark edge around a paper interior. If the interior is
        # just as dark, this is a dark page, and trimming it would take a bite
        # out of one for no reason.
        logger.debug("border trim rejected: the page is dark all the way through")
        return None
    return (left, top, right, bottom)


def widen_for_rotation(
    box: Tuple[int, int, int, int],
    size: Tuple[int, int],
    rotated_size: Tuple[int, int],
    degrees: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Carry a content box measured before a rotation onto the rotated canvas.

    Deskewing runs first, which leaves the black band tilted and no longer
    square to anything - too thin a stripe, at too shallow an angle, for a
    row-and-column count to find again. So the band is measured while it is
    still square, and the trim is then widened by exactly how far the rotation
    pushed it: a side of the page ``e`` pixels long swings out by ``e * sin(a)``
    when the page turns by ``a``, and that, plus the band itself, is what has to
    come off.

    Only sides that had a border are widened. A side that was clean is left
    alone, corner fill and all, because cutting it would take a wedge of real
    page with it for no reason.

    Returns ``None`` if what is left would be under 40% of the page either way.
    """
    left, top, right, bottom = box
    width, height = size
    new_width, new_height = rotated_size
    sine = abs(float(np.sin(np.radians(degrees))))
    cosine = abs(float(np.cos(np.radians(degrees))))

    def grown(trim: int, across: int) -> int:
        return 0 if trim <= 0 else int(np.ceil(across * sine + trim * cosine))

    new_left = grown(left, height)
    new_right = new_width - grown(width - right, height)
    new_top = grown(top, width)
    new_bottom = new_height - grown(height - bottom, width)
    new_left = max(0, min(new_left, new_width))
    new_top = max(0, min(new_top, new_height))
    new_right = max(new_left, min(new_right, new_width))
    new_bottom = max(new_top, min(new_bottom, new_height))
    if (new_left, new_top, new_right, new_bottom) == (0, 0, new_width, new_height):
        return None
    if new_right - new_left < new_width * 0.4 or new_bottom - new_top < new_height * 0.4:
        logger.debug("border trim rejected after rotation: too little page left")
        return None
    return (new_left, new_top, new_right, new_bottom)


def union_boxes(
    first: Optional[Tuple[int, int, int, int]],
    second: Optional[Tuple[int, int, int, int]],
    size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """The tighter of two content boxes on each side, or ``None`` if neither trims.

    A black band is square to the image when it belongs to the scanner and
    square to the page when it belongs to the page, and a deskew turns each of
    those into the other. Both are looked for, and whatever either one found
    comes off.
    """
    boxes = [box for box in (first, second) if box is not None]
    if not boxes:
        return None
    width, height = size
    left = max(box[0] for box in boxes)
    top = max(box[1] for box in boxes)
    right = min(box[2] for box in boxes)
    bottom = min(box[3] for box in boxes)
    if (left, top, right, bottom) == (0, 0, width, height):
        return None
    if right - left < width * 0.4 or bottom - top < height * 0.4:
        logger.debug("border trim rejected: the two readings together keep too little")
        return None
    return (left, top, right, bottom)


def describe_border(box: Tuple[int, int, int, int], size: Tuple[int, int]) -> str:
    """Plain sentence for what :func:`find_content_box` decided to cut."""
    left, top, right, bottom = box
    width, height = size
    cuts = []
    for amount, side in (
        (left, "left"),
        (width - right, "right"),
        (top, "top"),
        (height - bottom, "bottom"),
    ):
        if amount > 0:
            cuts.append("{0} px off the {1}".format(amount, side))
    return "trimmed {0}, leaving {1} x {2}".format(
        ", ".join(cuts), right - left, bottom - top
    )


# --------------------------------------------------------------------------- #
# denoise
# --------------------------------------------------------------------------- #

def median_window_for(text_height: Optional[float]) -> int:
    """The median window that fits text ``text_height`` pixels tall: 3 or 5.

    A median window has to stay well inside the width of a stroke, or it rounds
    the corners off letters and thins them until the OCR engine sees something
    else. Only text tall enough to have strokes several pixels wide can carry a
    5 x 5 window.
    """
    if text_height is not None and text_height >= MEDIAN_LARGE_TEXT_PX:
        return 5
    return 3


def _noise_patch(plane: np.ndarray) -> np.ndarray:
    """The middle of the page, at most :data:`NOISE_PATCH_SIDE` on a side."""
    height, width = plane.shape
    take_h = min(height, NOISE_PATCH_SIDE)
    take_w = min(width, NOISE_PATCH_SIDE)
    top = (height - take_h) // 2
    left = (width - take_w) // 2
    return plane[top: top + take_h, left: left + take_w]


def estimate_noise(plane: np.ndarray, paper: float) -> Tuple[float, float]:
    """Measure grain and specks on the paper. Returns ``(levels, speck_share)``.

    The measurement is taken on the paper only. Paper is supposed to be flat, so
    whatever moves there is noise; measuring over the whole page instead would
    count every stroke edge as grain and call a crisp page noisy.
    """
    patch = _noise_patch(plane)
    if patch.size < 64:
        return 0.0, 0.0
    smoothed = _images.to_array(_images.from_array(patch).filter(ImageFilter.MedianFilter(3)))
    difference = np.abs(patch.astype(np.float64) - smoothed.astype(np.float64))
    on_paper = patch >= paper - 20.0
    sample = difference[on_paper] if int(on_paper.sum()) >= 200 else difference.ravel()
    if sample.size == 0:                 # pragma: no cover - guarded by the size check
        return 0.0, 0.0
    levels = 1.4826 * float(np.median(sample))
    specks = float(np.mean(sample > SPECK_LEVELS))
    return levels, specks


def is_noisy(levels: float, specks: float) -> bool:
    """Whether a page has enough grain or specks to be worth filtering."""
    return levels >= NOISE_FLOOR or specks >= SPECK_SHARE_FLOOR


def median_filter(image: Image.Image, size: int) -> Image.Image:
    """Apply a ``size`` x ``size`` median filter."""
    return image.filter(ImageFilter.MedianFilter(int(size)))


# --------------------------------------------------------------------------- #
# threshold
# --------------------------------------------------------------------------- #

def adaptive_window_for(text_height: Optional[float]) -> int:
    """Odd local-mean window, in pixels, for text ``text_height`` tall."""
    if text_height is None or text_height <= 0.0:
        window = ADAPTIVE_WINDOW_DEFAULT
    else:
        window = int(round(ADAPTIVE_WINDOW_FACTOR * float(text_height)))
    window = max(ADAPTIVE_WINDOW_MIN, min(ADAPTIVE_WINDOW_MAX, window))
    return window if window % 2 == 1 else window + 1


def local_mean(plane: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``plane`` over a ``window`` x ``window`` box around every pixel."""
    return _images.local_mean(plane, max(1, int(window) // 2))


def adaptive_threshold(
    plane: np.ndarray, window: int, offset: float
) -> np.ndarray:
    """Binarise against the local mean: ink is ``offset`` below its surroundings.

    This is the default because a page lit unevenly is the normal case, not the
    exception. One global cut has to choose between losing the text in the dark
    corner and flooding the bright one; a local one asks the same question of
    every neighbourhood separately and gets both right.
    """
    means = local_mean(plane, window)
    ink = plane.astype(np.float64) < (means - float(offset))
    return np.where(ink, 0, 255).astype(np.uint8)


def otsu_level(plane: np.ndarray) -> int:
    """The global cut that best splits this page's histogram in two.

    Otsu's method: try every cut, keep the one whose two sides are internally
    tightest. Right for a page lit evenly end to end, and only for that.
    """
    counts = np.bincount(plane.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:                     # pragma: no cover - empty planes are caught earlier
        return BINARY_MIDPOINT
    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(counts)
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * levels)
    sum_total = sum_low[-1]
    usable = (weight_low > 0.0) & (weight_high > 0.0)
    if not usable.any():
        return BINARY_MIDPOINT
    mean_low = np.divide(sum_low, weight_low, out=np.zeros(256), where=weight_low > 0)
    mean_high = np.divide(
        sum_total - sum_low, weight_high, out=np.zeros(256), where=weight_high > 0
    )
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~usable] = -1.0
    return int(np.argmax(between))


def global_threshold(plane: np.ndarray, level: int) -> np.ndarray:
    """Binarise at one grey level: anything at or below ``level`` is ink."""
    return np.where(plane <= level, 0, 255).astype(np.uint8)


def is_binary(plane: np.ndarray) -> bool:
    """True when a plane holds nothing but 0 and 255."""
    return bool(np.isin(plane, (0, 255)).all())


# --------------------------------------------------------------------------- #
# upscale
# --------------------------------------------------------------------------- #

def upscale(image: Image.Image, factor: float, binary: bool) -> Tuple[Image.Image, str]:
    """Enlarge ``image`` by ``factor`` with Lanczos resampling.

    A binary page is snapped back to black and white afterwards. Lanczos puts a
    soft grey ramp on every edge it enlarges, which is fine for a greyscale page
    and a broken promise for one the caller asked to have thresholded.
    """
    width, height = image.size
    size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    bigger = image.resize(size, _images.LANCZOS)
    detail = "resized {0} x {1} to {2} x {3} (x{4:.2f}), Lanczos".format(
        width, height, size[0], size[1], factor
    )
    if binary:
        plane = _images.to_array(bigger)
        bigger = _images.from_array(global_threshold(plane, BINARY_MIDPOINT - 1))
        detail += ", snapped back to black and white"
    return bigger, detail
