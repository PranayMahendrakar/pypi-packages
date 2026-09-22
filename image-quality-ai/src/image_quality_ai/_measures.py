"""The five measures, and the arithmetic that turns each raw number into a score.

Every measure returns a :class:`~image_quality_ai._report.Metric` carrying the
raw value it actually computed, a 0-100 score, whether that counts as ok, and a
sentence naming the numbers and the boundary they were judged against. Nothing
here compares against a literal: the boundaries all come from
:class:`~image_quality_ai._thresholds.Thresholds`.

Scores are built from straight-line segments between named thresholds, so a
score of 50 always means "exactly on the borderline" and the number can be read
back without opening this file.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ._loading import LoadedImage, _BOX
from ._report import Metric
from ._thresholds import DIRECTIONAL_BLUR_RATIO, DIRECTIONAL_DETAIL_FLOOR, FRAMING_GRID, NOISE_GRID, NOISE_TILE, Thresholds

#: Order the measures are reported in.
MEASURE_NAMES = ("sharpness", "exposure", "contrast", "noise", "framing")

#: Number of bins in the luminance histogram the percentile figures come from.
#: 256 bins put every percentile within 1/255 of its true value, which is finer
#: than the difference between two adjacent 8-bit levels.
HISTOGRAM_BINS = 256

#: Scale factor from the median absolute deviation to a standard deviation for
#: normally distributed data.
MAD_TO_SIGMA = 1.4826


# ----------------------------------------------------------------------------
# score shapes
# ----------------------------------------------------------------------------
def _ramp(value: float, full_at: float, zero_at: float) -> float:
    """100 at ``full_at``, 0 at ``zero_at``, a straight line between them."""
    span = full_at - zero_at
    if span == 0.0:
        return 100.0 if value == full_at else 0.0
    fraction = (value - zero_at) / span
    return float(max(0.0, min(1.0, fraction)) * 100.0)


def _anchors(
    zero_at: float, half_at: float, full_at: float
) -> Tuple[float, float, float]:
    """Put the three anchors back in order after a user override crossed them.

    Every threshold is overridable, and nothing stops a caller moving a
    borderline past the floor or ceiling around it - lowering
    ``sharpness_blurry`` below ``sharpness_floor`` to grade soft imagery, say.
    A ramp whose middle anchor sits outside its ends doubles back on itself and
    returns a meaningless score, so the outer anchors give way to the borderline
    the caller actually asked for. With the defaults, which are already in
    order, this changes nothing.
    """
    if full_at >= zero_at:                  # bigger is better
        return min(zero_at, half_at), half_at, max(full_at, half_at)
    return max(zero_at, half_at), half_at, min(full_at, half_at)


def _segments(value: float, zero_at: float, half_at: float, full_at: float) -> float:
    """0 at ``zero_at``, 50 at ``half_at``, 100 at ``full_at``.

    Works in either direction, so it reads the same whether a bigger number is
    better (sharpness) or worse (noise).
    """
    zero_at, half_at, full_at = _anchors(zero_at, half_at, full_at)
    if full_at >= zero_at:
        if value <= zero_at:
            return 0.0
        if value >= full_at:
            return 100.0
        if value <= half_at:
            return 50.0 * _safe_fraction(value - zero_at, half_at - zero_at)
        return 50.0 + 50.0 * _safe_fraction(value - half_at, full_at - half_at)
    if value >= zero_at:
        return 0.0
    if value <= full_at:
        return 100.0
    if value >= half_at:
        return 50.0 * _safe_fraction(zero_at - value, zero_at - half_at)
    return 50.0 + 50.0 * _safe_fraction(half_at - value, half_at - full_at)


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def _log_segments(value: float, zero_at: float, half_at: float, full_at: float) -> float:
    """:func:`_segments` on a logarithmic axis, for quantities that span decades."""
    floor = 1e-6
    return _segments(
        math.log(max(value, floor)),
        math.log(max(zero_at, floor)),
        math.log(max(half_at, floor)),
        math.log(max(full_at, floor)),
    )


def _pct(fraction: float) -> str:
    """A share as a percentage, with enough digits to stay honest."""
    value = fraction * 100.0
    if 0.0 < value < 0.1:
        return "<0.1%"
    return "{0:.1f}%".format(value)


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------
def laplacian(plane: np.ndarray) -> np.ndarray:
    """The 4-neighbour Laplacian over the interior of ``plane``.

    Returns an array two pixels smaller in each direction, and an empty array
    when the plane is too small to have an interior at all. No padding is
    invented, so a 1x1 image reports no edge energy instead of a made-up number.
    """
    height, width = plane.shape
    if height < 3 or width < 3:
        return np.empty((0, 0), dtype=np.float32)
    middle = plane[1:-1, 1:-1]
    out = plane[:-2, 1:-1] + plane[2:, 1:-1]
    out = out + plane[1:-1, :-2] + plane[1:-1, 2:]
    return out - np.float32(4.0) * middle


def luminance_histogram(plane: np.ndarray) -> np.ndarray:
    """Counts per 1/256 luminance bin, over every pixel of the full-size plane."""
    quantised = np.clip(plane * np.float32(HISTOGRAM_BINS), 0, HISTOGRAM_BINS - 1)
    return np.bincount(
        quantised.astype(np.uint8).ravel(), minlength=HISTOGRAM_BINS
    ).astype(np.int64)


def percentile_from_histogram(histogram: np.ndarray, fraction: float) -> float:
    """The luminance below which ``fraction`` of the pixels sit."""
    total = int(histogram.sum())
    if total <= 0:
        return 0.0
    target = fraction * total
    cumulative = np.cumsum(histogram)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = max(0, min(HISTOGRAM_BINS - 1, index))
    return (index + 0.5) / float(HISTOGRAM_BINS)


def _median_filter_3x3(plane: np.ndarray) -> Optional[np.ndarray]:
    """3x3 median over the interior, or ``None`` when the plane is too small."""
    height, width = plane.shape
    if height < 3 or width < 3:
        return None
    windows = np.stack(
        [plane[row:row + height - 2, col:col + width - 2] for row in range(3) for col in range(3)]
    )
    return np.median(windows, axis=0)


def _tile_origins(length: int, tile: int, count: int) -> List[int]:
    """Evenly spaced, non-overlapping-where-possible tile starts along one axis."""
    if length <= tile:
        return [0]
    span = length - tile
    if count <= 1:
        return [span // 2]
    seen = []
    for index in range(count):
        start = int(round(span * index / float(count - 1)))
        if start not in seen:
            seen.append(start)
    return seen


def noise_residual(
    plane: np.ndarray, tile: int = NOISE_TILE, grid: int = NOISE_GRID
) -> Optional[np.ndarray]:
    """Absolute residual after a 3x3 median, sampled at the plane's own resolution.

    Tiles are taken on a fixed grid across the frame rather than from one crop,
    so a clean sky in the corner cannot pass a grainy photo and one busy patch of
    foliage cannot fail a clean one. Returns ``None`` when the plane is smaller
    than the 3x3 window the filter needs.
    """
    height, width = plane.shape
    if height < 3 or width < 3:
        return None
    pieces: List[np.ndarray] = []
    for top in _tile_origins(height, tile, grid):
        for left in _tile_origins(width, tile, grid):
            patch = plane[top:top + min(tile, height), left:left + min(tile, width)]
            smoothed = _median_filter_3x3(patch)
            if smoothed is None:            # pragma: no cover - guarded above
                continue
            pieces.append(np.abs(patch[1:-1, 1:-1] - smoothed).ravel())
    if not pieces:                          # pragma: no cover - guarded above
        return None
    return np.concatenate(pieces)


# ----------------------------------------------------------------------------
# the measures
# ----------------------------------------------------------------------------

def directional_detail(plane: np.ndarray) -> "tuple[float, float]":
    """Edge energy along each axis, as (horizontal_detail, vertical_detail).

    The Laplacian is isotropic, so it cannot see motion blur: smearing an image
    sideways destroys the horizontal detail while leaving every vertical edge
    intact, and the combined variance barely moves. A panned photo therefore
    scored as sharp as the original - in testing, higher than it. Measuring the
    two axes separately makes the smear obvious, because one axis collapses while
    the other does not.
    """
    if plane.shape[0] < 3 or plane.shape[1] < 3:
        return 0.0, 0.0
    d_x = np.diff(plane.astype(np.float64), axis=1)
    d_y = np.diff(plane.astype(np.float64), axis=0)
    return float(np.var(d_x)), float(np.var(d_y))


def measure_sharpness(image: LoadedImage, thresholds: Thresholds) -> Metric:
    """Variance of the Laplacian, normalised by measuring at a fixed size."""
    plane = image.analysis * np.float32(255.0)
    response = laplacian(plane)
    if response.size == 0:
        return Metric(
            value=0.0,
            score=0.0,
            ok=False,
            message=(
                "the image is {0}x{1}, too small to have any interior pixels, so no edge "
                "detail can be measured".format(image.width, image.height)
            ),
            details={"laplacian_variance": 0.0, "measured_pixels": 0},
            name="sharpness",
        )
    value = float(np.var(response, dtype=np.float64))

    # Directional blur check. A sideways smear leaves vertical edges untouched, so the
    # isotropic Laplacian above stays high and a badly panned photo passes as crisp.
    # When one axis holds far less detail than the other, the lower axis is what the
    # viewer actually sees, so sharpness is judged on it.
    detail_x, detail_y = directional_detail(plane)
    weaker = min(detail_x, detail_y)
    stronger = max(detail_x, detail_y)
    directional = (
        stronger > 0.0
        and weaker < DIRECTIONAL_DETAIL_FLOOR
        and weaker / stronger < DIRECTIONAL_BLUR_RATIO
    )
    if directional:
        # Judge the frame on the axis the viewer actually loses. The factor of four puts
        # first-difference variance on roughly the same scale as the Laplacian variance
        # the thresholds are written in.
        #
        # The penalty stops at the "soft" boundary rather than bottoming out, because a
        # still image cannot always tell a smear from a subject that genuinely varies in
        # only one direction: crisp vertical stripes, a page of text, a picket fence
        # against plain sky all leave one axis nearly featureless while being perfectly
        # sharp. Flagging the imbalance is useful; condemning the photograph on it alone
        # would fail exactly those subjects, so the message says what was seen and lets
        # the reader judge.
        floor = thresholds.sharpness_blurry * 1.6
        value = min(value, max(weaker * 4.0, floor))

    score = _log_segments(
        value, thresholds.sharpness_floor, thresholds.sharpness_blurry, thresholds.sharpness_good
    )
    if value <= thresholds.sharpness_floor:
        verdict = "no edge detail at all, the frame is blank or featureless"
    elif directional:
        axis = "sideways" if detail_x < detail_y else "vertically"
        verdict = (
            "detail runs almost entirely one way, as though smeared {0}; that is usually "
            "motion blur, though a subject that only varies in one direction - stripes, "
            "lines of text, a fence against plain sky - looks the same to a still frame"
        ).format(axis)
    elif value < thresholds.sharpness_blurry * 0.25:
        verdict = "badly out of focus"
    elif value < thresholds.sharpness_blurry:
        verdict = "soft"
    elif value < thresholds.sharpness_good:
        verdict = "acceptably sharp"
    else:
        verdict = "crisp"
    return Metric(
        value=value,
        score=score,
        ok=score >= thresholds.measure_ok_score,
        message=(
            "Laplacian variance {0:.1f} measured at {1}x{2}: {3}. Below {4:g} reads as "
            "blurry, {5:g} and up is crisp.".format(
                value,
                image.analysis.shape[1],
                image.analysis.shape[0],
                verdict,
                thresholds.sharpness_blurry,
                thresholds.sharpness_good,
            )
        ),
        details={
            "laplacian_variance": value,
            "detail_horizontal": detail_x,
            "detail_vertical": detail_y,
            "directional_blur": directional,
            "measured_width": int(image.analysis.shape[1]),
            "measured_height": int(image.analysis.shape[0]),
        },
        name="sharpness",
    )


def measure_exposure(image: LoadedImage, thresholds: Thresholds) -> Metric:
    """Mean luminance, plus the share of pixels clipped at black and at white."""
    plane = image.luminance
    total = float(plane.size)
    mean = float(np.mean(plane, dtype=np.float64))
    black = float(np.count_nonzero(plane <= np.float32(thresholds.black_level))) / total
    white = float(np.count_nonzero(plane >= np.float32(thresholds.white_level))) / total

    if mean < thresholds.exposure_mean_low:
        mean_score = _ramp(mean, thresholds.exposure_mean_low, thresholds.exposure_mean_dark)
    elif mean > thresholds.exposure_mean_high:
        mean_score = _ramp(mean, thresholds.exposure_mean_high, thresholds.exposure_mean_bright)
    else:
        mean_score = 100.0

    clipped = min(1.0, black + white)
    over = _safe_fraction(
        clipped - thresholds.clipping_ok, thresholds.clipping_bad - thresholds.clipping_ok
    ) if clipped > thresholds.clipping_ok else 0.0
    penalty = thresholds.clipping_penalty * over
    score = max(0.0, mean_score - penalty)

    if mean <= thresholds.exposure_mean_dark:
        verdict = "almost entirely black"
    elif mean < thresholds.exposure_mean_low:
        verdict = "underexposed"
    elif mean > thresholds.exposure_mean_bright:
        verdict = "almost entirely white"
    elif mean > thresholds.exposure_mean_high:
        verdict = "overexposed"
    else:
        verdict = "well exposed"

    return Metric(
        value=mean,
        score=score,
        ok=score >= thresholds.measure_ok_score,
        message=(
            "mean luminance {0:.3f} of 1.0: {1}. {2} of pixels crushed to black and {3} "
            "blown out to white. A well exposed frame averages {4:g} to {5:g}.".format(
                mean,
                verdict,
                _pct(black),
                _pct(white),
                thresholds.exposure_mean_low,
                thresholds.exposure_mean_high,
            )
        ),
        details={
            "mean_luminance": mean,
            "clipped_black": black,
            "clipped_white": white,
            "clipping_penalty": penalty,
        },
        name="exposure",
    )


def measure_contrast(
    image: LoadedImage, thresholds: Thresholds, histogram: np.ndarray
) -> Metric:
    """Luminance standard deviation and the 5th-to-95th percentile spread."""
    std = float(np.std(image.luminance, dtype=np.float64))
    p5 = percentile_from_histogram(histogram, 0.05)
    p95 = percentile_from_histogram(histogram, 0.95)
    spread = max(0.0, p95 - p5)

    spread_score = _segments(
        spread,
        thresholds.contrast_spread_floor,
        thresholds.contrast_spread_flat,
        thresholds.contrast_spread_good,
    )
    std_score = _segments(
        std, thresholds.contrast_std_floor, thresholds.contrast_std_flat, thresholds.contrast_std_good
    )
    weight = min(1.0, max(0.0, thresholds.contrast_spread_weight))
    score = weight * spread_score + (1.0 - weight) * std_score

    if spread < thresholds.contrast_spread_floor:
        verdict = "completely flat, every pixel the same shade"
    elif spread < thresholds.contrast_spread_flat:
        verdict = "washed out"
    elif spread < thresholds.contrast_spread_good:
        verdict = "acceptable"
    else:
        verdict = "punchy"

    return Metric(
        value=spread,
        score=score,
        ok=score >= thresholds.measure_ok_score,
        message=(
            "the middle 90% of pixels span {0:.3f} of the luminance range (5th percentile "
            "{1:.3f}, 95th {2:.3f}) with a standard deviation of {3:.3f}: {4}. Below {5:g} "
            "is flat.".format(
                spread, p5, p95, std, verdict, thresholds.contrast_spread_flat
            )
        ),
        details={
            "p5_p95_spread": spread,
            "percentile_5": p5,
            "percentile_95": p95,
            "std": std,
        },
        name="contrast",
    )


def measure_noise(image: LoadedImage, thresholds: Thresholds) -> Metric:
    """High-frequency energy left over after a small median filter.

    A median filter removes single-pixel speckle but keeps edges, so what is
    left is grain rather than content. The spread of that residual is measured
    with a median absolute deviation, which the handful of real edges that do
    survive cannot drag upwards.

    This is the one measure taken at the image's own resolution rather than at
    the analysis size: grain lives in the pixels the camera produced, and
    area-averaging a 12-megapixel frame down to 512px would average away exactly
    what is being looked for. Tiles are sampled across the frame instead, so the
    cost does not grow with the megapixels.
    """
    plane = image.luminance * np.float32(255.0)
    residual = noise_residual(plane)
    if residual is None:
        # Nothing was measured, so there is no verdict to give. Saying "clean,
        # 100 of 100" here would hand a 1x1 image marks it never earned; the
        # measure steps aside instead, exactly as framing does when there is no
        # subject to place, and the overall score is taken over the rest.
        return Metric(
            value=0.0,
            score=50.0,
            ok=True,
            message=(
                "the image is {0}x{1}, too small for a 3x3 median filter, so no grain "
                "could be measured and noise was not judged".format(image.width, image.height)
            ),
            details={"noise_sigma_255": 0.0, "measured": False, "sampled_pixels": 0},
            applies=False,
            name="noise",
        )
    sigma = float(MAD_TO_SIGMA * np.median(residual))
    score = _log_segments(
        sigma, thresholds.noise_bad_255, thresholds.noise_noticeable_255, thresholds.noise_clean_255
    )
    if sigma <= thresholds.noise_clean_255:
        verdict = "clean"
    elif sigma <= thresholds.noise_noticeable_255:
        verdict = "visible grain"
    else:
        verdict = "heavy grain"
    return Metric(
        value=sigma,
        score=score,
        ok=score >= thresholds.measure_ok_score,
        message=(
            "after a 3x3 median filter the residual sigma is {0:.2f} of 255: {1}. Under "
            "{2:g} is clean, {3:g} is visible grain.".format(
                sigma, verdict, thresholds.noise_clean_255, thresholds.noise_noticeable_255
            )
        ),
        details={
            "noise_sigma_255": sigma,
            "measured": True,
            "sampled_pixels": int(residual.size),
        },
        name="noise",
    )


def _detail_grid(plane: np.ndarray) -> Optional[np.ndarray]:
    """Mean absolute Laplacian per cell of a coarse grid, on a 0-255 scale."""
    response = laplacian(plane)
    if response.size == 0:
        return None
    detail = np.abs(response).astype(np.float32)
    rows, cols = detail.shape
    grid_rows = min(FRAMING_GRID, rows)
    grid_cols = min(FRAMING_GRID, cols)
    if (rows, cols) == (grid_rows, grid_cols):
        return detail
    cells = Image.fromarray(np.ascontiguousarray(detail), mode="F").resize(
        (grid_cols, grid_rows), _BOX
    )
    return np.asarray(cells, dtype=np.float32)


def _largest_region(mask: np.ndarray, weights: np.ndarray) -> List[Tuple[int, int]]:
    """Cells of the 4-connected component carrying the most detail."""
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    best_weight = -1.0
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component: List[Tuple[int, int]] = []
            weight = 0.0
            while stack:
                here_row, here_col = stack.pop()
                component.append((here_row, here_col))
                weight += float(weights[here_row, here_col])
                for next_row, next_col in (
                    (here_row - 1, here_col),
                    (here_row + 1, here_col),
                    (here_row, here_col - 1),
                    (here_row, here_col + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
            if weight > best_weight:
                best_weight = weight
                best = component
    return best


def measure_framing(image: LoadedImage, thresholds: Thresholds) -> Metric:
    """Where the strongest detail sits, and how much of the frame it fills."""
    plane = image.analysis * np.float32(255.0)
    cells = _detail_grid(plane)
    undetermined = Metric(
        value=0.0,
        score=50.0,
        ok=True,
        message=(
            "no distinct area of detail stands out, so there is no subject to place and "
            "framing was not judged"
        ),
        details={"subject_offset": 0.0, "subject_fill": 0.0},
        applies=False,
        name="framing",
    )
    if cells is None:
        return undetermined
    peak = float(cells.max())
    if peak < thresholds.framing_min_detail_255:
        return undetermined

    mask = cells >= np.float32(peak * thresholds.framing_detail_fraction)
    component = _largest_region(mask, cells)
    if not component:
        return undetermined

    rows, cols = cells.shape
    total_weight = 0.0
    sum_x = 0.0
    sum_y = 0.0
    min_row = rows
    max_row = -1
    min_col = cols
    max_col = -1
    for row, col in component:
        weight = float(cells[row, col])
        total_weight += weight
        sum_x += weight * ((col + 0.5) / cols)
        sum_y += weight * ((row + 0.5) / rows)
        min_row = min(min_row, row)
        max_row = max(max_row, row)
        min_col = min(min_col, col)
        max_col = max(max_col, col)
    if total_weight <= 0.0:
        return undetermined

    centre_x = sum_x / total_weight
    centre_y = sum_y / total_weight
    half_diagonal = math.sqrt(0.5)          # from the centre to a corner of a unit frame
    offset = math.hypot(centre_x - 0.5, centre_y - 0.5) / half_diagonal
    fill = len(component) / float(rows * cols)

    offset_score = _ramp(offset, thresholds.framing_offset_ok, thresholds.framing_offset_bad)
    fill_score = _ramp(fill, thresholds.framing_fill_good, thresholds.framing_fill_low)
    weight = min(1.0, max(0.0, thresholds.framing_offset_weight))
    score = weight * offset_score + (1.0 - weight) * fill_score

    if offset <= thresholds.framing_offset_ok:
        placing = "sits near the centre"
    elif offset >= thresholds.framing_offset_bad:
        placing = "is pushed right to the edge of the frame"
    else:
        placing = "sits off to one side"

    return Metric(
        value=offset,
        score=score,
        ok=score >= thresholds.measure_ok_score,
        message=(
            "the main area of detail {0} ({1:.2f} of the way from centre to corner) and "
            "fills {2} of the frame. Under {3:g} counts as centred, and a subject should "
            "fill at least {4} of the frame.".format(
                placing,
                offset,
                _pct(fill),
                thresholds.framing_offset_ok,
                _pct(thresholds.framing_fill_low),
            )
        ),
        details={
            "subject_offset": offset,
            "subject_fill": fill,
            "subject_centre_x": centre_x,
            "subject_centre_y": centre_y,
            "subject_box": [
                min_col / float(cols),
                min_row / float(rows),
                (max_col + 1) / float(cols),
                (max_row + 1) / float(rows),
            ],
        },
        name="framing",
    )


def measure_all(image: LoadedImage, thresholds: Thresholds) -> Dict[str, Metric]:
    """Every measure, in report order."""
    histogram = luminance_histogram(image.luminance)
    return {
        "sharpness": measure_sharpness(image, thresholds),
        "exposure": measure_exposure(image, thresholds),
        "contrast": measure_contrast(image, thresholds, histogram),
        "noise": measure_noise(image, thresholds),
        "framing": measure_framing(image, thresholds),
    }
