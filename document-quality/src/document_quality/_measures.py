"""What is actually measured on a page, and the fix each measurement implies.

Every function here returns a :class:`~document_quality._report.Measure` and,
when something is wrong, the :class:`~document_quality._report.Issue` that goes
with it. Keeping the two together is deliberate: the remedy belongs next to the
measurement that motivates it, so a threshold can never drift away from the
advice it produces.

Three ideas hold the file together.

**Paper is local.** A page lit from one side has no single paper level, so the
work plane is divided into tiles, each tile's own paper level is taken, and the
grid is stretched back over the page. Everything downstream is measured against
that surface rather than one number for the whole sheet, which is why uneven
lighting does not masquerade as low contrast or as show-through.

**Sharpness is measured against the size of the text.** Optical softness is
roughly fixed in physical terms, so the same lens spreads an edge over twice as
many pixels at 600 dpi as at 300. Judging edges by pixels alone would punish
the better scan. The crispness of an edge is therefore scaled by the height of
the text it belongs to, so two scans of one sheet at different resolutions come
out level.

**A page with nothing on it is not a broken page.** Blank sheets are settled
first, before any mask is built, both because dividing by a contrast of zero is
meaningless and because a blank page deserves one honest sentence rather than
eight complaints about the text it does not have.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import _skew
from ._images import PagePlanes
from ._report import Issue, Measure
from ._skew import LineGeometry
from ._thresholds import Thresholds, bend

logger = logging.getLogger(__name__)

#: Tiles across and down the work plane used to find the local paper level.
PAPER_TILES = 8
#: Percentile within a tile taken as that tile's paper level.
PAPER_PERCENTILE = 95.0
#: Percentile of the whole work plane taken as the page's ink level. A page
#: with as little as one line of text has more inked pixels than this.
INK_PERCENTILE = 0.5
#: Normalised level below which a pixel counts as ink rather than paper.
INK_LEVEL = 0.5
#: Normalised level above which a pixel counts as clean paper.
PAPER_LEVEL = 0.85
#: Normalised band in which show-through lives: darker than paper, far lighter
#: than ink.
SHOW_THROUGH_BAND = (0.25, 0.80)
#: Gradient, in normalised units per pixel, below which a mark is soft enough
#: to have come through the sheet rather than been printed on it.
SHOW_THROUGH_SMOOTHNESS = 0.10
#: How far from real ink a soft grey mark must sit to count as show-through,
#: in work-plane pixels. Antialiasing around a stroke is not show-through.
SHOW_THROUGH_CLEARANCE = 2
#: Native-resolution tiles sampled for sharpness, and how big each one is.
SHARPNESS_TILES = 24
SHARPNESS_TILE_PX = 256
#: Percentile of the gradient within a tile taken as its strongest edge.
EDGE_PERCENTILE = 99.5
#: Text height, in pixels, that sharpness is normalised to. A page whose text
#: is this tall is judged on its edges as they are; taller text is expected to
#: spread its edges over proportionally more pixels.
REFERENCE_TEXT_HEIGHT_PX = 26.0
#: Smallest share of the page's own strongest edge that still counts as an
#: edge when building the text mask. Taking a share of the page's own maximum
#: rather than a fixed number is what keeps a softly focused page of text from
#: being read as having no text on it.
EDGE_SHARE = 0.5
#: Floor under that share, so a page of pure noise cannot call itself text.
EDGE_FLOOR = 0.10

_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


# --------------------------------------------------------------------------
# small array helpers
# --------------------------------------------------------------------------


def gradient(plane: np.ndarray) -> np.ndarray:
    """Per-pixel step size: the larger of the vertical and horizontal change.

    Forward differences, padded back to the plane's own shape, so the result
    can be masked against the plane it came from.
    """
    step = np.zeros_like(plane)
    if plane.shape[0] > 1:
        np.abs(plane[1:, :] - plane[:-1, :], out=step[:-1, :])
    if plane.shape[1] > 1:
        across = np.abs(plane[:, 1:] - plane[:, :-1])
        np.maximum(step[:, :-1], across, out=step[:, :-1])
    return step


def dilate(mask: np.ndarray, rounds: int = 1) -> np.ndarray:
    """Grow ``mask`` by ``rounds`` pixels in the four compass directions."""
    grown = mask
    for _ in range(max(0, int(rounds))):
        out = grown.copy()
        out[1:, :] |= grown[:-1, :]
        out[:-1, :] |= grown[1:, :]
        out[:, 1:] |= grown[:, :-1]
        out[:, :-1] |= grown[:, 1:]
        grown = out
    return grown


def tile_spans(length: int, tile: int) -> List[Tuple[int, int]]:
    """Cover ``length`` with spans of ``tile``, the last one nudged to fit."""
    if length <= tile:
        return [(0, length)]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(start, start + tile) for start in starts]


def paper_grid(plane: np.ndarray, tiles: int = PAPER_TILES) -> np.ndarray:
    """The paper level in each tile of ``plane``, as a small 2-D array.

    A high percentile within the tile, not its maximum, so one speck of glare
    cannot set the level for its whole neighbourhood.
    """
    height, width = plane.shape
    rows = int(min(tiles, max(1, height // 8)))
    columns = int(min(tiles, max(1, width // 8)))
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    column_edges = np.linspace(0, width, columns + 1).astype(int)
    grid = np.empty((rows, columns), dtype=np.float64)
    for r in range(rows):
        for c in range(columns):
            block = plane[row_edges[r]:row_edges[r + 1],
                          column_edges[c]:column_edges[c + 1]]
            grid[r, c] = (
                float(np.percentile(block, PAPER_PERCENTILE)) if block.size else 0.0
            )
    return grid


def stretch(grid: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Stretch a tile grid smoothly back over a plane of ``shape``."""
    height, width = shape
    if grid.shape == (1, 1):
        return np.full(shape, float(grid[0, 0]), dtype=np.float32)
    scaled = np.clip(grid * 255.0, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L").resize((width, height), _BILINEAR)
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def zone_name(row: int, column: int, rows: int, columns: int) -> str:
    """Name the ninth of the page a tile sits in, e.g. ``"top-left corner"``."""
    down = ("top", "middle", "bottom")[min(2, row * 3 // max(rows, 1))]
    across = ("left", "centre", "right")[min(2, column * 3 // max(columns, 1))]
    if down == "middle" and across == "centre":
        return "centre of the page"
    if down == "middle":
        return "{0} edge".format(across)
    if across == "centre":
        return "{0} edge".format(down)
    return "{0}-{1} corner".format(down, across)


# --------------------------------------------------------------------------
# the shared statistics
# --------------------------------------------------------------------------


@dataclass
class PageStats:
    """Everything the measures share, computed once per page."""

    #: Luminance of the darkest real ink on the page, 0 to 1.
    ink_level: float
    #: Luminance of clean paper, 0 to 1, taken over the whole page.
    paper_level: float
    #: Ink-to-paper separation, 0 to 1.
    contrast: float
    #: Paper level per tile, as a small grid.
    grid: np.ndarray
    #: The work plane mapped to 0 at ink and 1 at the local paper level.
    normalised: np.ndarray
    #: Share of the page that is ink.
    ink_share: float
    #: Share of the page that is clean paper.
    paper_share: float
    #: Share of the page that is ink belonging to thin, edge-bounded strokes.
    text_share: float
    #: Paper-level swing across the page, relative to the page's paper level.
    lighting_swing: float
    #: Where the page is darkest, in words.
    dark_zone: str
    #: Share of the page carrying soft grey marks from the reverse side.
    show_through: float
    #: Share of pixels at pure black, at native resolution.
    black_clipping: float
    #: Share of pixels at pure white, at native resolution.
    white_clipping: float
    #: Strongest edge step found, in luminance per pixel.
    edge_step: Optional[float]
    #: How much of the page is blank sheet rather than anything at all.
    blank: bool = False


def _clipping(lum: np.ndarray) -> Tuple[float, float]:
    """Share of native pixels crushed to 0 and blown to 255."""
    size = float(lum.size) or 1.0
    return (
        float(np.count_nonzero(lum == 0)) / size,
        float(np.count_nonzero(lum == 255)) / size,
    )


def _edge_step(planes: PagePlanes, normalised: np.ndarray) -> Optional[float]:
    """Strongest typical edge on the page, in luminance per pixel.

    Measured at native resolution, because resizing an image changes exactly
    the thing this measures, and only on the tiles carrying the most ink, so
    the number describes strokes rather than empty margins. The median across
    tiles is taken so one speck of dust cannot stand in for the whole page.
    """
    lum = planes.lum
    height, width = lum.shape
    tile = int(min(SHARPNESS_TILE_PX, max(16, min(height, width))))
    spans = [
        (top, bottom, left, right)
        for top, bottom in tile_spans(height, tile)
        for left, right in tile_spans(width, tile)
    ]
    if not spans:                       # pragma: no cover - a 0-pixel image
        return None
    scale = planes.work_scale
    plane_height, plane_width = normalised.shape

    def ink_in(span: Tuple[int, int, int, int]) -> float:
        top, bottom, left, right = span
        y0 = min(plane_height - 1, int(top * scale))
        y1 = max(y0 + 1, min(plane_height, int(round(bottom * scale))))
        x0 = min(plane_width - 1, int(left * scale))
        x1 = max(x0 + 1, min(plane_width, int(round(right * scale))))
        return 1.0 - float(normalised[y0:y1, x0:x1].mean())

    spans.sort(key=ink_in, reverse=True)
    steps = []
    for top, bottom, left, right in spans[:SHARPNESS_TILES]:
        block = lum[top:bottom, left:right].astype(np.float32) / np.float32(255.0)
        if block.size < 16:             # pragma: no cover - guarded by tile size
            continue
        steps.append(float(np.percentile(gradient(block), EDGE_PERCENTILE)))
    if not steps:                       # pragma: no cover - needs a 3-pixel image
        return None
    return float(np.median(steps))


def analyse(planes: PagePlanes, thresholds: Thresholds) -> PageStats:
    """Measure everything the individual measures share, once.

    A sheet with no ink-to-paper separation is settled here and returned with
    ``blank`` set, because every mask below it would be a division by nothing.
    """
    work = planes.work
    ink_level = float(np.percentile(work, INK_PERCENTILE))
    paper_level = float(np.percentile(work, 100.0 - INK_PERCENTILE))
    contrast = max(0.0, paper_level - ink_level)

    grid = paper_grid(work)
    middle = float(np.median(grid))
    swing = float(np.percentile(grid, 90.0) - np.percentile(grid, 10.0))
    lighting = swing / max(middle, 1e-6)
    darkest = int(np.argmin(grid))
    dark_zone = zone_name(
        darkest // grid.shape[1], darkest % grid.shape[1], *grid.shape
    )
    black_clipping, white_clipping = _clipping(planes.lum)

    if contrast < thresholds.blank_contrast:
        return PageStats(
            ink_level=ink_level, paper_level=paper_level, contrast=contrast,
            grid=grid, normalised=np.ones_like(work), ink_share=0.0,
            paper_share=1.0, text_share=0.0, lighting_swing=lighting,
            dark_zone=dark_zone, show_through=0.0,
            black_clipping=black_clipping, white_clipping=white_clipping,
            edge_step=None, blank=True,
        )

    local_paper = stretch(grid, work.shape)
    span = np.maximum(local_paper - np.float32(ink_level), np.float32(contrast * 0.25))
    normalised = np.clip((work - np.float32(ink_level)) / span, 0.0, 1.0)

    ink_mask = normalised < INK_LEVEL
    paper_share = float(np.count_nonzero(normalised > PAPER_LEVEL)) / normalised.size
    ink_share = float(np.count_nonzero(ink_mask)) / normalised.size

    step = gradient(normalised)
    strongest = float(np.percentile(step, EDGE_PERCENTILE)) if step.size else 0.0
    edge_mask = step > max(EDGE_FLOOR, EDGE_SHARE * strongest)
    text_mask = ink_mask & dilate(edge_mask, 1)
    text_share = float(np.count_nonzero(text_mask)) / normalised.size

    low, high = SHOW_THROUGH_BAND
    soft_grey = (normalised > low) & (normalised < high)
    soft_grey &= step < SHOW_THROUGH_SMOOTHNESS
    soft_grey &= ~dilate(normalised < low, SHOW_THROUGH_CLEARANCE)
    show_through = float(np.count_nonzero(soft_grey)) / normalised.size

    blank = ink_share < thresholds.blank_ink_share
    return PageStats(
        ink_level=ink_level, paper_level=paper_level, contrast=contrast,
        grid=grid, normalised=normalised, ink_share=ink_share,
        paper_share=paper_share, text_share=text_share, lighting_swing=lighting,
        dark_zone=dark_zone, show_through=show_through,
        black_clipping=black_clipping, white_clipping=white_clipping,
        edge_step=None if blank else _edge_step(planes, normalised), blank=blank,
    )


# --------------------------------------------------------------------------
# what kind of page this is
# --------------------------------------------------------------------------


def _holds_lines(
    geometry: LineGeometry, rows: int, thresholds: Thresholds
) -> bool:
    """Whether the profile repeats the way rows of text repeat.

    Swing alone is not enough. A photograph of sky over land swings hard across
    three broad bands and would pass a contrast test on its own, so the bands
    also have to be fine enough to repeat down the page the way lines of text
    do. Both tests are applied here so the verdict can name whichever failed.
    """
    if geometry.line_contrast < thresholds.document_line_contrast:
        return False
    if not geometry.pitch or geometry.pitch <= 0:
        return False
    return geometry.pitch * _skew.MIN_LINE_REPEATS <= rows


def classify(
    planes: PagePlanes,
    stats: PageStats,
    geometry: LineGeometry,
    thresholds: Thresholds,
) -> Tuple[str, Dict[str, float], List[str]]:
    """Decide whether this is a document, a blank sheet or a photograph.

    Blank comes first and is decided on its own: either the sheet has no
    ink-to-paper separation at all, or it has separation but almost nothing
    inked. Either way there is nothing to read.

    A document is then recognised by three tests, and a page is called a
    photograph when it fails :attr:`Thresholds.document_failed_tests` of them:

    1. most of the page is somewhere near paper white,
    2. the projection profile swings the way rows of text make it swing,
    3. the page is close enough to neutral to be ink on paper.

    Any one test can be wrong on its own - a full-page table is dark, a title
    page has few rows, a letterhead is coloured - which is why no single one
    decides. Returns the kind, the numbers behind it, and the tests that
    failed, all of which go into the report so the verdict can be argued with.
    """
    evidence = {
        "contrast": stats.contrast,
        "line_pitch": float(geometry.pitch or 0.0),
        "ink_share": stats.ink_share,
        "paper_share": stats.paper_share,
        "text_share": stats.text_share,
        "colour_spread": planes.colour,
        "line_contrast": geometry.line_contrast,
        "line_count": float(geometry.line_count),
    }
    if stats.blank:
        why = (
            "ink-to-paper contrast {0:.3f} is under {1:.2f}".format(
                stats.contrast, thresholds.blank_contrast
            )
            if stats.contrast < thresholds.blank_contrast
            else "only {0:.3%} of the page is inked".format(stats.ink_share)
        )
        return "blank", evidence, [why]

    failed: List[str] = []
    if stats.paper_share < thresholds.document_paper_share:
        failed.append(
            "only {0:.0%} of the page is near paper white, under {1:.0%}".format(
                stats.paper_share, thresholds.document_paper_share
            )
        )
    if not _holds_lines(geometry, planes.work.shape[0], thresholds):
        failed.append(
            "the profile swings {0:.2f} times its mean across bands {1} rows "
            "apart, which is not how rows of text repeat".format(
                geometry.line_contrast,
                "?" if not geometry.pitch else "{0:.0f}".format(geometry.pitch),
            )
        )
    if planes.colour > thresholds.document_colour_spread:
        failed.append(
            "colour spread {0:.2f} is over {1:.2f}, too colourful for ink on "
            "paper".format(planes.colour, thresholds.document_colour_spread)
        )
    if len(failed) >= thresholds.document_failed_tests:
        return "photograph", evidence, failed
    return "document", evidence, failed


# --------------------------------------------------------------------------
# the measures
# --------------------------------------------------------------------------

Built = Tuple[Measure, Optional[Issue]]


def _measure(
    name: str,
    value: Optional[float],
    unit: str,
    score: Optional[float],
    ok: bool,
    message: str,
    applies: bool = True,
    **details: Any,
) -> Measure:
    """Build a :class:`Measure`, keeping the call sites readable."""
    return Measure(
        name=name, value=value, unit=unit, score=score, ok=ok,
        applies=applies, message=message, details=details,
    )


def resolution(planes: PagePlanes, thresholds: Thresholds) -> Built:
    """How many dots per inch the page was scanned at.

    When no dpi is known this measure steps aside rather than guessing one
    from the pixel count: a 4000 pixel wide image is a 300 dpi letter page, a
    600 dpi receipt or a phone photograph, and nothing in the pixels says
    which.
    """
    if planes.dpi is None:
        return _measure(
            "resolution", None, "dpi", None, True,
            "No dpi in the file and none was given, so resolution advice is "
            "left out rather than guessed. Pass dpi= if you know it.",
            applies=False,
        ), None

    dpi = planes.dpi
    inches = (planes.width / dpi, planes.height / dpi)
    score = bend(dpi, thresholds.target_dpi, thresholds.limit_dpi,
                 thresholds.hopeless_dpi)
    ok = dpi >= thresholds.limit_dpi
    longest = max(inches)
    credible = thresholds.min_credible_inches <= longest <= thresholds.max_credible_inches
    message = (
        "Scanned at {0:.0f} dpi ({1}), which makes this a {2:.1f} by {3:.1f} inch "
        "sheet; {4:.0f} dpi is the least OCR wants and {5:.0f} dpi is the "
        "target.".format(
            dpi, planes.dpi_source, inches[0], inches[1],
            thresholds.limit_dpi, thresholds.target_dpi,
        )
    )
    measure = _measure(
        "resolution", dpi, "dpi", score, ok, message,
        page_inches=[inches[0], inches[1]], source=planes.dpi_source,
        credible_page_size=credible,
    )
    if not credible:
        return measure, Issue(
            "dpi_tag", "warning",
            "A dpi of {0:.0f} would make this a {1:.1f} by {2:.1f} inch sheet, "
            "which no scanner produced.".format(dpi, inches[0], inches[1]),
            "pass the real resolution as dpi= instead of trusting the file's tag",
        )
    if ok:
        return measure, None
    needed = thresholds.target_dpi
    return measure, Issue(
        "resolution", "failure",
        "Scanned at {0:.0f} dpi, and {1:.0f} dpi is the least OCR reads "
        "reliably.".format(dpi, thresholds.limit_dpi),
        "rescan at {0:.0f} dpi".format(needed),
    )


def text_size(
    planes: PagePlanes,
    geometry: LineGeometry,
    scale: float,
    thresholds: Thresholds,
) -> Tuple[Measure, Optional[Issue], Optional[float]]:
    """How tall a line of text stands, in pixels.

    Measured as the median vertical extent of the inked bands in the
    projection profile, so it reaches from the ascenders down to the
    descenders. This is the number that decides whether an OCR engine has
    enough pixels per character to work with; below roughly 16 it does not.

    ``scale`` maps native pixels to the plane ``geometry`` was measured on, so
    the plane-pixel height divided by it is the height on the page itself.

    Returns the measure, any issue, and the height in native pixels so the
    report and the sharpness measure can both use it.
    """
    if geometry.text_height is None or scale <= 0:
        return _measure(
            "text_size", None, "px", None, True,
            "No rows of text were found, so there is no text height to report.",
            applies=False,
        ), None, None

    height = geometry.text_height / scale
    pitch = None if geometry.pitch is None else geometry.pitch / scale
    score = bend(height, thresholds.target_text_height_px,
                 thresholds.limit_text_height_px,
                 thresholds.hopeless_text_height_px)
    ok = height >= thresholds.limit_text_height_px
    measure = _measure(
        "text_size", height, "px", score, ok,
        "Text lines stand about {0:.0f} px tall across {1} row(s); {2:.0f} px is "
        "the least OCR reads reliably.".format(
            height, geometry.line_count, thresholds.limit_text_height_px
        ),
        line_pitch_px=pitch, line_count=geometry.line_count,
    )
    if ok:
        return measure, None, height
    if planes.dpi:
        needed = planes.dpi * thresholds.target_text_height_px / max(height, 1e-6)
        fix = "rescan at {0:.0f} dpi".format(25.0 * np.ceil(needed / 25.0))
    else:
        fix = (
            "rescan at a higher resolution, enough to make text lines at least "
            "{0:.0f} px tall instead of {1:.0f}".format(
                thresholds.target_text_height_px, height
            )
        )
    return measure, Issue(
        "text_size", "failure",
        "Text lines are only {0:.0f} px tall, and {1:.0f} px is the least OCR "
        "reads reliably.".format(height, thresholds.limit_text_height_px),
        fix,
    ), height


def skew(degrees: float, thresholds: Thresholds) -> Built:
    """How far the text runs off horizontal, in degrees."""
    amount = abs(degrees)
    score = bend(amount, thresholds.target_skew_degrees,
                 thresholds.limit_skew_degrees, thresholds.hopeless_skew_degrees)
    ok = amount <= thresholds.limit_skew_degrees
    if amount < 0.005:
        run = "Text runs level with the horizontal"
    else:
        run = "Text runs {0:.2f} degrees {1} of horizontal".format(
            amount, "counter-clockwise" if degrees > 0 else "clockwise"
        )
    measure = _measure(
        "skew", degrees, "degrees", score, ok,
        "{0}; {1:.1f} degrees is as far as OCR tolerates.".format(
            run, thresholds.limit_skew_degrees
        ),
    )
    if ok:
        return measure, None
    return measure, Issue(
        "skew", "warning" if amount < 2.0 * thresholds.limit_skew_degrees
        else "failure",
        "The page is turned {0:.2f} degrees {1} of horizontal.".format(
            amount, direction
        ),
        "deskew by {0:.1f} degrees".format(abs(degrees)),
    )


def contrast(stats: PageStats, thresholds: Thresholds) -> Built:
    """How far the ink sits from the paper, on a 0 to 1 luminance scale."""
    value = stats.contrast
    score = bend(value, thresholds.target_contrast, thresholds.limit_contrast,
                 thresholds.hopeless_contrast)
    ok = value >= thresholds.limit_contrast
    measure = _measure(
        "contrast", value, "0-1", score, ok,
        "Ink sits {0:.2f} below paper on a 0 to 1 scale (ink {1:.2f}, paper "
        "{2:.2f}); {3:.2f} is the least that keeps strokes separate.".format(
            value, stats.ink_level, stats.paper_level, thresholds.limit_contrast
        ),
        ink_level=stats.ink_level, paper_level=stats.paper_level,
    )
    if ok:
        return measure, None
    return measure, Issue(
        "contrast", "failure",
        "Ink is only {0:.2f} darker than paper, and {1:.2f} is the least that "
        "keeps strokes apart from the page.".format(value, thresholds.limit_contrast),
        "raise the scanner contrast, or rescan in greyscale instead of colour",
    )


def sharpness(
    stats: PageStats,
    text_height_px: Optional[float],
    thresholds: Thresholds,
) -> Built:
    """How crisply ink turns into paper, measured against the size of the text.

    The raw quantity is the strongest edge step divided by the page's own
    ink-to-paper contrast, which is one over the width of an edge in pixels.
    That is then scaled by the height of the text, because a 600 dpi scan is
    expected to spread the same physical edge over twice as many pixels as a
    300 dpi one and should not be marked down for it.
    """
    if stats.edge_step is None or stats.contrast <= 0:
        return _measure(
            "sharpness", None, "0-1", None, True,
            "There is no ink on this page, so there are no edges to measure.",
            applies=False,
        ), None

    acutance = stats.edge_step / max(stats.contrast, 1e-6)
    scale = 1.0
    if text_height_px:
        scale = text_height_px / REFERENCE_TEXT_HEIGHT_PX
    value = float(np.clip(acutance * scale, 0.0, 2.0))
    edge_px = 1.0 / max(acutance, 1e-6)
    score = bend(value, thresholds.target_sharpness, thresholds.limit_sharpness,
                 thresholds.hopeless_sharpness)
    ok = value >= thresholds.limit_sharpness
    measure = _measure(
        "sharpness", value, "0-1", score, ok,
        "Ink reaches paper over about {0:.1f} px{1}; {2:.2f} is the least "
        "crispness OCR separates strokes at.".format(
            edge_px,
            "" if not text_height_px
            else " against {0:.0f} px text".format(text_height_px),
            thresholds.limit_sharpness,
        ),
        edge_width_px=edge_px, raw_acutance=acutance,
        normalised_to_text=bool(text_height_px),
    )
    if ok:
        return measure, None
    return measure, Issue(
        "sharpness", "failure",
        "Strokes are soft: ink takes about {0:.1f} px to reach paper.".format(edge_px),
        "rescan with the page flat on the glass and the lid closed, or refocus "
        "the camera",
    )


def lighting(stats: PageStats, thresholds: Thresholds) -> Built:
    """How much the paper level swings across the page."""
    value = stats.lighting_swing
    score = bend(value, thresholds.target_lighting, thresholds.limit_lighting,
                 thresholds.hopeless_lighting)
    ok = value <= thresholds.limit_lighting
    measure = _measure(
        "lighting", value, "0-1", score, ok,
        "The paper level swings {0:.0%} across the page, darkest at the {1}; "
        "{2:.0%} is as uneven as OCR tolerates.".format(
            value, stats.dark_zone, thresholds.limit_lighting
        ),
        dark_zone=stats.dark_zone,
    )
    if ok:
        return measure, None
    return measure, Issue(
        "lighting", "warning" if value < 2.0 * thresholds.limit_lighting else "failure",
        "The page is lit unevenly: the paper level swings {0:.0%} between its "
        "brightest and darkest parts.".format(value),
        "increase lighting on the {0}, or lay the page flat so it does not "
        "curl away from the light".format(stats.dark_zone),
    )


def show_through(stats: PageStats, thresholds: Thresholds) -> Built:
    """How much of the reverse side is showing through the sheet.

    Marks printed on this side have hard edges; marks bleeding through from
    the back are soft and pale. This counts pale, soft, mid-grey pixels that
    are not close enough to real ink to be its antialiasing.
    """
    value = stats.show_through
    score = bend(value, thresholds.target_show_through,
                 thresholds.limit_show_through, thresholds.hopeless_show_through)
    ok = value <= thresholds.limit_show_through
    measure = _measure(
        "show_through", value, "share", score, ok,
        "{0:.1%} of the page carries soft grey marks from the reverse side; "
        "{1:.1%} is as much as OCR tolerates.".format(
            value, thresholds.limit_show_through
        ),
    )
    if ok:
        return measure, None
    return measure, Issue(
        "show_through", "warning" if value < 3.0 * thresholds.limit_show_through
        else "failure",
        "{0:.1%} of the page shows print from the reverse side.".format(value),
        "put a sheet of black paper behind the page and rescan",
    )


def clipping(stats: PageStats, thresholds: Thresholds) -> Built:
    """How much of the page has been crushed to black or blown to white.

    Black clipping is scored, because a stroke flattened to solid black has
    lost the shape an OCR engine matches on. White clipping is only ever
    warned about: a bilevel scan is almost entirely pure white and reads
    perfectly well, so pure white alone proves nothing.
    """
    black, white = stats.black_clipping, stats.white_clipping
    score = bend(black, thresholds.target_black_clipping,
                 thresholds.limit_black_clipping,
                 thresholds.hopeless_black_clipping)
    ok = black <= thresholds.limit_black_clipping
    measure = _measure(
        "clipping", black, "share", score, ok,
        "{0:.1%} of the page is crushed to pure black and {1:.1%} is blown to "
        "pure white; {2:.0%} black is where strokes start losing their "
        "shape.".format(black, white, thresholds.limit_black_clipping),
        white_clipping=white,
    )
    if not ok:
        return measure, Issue(
            "clipping", "warning",
            "{0:.1%} of the page is solid black, so stroke shapes have been "
            "flattened away.".format(black),
            "lower the scanner contrast or brightness and rescan",
        )
    if (white >= thresholds.warn_white_clipping
            and stats.text_share < thresholds.faint_text_coverage):
        return measure, Issue(
            "clipping", "warning",
            "{0:.1%} of the page is pure white with almost nothing on it, so "
            "faint content may have been erased by the white point.".format(white),
            "lower the scanner brightness and rescan to see whether anything "
            "faint appears",
        )
    return measure, None


def text_coverage(stats: PageStats) -> Measure:
    """How much of the page looks like text.

    Ink that belongs to thin, edge-bounded strokes, as a share of the page.
    Reported rather than scored: a dense legal page and a title page are both
    perfectly OCR-able and sit at opposite ends of this number. It is here
    because it is the quantity behind the blank and photograph verdicts.
    """
    return _measure(
        "text_coverage", stats.text_share, "share", None, True,
        "{0:.1%} of the page is ink in thin strokes, out of {1:.1%} inked "
        "altogether.".format(stats.text_share, stats.ink_share),
        ink_share=stats.ink_share, paper_share=stats.paper_share,
    )
