"""What is actually measured on a page, and the fix each measurement implies.

Every function here returns a :class:`~document_quality._report.Measure` and,
when something is wrong, the :class:`~document_quality._report.Issue` that goes
with it. Keeping the two together is deliberate: the remedy belongs next to the
measurement that motivates it, so a threshold can never drift away from the
advice it produces.

Three ideas hold the file together.

**Paper is local.** A page lit from one side has no single paper level, so the
paper level is estimated everywhere on the page (see
:mod:`document_quality._lighting`) and the page is divided by it before any ink
is measured - a flat-field correction, which is what uneven light physically
calls for, because light multiplies ink and paper alike. That is why a shadow
along one edge is reported as a lighting gradient with a lighting fix, and not
as low contrast, as show-through, or as a scanner border. Only genuinely black
bands running in from the edge of the image are borders, and those are cropped
away before anything else is measured.

**Sharpness is measured against the size of the text.** Optical softness is
roughly fixed in physical terms, so the same lens spreads an edge over twice as
many pixels at 600 dpi as at 300. Judging edges by pixels alone would punish
the better scan. The crispness of an edge is therefore scaled by the height of
the text it belongs to, so two scans of one sheet at different resolutions come
out level.

**A page with nothing on it is not a broken page.** A blank sheet deserves one
honest sentence rather than eight complaints about the text it does not have.
But faint is not the same as blank: a pale pencil page has almost no contrast
and still carries rows of text, and telling its owner to skip it would lose
the page. So low contrast alone never makes a page blank - it also has to have
no rows of text in its projection profile.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import _images, _lighting, _skew
from ._images import PagePlanes, percentile_sample
from ._lighting import Border, PaperSurface
from ._report import Issue, Measure
from ._skew import LineGeometry
from ._thresholds import Thresholds, bend

logger = logging.getLogger(__name__)

#: Percentile of the flattened work plane taken as the page's ink level. A page
#: with as little as one line of text has more inked pixels than this: one
#: line across a letter page inks about 0.4% of it, so a higher percentile
#: would read a sparse page's ink level off bare paper and call it blank.
INK_PERCENTILE = 0.1
#: Percentile taken as the page's paper level. Higher than the ink side's
#: mirror image, so a blown highlight or a speck of glare cannot set it.
PAPER_PERCENTILE = 99.5
#: Ink-to-paper separation below which there is nothing to normalise at all:
#: a sheet this flat is settled as blank without building any mask.
MIN_MEASURABLE_CONTRAST = 0.01
#: Normalised level below which a pixel counts as ink rather than paper.
INK_LEVEL = 0.5
#: Normalised level above which a pixel counts as clean paper.
PAPER_LEVEL = 0.85
#: Normalised band in which show-through lives: darker than paper, far lighter
#: than ink. The top of the band is an eighth below paper, where print
#: bleeding through the sheet is plainly visible to anyone holding it.
SHOW_THROUGH_BAND = (0.25, 0.88)
#: Long edge of the plane show-through is measured on. Twice the work plane,
#: so a ghost stroke a few native pixels wide is still a stroke.
SHOW_THROUGH_LONG_EDGE = 2048
#: Gradient, in normalised units per *native* pixel, below which a mark is
#: soft enough to have come through the sheet rather than been printed on it.
#: It is measured on the work plane, where one pixel spans several native
#: ones, so the limit is scaled up by that factor - capped at
#: :data:`SHOW_THROUGH_MAX_STEP` so a printed edge can never pass for soft.
SHOW_THROUGH_SMOOTHNESS = 0.10
SHOW_THROUGH_MAX_STEP = 0.35
#: How far from real ink a soft grey mark must sit to count as show-through,
#: in pixels of the show-through plane. Antialiasing around a stroke is not
#: show-through. The clearance grows with the measured edge width, so the
#: grey halo around the strokes of an out-of-focus page is not counted as
#: print from the back.
SHOW_THROUGH_CLEARANCE = 2
#: Most the clearance is allowed to grow to, in pixels of that plane.
SHOW_THROUGH_MAX_CLEARANCE = 20
#: Native-resolution tiles sampled for sharpness, and how big each one is.
SHARPNESS_TILES = 24
SHARPNESS_TILE_PX = 256
#: A tile is sampled only when its inked share is at least this share of the
#: most-inked tile's, and at least :data:`SHARPNESS_MIN_TILE_INK` outright.
SHARPNESS_TILE_SHARE = 0.25
SHARPNESS_MIN_TILE_INK = 0.002
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
#: The ink-is-text-shaped document test fails when less than this share of the
#: ink lies in thin, edge-bounded strokes. Type of any weight measures 0.8 and
#: up, heavily blurred type about 0.35, word-shaped blocks 0.2; a solid black
#: shape measures a few per cent.
MIN_STROKE_SHARE = 0.12
#: It also fails when more than this share of the page is inked at all: the
#: densest page of type inks well under a third of the sheet.
MAX_TEXT_INK_SHARE = 0.5


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


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow ``mask`` by a ``2 * radius + 1`` square, in time independent of it.

    A running count along each axis says whether any set pixel lies within
    the window, so a wide clearance costs the same as a narrow one.
    """
    radius = int(radius)
    if radius <= 0 or not mask.any():
        return mask.copy()
    grown = mask
    for axis in (0, 1):
        length = grown.shape[axis]
        counts = np.cumsum(grown, axis=axis, dtype=np.int32)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        counts = np.pad(counts, pad)
        upper = np.minimum(np.arange(length) + radius + 1, length)
        lower = np.maximum(np.arange(length) - radius, 0)
        if axis == 0:
            grown = (counts[upper, :] - counts[lower, :]) > 0
        else:
            grown = (counts[:, upper] - counts[:, lower]) > 0
    return grown


def tile_spans(length: int, tile: int) -> List[Tuple[int, int]]:
    """Cover ``length`` with spans of ``tile``, the last one nudged to fit."""
    if length <= tile:
        return [(0, length)]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(start, start + tile) for start in starts]


# --------------------------------------------------------------------------
# the shared statistics
# --------------------------------------------------------------------------


@dataclass
class PageStats:
    """Everything the measures share, computed once per page."""

    #: Luminance of the darkest real ink on the page, 0 to 1, after the page
    #: has been divided by its own paper surface.
    ink_level: float
    #: Luminance of clean paper, 0 to 1, on the same flattened page.
    paper_level: float
    #: Ink-to-paper separation, 0 to 1.
    contrast: float
    #: The paper surface and what it says about the light.
    light: PaperSurface
    #: The work plane mapped to 0 at ink and 1 at the local paper level.
    normalised: np.ndarray
    #: Share of the page that is ink.
    ink_share: float
    #: Share of the page, as scanned, that is near its brightest paper.
    paper_share: float
    #: Share of the page that is ink belonging to thin, edge-bounded strokes.
    text_share: float
    #: Paper-level fall from brightest to darkest, relative to the brightest.
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
    #: True when the sheet has too little contrast or too little ink to hold
    #: anything. Provisional: :func:`classify` still asks whether the profile
    #: shows rows of text, because a faint page is not a blank one.
    blank: bool = False
    #: Genuinely black scanner border cropped off before measuring.
    border: Border = field(default_factory=Border)


def turn_stats(stats: PageStats, quarter_turns: int) -> PageStats:
    """``stats`` for the same page turned ``quarter_turns`` x 90 degrees.

    Every number here is blind to a quarter turn - contrast, ink, paper,
    clipping, show-through, how steeply the light falls - except where the
    dark side is, so the maps are turned and the dark side named again rather
    than the whole page measured a second time.
    """
    k = int(quarter_turns) % 4
    if k == 0:
        return stats
    light = stats.light.turned(k)
    return dataclasses.replace(
        stats,
        light=light,
        normalised=np.ascontiguousarray(np.rot90(stats.normalised, k)),
        dark_zone=light.dark_zone,
    )


def _clipping(lum: np.ndarray) -> Tuple[float, float]:
    """Share of native pixels crushed to 0 and blown to 255."""
    size = float(lum.size) or 1.0
    return (
        float(np.count_nonzero(lum == 0)) / size,
        float(np.count_nonzero(lum == 255)) / size,
    )


def _edge_step(
    planes: PagePlanes, normalised: np.ndarray, light: Optional[PaperSurface] = None
) -> Optional[float]:
    """Strongest typical edge on the page, in luminance per pixel.

    Measured at native resolution, because resizing an image changes exactly
    the thing this measures, and only on tiles that actually carry ink, so the
    number describes strokes rather than empty margins. A tile counts when its
    inked share is at least :data:`SHARPNESS_TILE_SHARE` of the most-inked
    tile's: a cover page with three lines of text has three or four such
    tiles, not twenty-four, and taking the most-inked twenty-four regardless
    would put the median on bare paper and call a crisp page soft. The median
    across the tiles that qualify is taken so one speck of dust cannot stand
    in for the whole page.
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

    inked = normalised < INK_LEVEL
    if light is not None and light.black.any():
        # Light too deep for the surface to call paper leaves dim paper that
        # reads as ink; a tile of it has no strokes, only a shadow.
        inked &= ~(_lighting.resample(light.black.astype(np.float32), inked.shape) > 0.0)

    def ink_in(span: Tuple[int, int, int, int]) -> float:
        top, bottom, left, right = span
        y0 = min(plane_height - 1, int(top * scale))
        y1 = max(y0 + 1, min(plane_height, int(round(bottom * scale))))
        x0 = min(plane_width - 1, int(left * scale))
        x1 = max(x0 + 1, min(plane_width, int(round(right * scale))))
        return float(inked[y0:y1, x0:x1].mean())

    shares = [ink_in(span) for span in spans]
    order = sorted(range(len(spans)), key=lambda index: -shares[index])
    most = shares[order[0]] if order else 0.0
    if most <= 0.0:
        return None
    floor = max(SHARPNESS_MIN_TILE_INK, SHARPNESS_TILE_SHARE * most)
    chosen = [spans[index] for index in order if shares[index] >= floor]
    steps = []
    for top, bottom, left, right in chosen[:SHARPNESS_TILES]:
        block = lum[top:bottom, left:right].astype(np.float32) / np.float32(255.0)
        if block.size < 16:             # pragma: no cover - guarded by tile size
            continue
        steps.append(float(np.percentile(gradient(block), EDGE_PERCENTILE)))
    if not steps:                       # pragma: no cover - needs a 3-pixel image
        return None
    return float(np.median(steps))


def _paper_share(plane: np.ndarray) -> float:
    """Share of ``plane`` within :data:`PAPER_LEVEL` of its own paper level."""
    ink, paper = _levels(plane)
    near = plane > np.float32(ink + PAPER_LEVEL * max(paper - ink, 1e-6))
    return float(np.count_nonzero(near)) / float(max(plane.size, 1))


def _show_through(
    planes: PagePlanes,
    light: PaperSurface,
    ink_level: float,
    contrast: float,
    edge_step: Optional[float],
    work_normalised: Optional[np.ndarray] = None,
) -> float:
    """Share of the page carrying soft, pale marks well away from real ink.

    Measured on its own plane, :data:`SHOW_THROUGH_LONG_EDGE` pixels on the
    long side, rather than the work plane. Print bleeding through from the
    back of a letter page is a few pixels wide; the work plane averages it
    with the paper around it until it is barely darker than paper, and a
    plainly visible ghost of the reverse side measures as nothing. The plane
    is flattened by the same paper surface and put on the same ink-to-paper
    scale as the rest of the measures.
    """
    longest = max(planes.lum.shape)
    finer = min(1.0, SHOW_THROUGH_LONG_EDGE / float(max(longest, 1)))
    if work_normalised is not None and finer < 1.7 * planes.work_scale:
        # A page small enough that the finer plane would add little over the
        # work plane - under twice its resolution: measure on the work plane.
        normalised, scale = work_normalised, planes.work_scale
    else:
        plane, scale = _images.downscale_plane(planes.lum, SHOW_THROUGH_LONG_EDGE)
        if plane.size == 0:             # pragma: no cover - guarded by prepare
            return 0.0
        surface = light.surface_for(plane.shape)
        flat = plane / np.maximum(surface, np.float32(1e-3))
        flat = np.clip(flat * np.float32(light.paper_level), 0.0, 1.0)
        normalised = np.clip(
            (flat - np.float32(ink_level)) / np.float32(max(contrast, 1e-6)), 0.0, 1.0
        )
    clearance = SHOW_THROUGH_CLEARANCE
    if edge_step is not None and edge_step > 0:
        edge_px = contrast / edge_step * scale
        clearance = int(np.clip(np.ceil(edge_px) + 1, SHOW_THROUGH_CLEARANCE,
                                SHOW_THROUGH_MAX_CLEARANCE))
    low, high = SHOW_THROUGH_BAND
    soft_grey = (normalised > low) & (normalised < high)
    smooth_limit = min(
        SHOW_THROUGH_MAX_STEP, SHOW_THROUGH_SMOOTHNESS / max(scale, 1e-6)
    )
    soft_grey &= gradient(normalised) < np.float32(smooth_limit)
    if soft_grey.any():
        inked = normalised < low
        if clearance <= 4:
            soft_grey &= ~dilate(inked, clearance)
        else:
            # A wide clearance - a soft page - as a square of three quarters
            # the radius, which keeps about the same area clear as the
            # diamond it stands for at a cost that does not grow with it.
            soft_grey &= ~dilate_square(inked, int(round(0.75 * clearance)))
    if soft_grey.any() and light.black.any():
        # Where the light is so deep that the surface gave up on it as paper,
        # the page reads as flat mid-grey: dim paper, not print from the back.
        black = _lighting.resample(
            _lighting.max_filter(light.black.astype(np.float32), 1), normalised.shape
        ) > 0.0
        soft_grey &= ~black
    return float(np.count_nonzero(soft_grey)) / float(normalised.size)


def _levels(plane: np.ndarray) -> Tuple[float, float]:
    """The ink and paper levels of ``plane``: its low and high percentiles.

    Both come from one partition of the data rather than two, which on a
    megapixel plane is most of what an extra percentile costs.
    """
    ink, paper = np.percentile(
        percentile_sample(plane), [INK_PERCENTILE, PAPER_PERCENTILE]
    )
    return float(ink), float(paper)


def analyse(
    planes: PagePlanes,
    thresholds: Thresholds,
    border: Optional[Border] = None,
) -> PageStats:
    """Measure everything the individual measures share, once.

    ``planes`` should already be cropped to the inside of any scanner border;
    ``border`` is carried through so the report can say what was cropped.

    A sheet with no measurable ink-to-paper separation at all is settled here
    and returned with ``blank`` set, because every mask below it would be a
    division by nothing. A merely faint sheet is measured in full and flagged,
    and :func:`classify` decides.
    """
    work = planes.work
    light = _lighting.paper_surface(work)
    # Flat-field: divide by the local paper level, then put the page back on
    # the scale of its own paper. A shadowed corner comes out as bright as the
    # rest, with its ink exactly as dark relative to it as it was.
    flat = work / np.maximum(light.surface, np.float32(1e-3))
    flat = np.clip(flat * np.float32(light.paper_level), 0.0, 1.0)

    ink_level, paper_level = _levels(flat)
    contrast = max(0.0, paper_level - ink_level)
    black_clipping, white_clipping = _clipping(planes.lum)
    border = border if border is not None else Border()

    if contrast < MIN_MEASURABLE_CONTRAST:
        return PageStats(
            ink_level=ink_level, paper_level=paper_level, contrast=contrast,
            light=light, normalised=np.ones_like(work), ink_share=0.0,
            paper_share=1.0, text_share=0.0, lighting_swing=light.swing,
            dark_zone=light.dark_zone, show_through=0.0,
            black_clipping=black_clipping, white_clipping=white_clipping,
            edge_step=None, blank=True, border=border,
        )

    normalised = np.clip(
        (flat - np.float32(ink_level)) / np.float32(max(contrast, 1e-6)), 0.0, 1.0
    )

    ink_mask = normalised < INK_LEVEL
    # Paper share is never read off the page divided by its full surface: that
    # would rescue a shadowed page but also turn the smooth tones of a
    # photograph into something that looks like evenly lit paper, and this is
    # one of the numbers that tells the two apart. It is read off the page as
    # scanned, and off the page divided by the light alone - a smooth,
    # low-order fit through the paper surface - and the better of the two is
    # kept. As scanned, a strong ramp of light across a page of text leaves
    # only its bright side near paper white; the fit evens out the ramp and
    # leaves a picture's tones where they were.
    paper_share = _paper_share(work)
    if light.model is not None:
        sample = percentile_sample(work)
        stride = max(1, work.shape[0] // max(sample.shape[0], 1))
        rows = np.minimum(np.arange(sample.shape[0]) * stride // _lighting.BLOCK_PX,
                          light.model.shape[0] - 1)
        columns = np.minimum(np.arange(sample.shape[1]) * stride // _lighting.BLOCK_PX,
                             light.model.shape[1] - 1)
        lit = sample / light.model[rows][:, columns] * np.float32(light.paper_level)
        paper_share = max(paper_share, _paper_share(lit))
    ink_share = float(np.count_nonzero(ink_mask)) / normalised.size
    blank = (
        contrast < thresholds.blank_contrast
        or ink_share < thresholds.blank_ink_share
    )

    step = gradient(normalised)
    strongest = (
        float(np.percentile(percentile_sample(step), EDGE_PERCENTILE)) if step.size else 0.0
    )
    edge_mask = step > max(EDGE_FLOOR, EDGE_SHARE * strongest)
    text_mask = ink_mask & dilate(edge_mask, 1)
    text_share = float(np.count_nonzero(text_mask)) / normalised.size

    edge_step = _edge_step(planes, normalised, light) if ink_share > 0.0 else None
    show_through = _show_through(planes, light, ink_level, contrast, edge_step, normalised)

    return PageStats(
        ink_level=ink_level, paper_level=paper_level, contrast=contrast,
        light=light, normalised=normalised, ink_share=ink_share,
        paper_share=paper_share, text_share=text_share,
        lighting_swing=light.swing, dark_zone=light.dark_zone,
        show_through=show_through, black_clipping=black_clipping,
        white_clipping=white_clipping, edge_step=edge_step, blank=blank,
        border=border,
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
    do. And twenty specks of dust down a blank sheet make a spiky, repeating
    profile too, so each band also has to carry ink across the page the way a
    line of words does. All three tests are applied here so the verdict can
    name whichever failed.
    """
    if geometry.line_contrast < thresholds.document_line_contrast:
        return False
    if not geometry.pitch or geometry.pitch <= 0:
        return False
    if geometry.band_coverage < _skew.MIN_BAND_COVERAGE:
        return False
    return geometry.pitch * _skew.MIN_LINE_REPEATS <= rows


def settled_without_lines(
    planes: PagePlanes, stats: PageStats, thresholds: Thresholds
) -> bool:
    """True when the tone tests alone already make this a photograph.

    Three of :func:`classify`'s four document tests need no projection
    profile: how much of the page is near paper white, how colourful it is,
    and whether its ink is text-shaped. When enough of those already fail,
    the line test cannot rescue the page, so the caller can skip the
    orientation search that would feed it - which is the most expensive thing
    done to a photograph.
    """
    if stats.blank:
        return False
    failed = int(stats.paper_share < thresholds.document_paper_share)
    failed += int(planes.colour > thresholds.document_colour_spread)
    failed += int(not _ink_is_text_shaped(stats))
    return failed >= thresholds.document_failed_tests


def _ink_is_text_shaped(stats: PageStats) -> bool:
    """Thin strokes covering well under half the page, as type does."""
    if stats.ink_share > MAX_TEXT_INK_SHARE:
        return False
    if stats.ink_share <= 0.0:
        return True
    return stats.text_share / stats.ink_share >= MIN_STROKE_SHARE


def classify(
    planes: PagePlanes,
    stats: PageStats,
    geometry: LineGeometry,
    thresholds: Thresholds,
) -> Tuple[str, Dict[str, float], List[str]]:
    """Decide whether this is a document, a blank sheet or a photograph.

    Blank comes first: either the sheet has almost no ink-to-paper separation,
    or it has separation but almost nothing inked, and in both cases the
    projection profile shows no rows of text. The last condition is what keeps
    a faint pencil page from being skipped as empty; it is reported as a
    document with a contrast problem instead.

    A document is then recognised by four tests, and a page is called a
    photograph when it fails :attr:`Thresholds.document_failed_tests` of them:

    1. most of the page is somewhere near paper white,
    2. the projection profile swings the way rows of text make it swing,
    3. the page is close enough to neutral to be ink on paper,
    4. the ink is text-shaped: thin strokes, covering well under half the
       page, rather than solid shapes or broad bands of tone.

    Any one test can be wrong on its own - a full-page table is dark, a title
    page has few rows, a letterhead is coloured, a heading in heavy type is
    mostly solid ink - which is why no single one decides. Returns the kind, the numbers behind it, and the tests that
    failed, all of which go into the report so the verdict can be argued with.
    """
    evidence = {
        "stroke_share": stats.text_share / max(stats.ink_share, 1e-9),
        "contrast": stats.contrast,
        "line_pitch": float(geometry.pitch or 0.0),
        "ink_share": stats.ink_share,
        "paper_share": stats.paper_share,
        "text_share": stats.text_share,
        "colour_spread": planes.colour,
        "line_contrast": geometry.line_contrast,
        "line_count": float(geometry.line_count),
    }
    lines = _holds_lines(geometry, planes.work.shape[0], thresholds)
    if stats.blank and not lines:
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
    if not lines:
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
    stroked = stats.text_share / max(stats.ink_share, 1e-9)
    if stats.ink_share > MAX_TEXT_INK_SHARE:
        failed.append(
            "{0:.0%} of the page is inked, and text never covers more than "
            "{1:.0%}".format(stats.ink_share, MAX_TEXT_INK_SHARE)
        )
    elif stats.ink_share > 0.0 and stroked < MIN_STROKE_SHARE:
        failed.append(
            "only {0:.0%} of the ink is in thin strokes, so it is solid shapes "
            "rather than letters".format(stroked)
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
    document: bool = False,
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

    ``document`` says the page has been judged a document page. A document
    page with no rows of text on it at all - a thumbnail, a sliver, a single
    black shape - has nothing an OCR engine could read, and must not pass as
    ready with a perfect score just because every other measure had nothing
    to complain about. It scores 0 here and fails.
    """
    if geometry.text_height is None or scale <= 0:
        if document:
            return _measure(
                "text_size", None, "px", 0.0, False,
                "No rows of text were found on this page, so there is nothing "
                "an OCR engine could read.",
            ), Issue(
                "text_size", "failure",
                "No rows of text were found on this page, so there is nothing "
                "here an OCR engine could read.",
                "check that this is the printed side of a page of text, and that "
                "the scan shows the whole page rather than a corner of it",
            ), None
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


def skew(degrees: float, thresholds: Thresholds, applies: bool = True) -> Built:
    """How far the text runs off horizontal, in degrees.

    ``applies`` is ``False`` when no rows of text were found, in which case
    there is no angle to report and nothing is scored.
    """
    if not applies:
        return _measure(
            "skew", None, "degrees", None, True,
            "No rows of text were found, so there is no skew to measure.",
            applies=False,
        ), None
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
            amount, "counter-clockwise" if degrees > 0 else "clockwise"
        ),
        "deskew by {0:.1f} degrees {1}".format(
            amount, "clockwise" if degrees > 0 else "counter-clockwise"
        ),
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
    """How far the paper level falls across the page, and towards which side.

    This is a gradient, not a darkness test: the paper surface is estimated
    everywhere and the measure is how far it falls from its brightest part to
    its darkest, as a share of the brightest. Genuinely black regions are not
    paper and are left out, so a scanner border cannot pose as a shadow here
    any more than a shadow can pose as a border there.
    """
    light = stats.light
    value = stats.lighting_swing
    score = bend(value, thresholds.target_lighting, thresholds.limit_lighting,
                 thresholds.hopeless_lighting)
    ok = value <= thresholds.limit_lighting
    measure = _measure(
        "lighting", value, "0-1", score, ok,
        "The paper level falls {0:.0%} from the brightest part of the page to "
        "the darkest{1}; {2:.0%} is as uneven as OCR tolerates.".format(
            value,
            "" if value < thresholds.target_lighting
            else ", which is towards the {0}".format(stats.dark_zone),
            thresholds.limit_lighting,
        ),
        dark_zone=stats.dark_zone, brightest=light.brightest,
        darkest=light.darkest, slope_across=light.slope_across,
        slope_down=light.slope_down,
    )
    if ok:
        return measure, None
    return measure, Issue(
        "lighting", "warning" if value < 2.0 * thresholds.limit_lighting else "failure",
        "The page is lit unevenly: the paper level falls {0:.0%} towards the {1}."
        .format(value, stats.dark_zone),
        "increase lighting on the {0}, or lay the page flat so it does not "
        "curl away from the light".format(stats.dark_zone),
    )


def border(stats: PageStats) -> Built:
    """Whether a genuinely black scanner border runs in from the image edges.

    Reported rather than scored: the border has already been cropped off
    before anything else was measured, so it costs nothing here, but an OCR
    engine handed the uncropped image will read it as a column of junk.
    """
    found = stats.border
    if not found.found:
        return _measure(
            "border", 0.0, "share", None, True,
            "No black scanner border runs in from the edges of the image.",
        ), None
    where = found.describe()
    measure = _measure(
        "border", found.area_share, "share", None, False,
        "A black scanner border covers {0:.1%} of the image along the {1} "
        "side(s); it was cropped off before measuring the page.".format(
            found.area_share, where
        ),
        top=found.top, bottom=found.bottom, left=found.left, right=found.right,
        sides=list(found.sides),
    )
    edges = ", ".join(
        "{0} px off the {1} edge".format(getattr(found, side), side)
        for side in found.sides
    )
    return measure, Issue(
        "border", "warning",
        "A black scanner border runs along the {0} side(s) of the image, and "
        "OCR engines read black bands as junk characters.".format(where),
        "crop the black border before OCR: {0}".format(edges),
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
