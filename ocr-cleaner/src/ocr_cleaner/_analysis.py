"""Everything the pipeline needs to know before it changes a pixel.

Three questions get answered here, and all three come out of one projection
profile:

1. **How far is the text turned?** Lines of text are the only strong periodic
   structure on a page. Shear the ink by a candidate angle, sum each row, and
   the profile becomes a comb: tall teeth where lines are, near-zero between
   them. The comb is sharpest at exactly one angle, and that angle is the skew.
2. **How tall is a line of text?** The teeth of that comb, measured at the angle
   that sharpened them. This is the number every later step is sized from - the
   median window, the adaptive threshold window, the upscale target.
3. **What kind of page is this?** Blank paper, a photograph, or a document. A
   blank sheet must not be thresholded into a field of noise, and a photograph
   must not be binarised at all, so both have to be recognised before anything
   is applied.

Nothing is rotated or resampled to measure an angle. Shearing by row offsets is
both faster than bicubic rotation and more honest here, because rotation smears
the very edges being counted.

Angles follow ``PIL.Image.rotate``: positive is counter-clockwise, so a page
whose text runs downhill to the right has a negative skew, and
``image.rotate(-estimate_skew(image))`` puts it straight.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import _images

logger = logging.getLogger(__name__)

#: Widest skew the search looks at, in degrees either side of upright. Beyond
#: this a page is not skewed, it is turned, which is a different repair.
MAX_SKEW_DEGREES = 15.0
#: The search, stage by stage: ``(strip width in columns, half-width of the
#: window in degrees, step in degrees, use the smooth shear)``. ``0`` columns
#: means the whole page in one piece. Each stage widens the aperture - which
#: sharpens the peak - and narrows the window around what the last stage found.
#: :func:`_search` explains why the aperture is the thing that matters.
SEARCH_STAGES = (
    (96, MAX_SKEW_DEGREES, 1.0, False),
    (288, 1.1, 0.2, True),
    (0, 0.28, 0.035, True),
)
#: Narrowest strip worth shearing on its own.
MIN_STRIP_COLUMNS = 24
#: Longest side the angle search itself runs on. Not a free choice: the comb
#: the search looks for only exists while the plane still has a few rows per
#: line of text, and a 300 dpi page of six-point type reduced to 800 pixels is
#: down to four. Reduced to 500 it is down to two and a half, and the measured
#: angle starts coming back degrees wrong rather than hundredths. Going the
#: other way buys nothing, so this sits where the margin is comfortable and the
#: cost is not. Line height is measured separately on the full analysis plane,
#: at the angle the search found.
SEARCH_MAX_SIDE = 800
#: Smallest plane side worth measuring lines on.
MIN_PLANE_SIDE = 24
#: Share of the profile swing that separates a line of text from the gap above.
LINE_THRESHOLD = 0.35
#: Where the cut for the full inked extent sits, as a share of a line's own
#: height above the paper between lines. It is deliberately close to the paper,
#: because only a handful of glyphs in a line carry an ascender or a descender:
#: measured on rendered Times and Arial from 10 to 60 pixels, those rows sum to
#: between 6 and 20 percent of an x-height row. A cut set anywhere inside that
#: band chops the tails off, and one set above it breaks the line into slivers.
#: Sweeping this against the true ascender-to-descender extent of rendered text
#: - eight sizes, both faces, tight and loose leading, grain to sigma 18 - every
#: value from 0.02 to 0.04 lands within a tenth of the truth; 0.02 starts
#: reading grain as a descender on the noisiest page, so the cut sits at the top
#: of the range that still measures the whole line.
EXTENT_SHARE = 0.04
#: Shortest run of rows that can be part of a line of text.
MIN_LINE_ROWS = 2
#: Profile swing relative to its own mean, below which there are no text rows.
#: Pages of text measure 0.75 and up, and a sparse one measures several;
#: photographs and grainy blank paper sit between 0.05 and 0.2.
TEXT_LINE_CONTRAST = 0.5
#: How many line-shaped bands must repeat down the page before it is text. A
#: photograph's broad tonal bands can out-swing a text comb, but they repeat
#: three or four times in a frame where text repeats dozens.
MIN_TEXT_LINES = 6
#: Share of the page that must come out as bare paper for it to be a document.
#: Scans of text run 0.9 and up; photographs rarely clear a half.
MIN_PAPER_SHARE = 0.72
#: Ink strength, on the 0 to 1 scale of :func:`ink_plane`, at which a pixel is a
#: stroke rather than paper.
INK_CUT = 0.25
#: Ink strength below which a pixel is bare paper.
PAPER_CUT = 0.08
#: Ink coverage at or below which a page holds nothing worth cleaning.
BLANK_INK_SHARE = 0.0015
#: Denominator for the background radius, as a share of the plane's short side.
#: Wide enough to ignore the text, narrow enough to follow a lighting gradient.
BACKGROUND_DIVISOR = 24.0
#: Floor for that radius, so a small page still gets a usable background.
BACKGROUND_MIN_RADIUS = 4
#: Percentile of the ink plane taken as "the darkest stroke".
INK_PEAK_PERCENTILE = 99.7
#: How far the darkest stroke must sit below the paper beside it, in grey
#: levels, before a page has ink on it at all. The ink plane is scaled to its
#: own darkest mark, which is what lets it read a faint pencil page - and which
#: would just as happily scale the grain on an empty sheet into what looks like
#: dense text. This is the floor that stops it.
MIN_INK_DEPTH = 14.0
#: How much the ink has to be gathered into marks, rather than spread evenly as
#: grain, before a page has anything on it. Measured by :func:`ink_structure`.
#: Pages of text measure 17 and up; empty paper, however grainy, stays under 5.
MIN_INK_STRUCTURE = 8.0
#: Denominator for the radius that blur is done at, as a share of the short side.
STRUCTURE_DIVISOR = 64.0
#: Floor for that radius.
STRUCTURE_MIN_RADIUS = 3

#: The three answers :attr:`PageStats.kind` can give.
PAGE_KINDS = ("document", "blank", "photograph")


def background_radius(plane: np.ndarray) -> int:
    """Box radius that follows the lighting on a page without following its text."""
    return int(max(BACKGROUND_MIN_RADIUS, round(min(plane.shape) / BACKGROUND_DIVISOR)))


class InkPlane(NamedTuple):
    """The ink on a page, and the two numbers that say whether it is ink at all."""

    #: 0.0 where the paper is, 1.0 at the darkest stroke.
    values: np.ndarray
    #: How far that darkest stroke ran below the paper beside it, in grey levels.
    depth: float


def ink_plane(plane: np.ndarray) -> InkPlane:
    """Turn luminance into ink. See :class:`InkPlane` for what comes back.

    ``values`` runs 0.0 where the paper is to 1.0 at the darkest stroke. That
    scaling is what lets the same code read a faint pencil page and a crisp
    laser one - and it would just as happily scale the grain on an empty sheet
    into what looks like dense text, which is why ``depth`` and ``light_depth``
    come back with it.

    Ink is measured against the paper *beside it*, not against one number for
    the whole sheet. A scan lit from one side can be seventy grey levels darker
    at the far edge than at the near one, and a single paper level then reads
    that whole edge as ink - which buries the comb the skew search is looking
    for, and makes a perfectly ordinary page look like a photograph. Subtracting
    a wide local mean instead removes any lighting that varies more slowly than
    the text does, and it flattens a black scanner margin to nothing at the same
    time, so the border stops pulling on the angle.

    The scale comes from a high percentile rather than the maximum, so a single
    hot pixel of impulse noise cannot decide what "the darkest stroke" means.
    """
    floats = plane.astype(np.float64)
    background = _images.local_mean(plane, background_radius(plane))
    ink = np.clip(background - floats, 0.0, None)
    peak = float(np.percentile(ink, INK_PEAK_PERCENTILE))
    if peak <= 0.0:
        return InkPlane(np.zeros_like(ink), 0.0)
    return InkPlane(np.clip(ink / peak, 0.0, 1.0), peak)


def ink_structure(ink: np.ndarray) -> float:
    """How far from evenly spread the ink on a page is, in grey levels.

    Ink gathers. Grain does not. Blur the ink plane over a small neighbourhood
    and grain averages away to a flat field, because it is as likely to land
    anywhere as anywhere else, while strokes and margins keep their difference,
    because ink is somewhere and not elsewhere. What comes back is how much of
    that difference is left, and it is the measurement that tells a faint page
    of pencil from a dusty empty one - the two look identical to anything that
    only counts how dark the darkest mark is.
    """
    if ink.size == 0:                    # pragma: no cover - guarded by callers
        return 0.0
    radius = int(max(STRUCTURE_MIN_RADIUS, round(min(ink.shape) / STRUCTURE_DIVISOR)))
    levels = np.clip(ink * 255.0, 0.0, 255.0).astype(np.uint8)
    return float(_images.local_mean(levels, radius).std())


def _shear_margin(width: int, degrees: float) -> int:
    """Rows of headroom a shear of ``degrees`` needs at each end of the profile."""
    return int(np.ceil(width * 0.5 * abs(np.tan(np.radians(degrees))))) + 1


class ProfileBank:
    """One ink plane, ready to be sheared to any angle inside its margin.

    A shear moves every pixel in a column by the same amount, because the
    offset depends on the column and not on the row. That one fact is what
    makes this cheap. Rather than computing a destination row for each of the
    plane's pixels and scattering them - which is a random write per pixel, and
    the slowest thing numpy does - the columns are grouped by the whole number
    of rows they move, each group is summed with a single ``add.reduceat``, and
    the handful of group totals are added into the profile at their offsets.
    The work stops being one pass per pixel and becomes two passes plus a short
    loop, and the profile that comes out is identical to the last float.

    Two shears are offered, and they cost a few times apart. :meth:`profile`
    rounds each column to a whole row, which is cheap enough to sweep the whole
    range. Rounding quantises the answer though: on a nearly upright page every
    angle under about a tenth of a degree rounds to the same offsets, so a
    sweep using it can only bracket the truth. :meth:`profile_fine` splits each
    column between the two rows it falls between, which makes the score a
    smooth function of the angle. The search uses the cheap one to find the
    neighbourhood and the smooth one to land inside it.
    """

    def __init__(self, ink: np.ndarray, max_degrees: float) -> None:
        self.ink = np.ascontiguousarray(ink, dtype=np.float64)
        self.height, self.width = self.ink.shape
        self.total = float(self.ink.sum())
        self.margin = _shear_margin(self.width, max_degrees)
        self._length = self.height + 2 * self.margin
        self._centred = np.arange(self.width, dtype=np.float64) - (self.width - 1) / 2.0

    @property
    def usable(self) -> bool:
        """True when enough whole rows survive the shear margin to mean anything."""
        return self.total > 0.0 and self.height - 2 * self.margin >= 8

    def _offsets(self, degrees: float) -> np.ndarray:
        """Row offset each column takes when the plane is sheared by ``degrees``."""
        return self._centred * np.tan(np.radians(degrees))

    @staticmethod
    def _runs(shifts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Where each run of equal shifts starts, and what it shifts by.

        The offsets grow steadily across the page, so columns that move by the
        same whole number of rows are always next to each other and a run is
        enough to describe them.
        """
        change = np.flatnonzero(np.diff(shifts)) + 1
        starts = np.empty(change.size + 1, dtype=np.intp)
        starts[0] = 0
        starts[1:] = change
        return starts, shifts[starts]

    def profile(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, each column rounded to a row."""
        starts, shifts = self._runs(np.rint(self._offsets(degrees)).astype(np.int64))
        grouped = np.add.reduceat(self.ink, starts, axis=1)
        summed = np.zeros(self._length + 2)
        for group, shift in enumerate(shifts):
            start = self.margin + int(shift)
            summed[start: start + self.height] += grouped[:, group]
        return summed[2 * self.margin: self.height]

    def profile_fine(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, split between adjacent rows.

        Each column lands between two rows and gives each its share, so nudging
        the angle by a hundredth of a degree moves the profile a little rather
        than not at all. This is what makes agreement with a known rotation to
        a fraction of a degree possible.
        """
        offsets = self._offsets(degrees)
        lower = np.floor(offsets)
        starts, shifts = self._runs(lower.astype(np.int64))
        upper_share = self.ink * (offsets - lower)[None, :]
        whole = np.add.reduceat(self.ink, starts, axis=1)
        upper = np.add.reduceat(upper_share, starts, axis=1)
        lower_part = whole - upper
        summed = np.zeros(self._length + 2)
        for group, shift in enumerate(shifts):
            start = self.margin + int(shift)
            summed[start: start + self.height] += lower_part[:, group]
            summed[start + 1: start + 1 + self.height] += upper[:, group]
        return summed[2 * self.margin: self.height]


def comb_score(profile: np.ndarray) -> float:
    """How comb-like a profile is: the energy of its first difference.

    Row variance peaks at the right angle too, but the first difference is
    blind to a page-wide lighting ramp, which variance happily rewards.
    """
    if profile.size < 2:
        return 0.0
    step = np.diff(profile)
    return float(np.dot(step, step))


def strip_banks(ink: np.ndarray, columns: int) -> List[ProfileBank]:
    """Cut the ink plane into vertical strips, each ready to be sheared.

    ``columns`` is the width each strip should be near; ``0`` means one strip
    covering the whole page. Strips are what set the search's aperture, and the
    aperture is what sets how wide the peak is - see :func:`_search`.
    """
    height, width = ink.shape
    if columns <= 0 or width <= columns * 1.5:
        return [ProfileBank(ink, MAX_SKEW_DEGREES)]
    count = max(1, int(round(width / float(columns))))
    edges = np.linspace(0, width, count + 1).astype(int)
    banks = [
        ProfileBank(np.ascontiguousarray(ink[:, start:stop]), MAX_SKEW_DEGREES)
        for start, stop in zip(edges[:-1], edges[1:])
        if stop - start >= MIN_STRIP_COLUMNS
    ]
    usable = [bank for bank in banks if bank.usable]
    return usable or [ProfileBank(ink, MAX_SKEW_DEGREES)]


def _stage_score(banks: Sequence[ProfileBank], degrees: float, smooth: bool) -> float:
    """Total comb score across every strip at one angle."""
    return sum(
        comb_score(bank.profile_fine(degrees) if smooth else bank.profile(degrees))
        for bank in banks
    )


def _refine(angles: np.ndarray, scores: np.ndarray, pick: int) -> float:
    """Where the peak really sits, from the three samples around the best one.

    The comb score near its maximum is close enough to a parabola that fitting
    one through three samples lands nearer the truth than the sample itself -
    and it is free, where another sweep costs a pass over the page.
    """
    best = float(angles[pick])
    if pick <= 0 or pick >= len(scores) - 1 or len(angles) < 2:
        return best
    left, middle, right = float(scores[pick - 1]), float(scores[pick]), float(scores[pick + 1])
    bend = left - 2.0 * middle + right
    if bend == 0.0:
        return best
    shift = 0.5 * (left - right) / bend
    if abs(shift) > 1.0:                 # pragma: no cover - a peak that is not a peak
        return best
    return best + shift * float(angles[1] - angles[0])


def _search(ink: np.ndarray) -> float:
    """Find the angle whose comb is sharpest, in degrees counter-clockwise.

    How sharply the comb answers to angle is set by a single ratio: the height
    of a line of text over the width of the page. A line stays lined up under a
    shear until the shear has moved one end of the page by about its own
    height, so the peak is roughly ``line_height / page_width`` radians wide. On
    a page of six-pixel text twenty-five hundred pixels across, that is a
    seventh of a degree, and a sweep in one-degree steps walks straight over it
    and reports whatever noise it lands on instead. This is not hypothetical: it
    put a page eight degrees out.

    Narrowing the *aperture* is the fix. The same lines measured across a strip
    a quarter of the page wide give a peak four times broader, centred on the
    same angle - and squeezing the page's columns together instead would not,
    because that scales the angle by exactly as much as it broadens the peak.
    So the search begins on narrow strips, where the peak is wide enough for a
    coarse sweep to find, and widens the aperture as it narrows the window:
    each stage sharper than the last, each looking only where the last one
    pointed. The strips' scores are summed rather than one strip chosen, so
    every line of ink on the page stays in the measurement.
    """
    best = 0.0
    for columns, span, step, smooth in SEARCH_STAGES:
        banks = strip_banks(ink, columns)
        if not banks or not any(bank.usable for bank in banks):
            return 0.0
        low = max(-MAX_SKEW_DEGREES, best - span)
        high = min(MAX_SKEW_DEGREES, best + span)
        count = max(int(round((high - low) / step)) + 1, 1)
        angles = np.linspace(low, high, count)
        scores = np.array([_stage_score(banks, angle, smooth) for angle in angles])
        best = _refine(angles, scores, int(np.argmax(scores)))
    return float(best)


def _runs(profile: np.ndarray, threshold: float, minimum: int) -> List[Tuple[int, int]]:
    """Runs of rows above ``threshold``, as half-open ``(start, stop)`` pairs."""
    above = profile > threshold
    if not above.any():
        return []
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(int(above.size))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= minimum]


@dataclass
class LineGeometry:
    """What the straightened profile says about the lines of text on a page."""

    #: Height of an inked line, ascender top to descender foot, in plane pixels.
    text_height: Optional[float] = None
    #: Median baseline-to-baseline distance, in plane pixels.
    pitch: Optional[float] = None
    #: How many line-shaped bands were found.
    line_count: int = 0
    #: Profile swing relative to its own mean. A page with no lines scores ~0.
    line_contrast: float = 0.0


def paper_between(profile: np.ndarray, cores: Sequence[Tuple[int, int]]) -> float:
    """The level the profile falls back to between two lines of text.

    The lowest point in each gap, medianed over the gaps, so one wide margin or
    one pair of lines that touch cannot move it. This is the zero that a line's
    own height is measured from - the profile's global minimum would do on a
    clean page, but on a lit or grainy one the darkest row of the page is not
    the level the gaps sit at.
    """
    dips = [
        float(profile[stop:start].min())
        for (_, stop), (start, _) in zip(cores, cores[1:])
        if start > stop
    ]
    return float(np.median(dips)) if dips else float(profile.min())


def _grow_to_extent(
    profile: np.ndarray, core: Tuple[int, int], cut: float, bounds: Tuple[int, int]
) -> Tuple[int, int]:
    """Walk out from a line's x-height band to its ascender top and descender foot.

    Growing outward from the core is what ties the extent to the line it
    belongs to. Thresholding the whole profile a second time and measuring
    whatever comes out does not: the low cut also lifts the ascenders and
    descenders sitting alone in the gaps into runs of their own, and they
    outnumber the lines.
    """
    start, stop = core
    floor, ceiling = bounds
    while start > floor and profile[start - 1] > cut:
        start -= 1
    while stop < ceiling and profile[stop] > cut:
        stop += 1
    return start, stop


def _extent_bounds(
    cores: Sequence[Tuple[int, int]], index: int, size: int
) -> Tuple[int, int]:
    """How far a line may grow: to the middle of the gap to each neighbour."""
    start, stop = cores[index]
    floor = 0 if index == 0 else (cores[index - 1][1] + start) // 2
    ceiling = size if index + 1 == len(cores) else (stop + cores[index + 1][0] + 1) // 2
    return min(floor, start), max(ceiling, stop)


def line_geometry(profile: np.ndarray) -> LineGeometry:
    """Measure the teeth of a straightened comb profile."""
    if profile.size < MIN_PLANE_SIDE:
        return LineGeometry()
    mean = float(profile.mean())
    if mean <= 0.0:
        return LineGeometry()
    low, high = float(profile.min()), float(profile.max())
    swing = high - low
    contrast = float(profile.std()) / mean
    if swing <= 0.0:
        return LineGeometry(line_contrast=contrast)
    cores = _runs(profile, low + LINE_THRESHOLD * swing, MIN_LINE_ROWS)
    if not cores:
        return LineGeometry(line_contrast=contrast)
    paper = paper_between(profile, cores)
    body = float(np.median([profile[start:stop].max() for start, stop in cores]))
    cut = paper + EXTENT_SHARE * max(body - paper, 0.0)
    heights = [
        stop - start
        for start, stop in (
            _grow_to_extent(
                profile, core, cut, _extent_bounds(cores, index, int(profile.size))
            )
            for index, core in enumerate(cores)
        )
    ]
    centres = [(start + stop) / 2.0 for start, stop in cores]
    pitch = None
    if len(centres) >= 2:
        gaps = np.diff(np.asarray(centres, dtype=np.float64))
        pitch = float(np.median(gaps))
    return LineGeometry(
        text_height=float(np.median(np.asarray(heights, dtype=np.float64))),
        pitch=pitch,
        line_count=len(cores),
        line_contrast=contrast,
    )


@dataclass
class PageStats:
    """One page, measured. Every later decision is taken from these numbers.

    Attributes:
        kind: ``"document"``, ``"blank"`` or ``"photograph"``.
        why: one sentence explaining the kind, in plain language.
        skew_degrees: counter-clockwise degrees the text is off horizontal.
        text_height: height of a line of text in *source* pixels, or ``None``
            when no lines were found.
        line_count: how many lines the profile found, on the analysis plane.
        line_contrast: how far the profile swings relative to its own mean.
        paper_level: luminance of clean paper, 0 to 255.
        ink_depth: how far the darkest stroke sits below the paper beside it,
            in grey levels. Below :data:`MIN_INK_DEPTH` there is no ink, only
            grain.
        ink_structure: how far from evenly spread that ink is, in grey levels.
            Below :data:`MIN_INK_STRUCTURE` it is grain, not marks.
        ink_share: share of pixels that are a stroke, on the local scale
            :func:`ink_plane` measures.
        paper_share: share of pixels that are bare paper on that same scale.
    """

    kind: str = "document"
    why: str = ""
    skew_degrees: float = 0.0
    text_height: Optional[float] = None
    line_count: int = 0
    line_contrast: float = 0.0
    paper_level: float = 255.0
    ink_depth: float = 0.0
    ink_structure: float = 0.0
    ink_share: float = 0.0
    paper_share: float = 0.0

    @property
    def is_document(self) -> bool:
        """True when the page is worth putting through the pipeline."""
        return self.kind == "document"


def measure(plane: np.ndarray, scale: float = 1.0) -> PageStats:
    """Measure an analysis plane: kind, skew, and the height of a line of text.

    Args:
        plane: an 8-bit luminance plane, already downscaled for analysis.
        scale: how much that downscale shrank the source, so lengths can be
            reported back in source pixels.
    """
    stats = PageStats(paper_level=_images.paper_level(plane) if plane.size else 255.0)

    if plane.size == 0 or min(plane.shape) < MIN_PLANE_SIDE:
        stats.kind = "blank"
        stats.why = "the page is too small to hold a line of text"
        return stats

    coarse, _ = _images.analysis_plane(plane, SEARCH_MAX_SIDE)
    coarse_ink, ink_depth = ink_plane(coarse)
    stats.ink_depth = ink_depth
    ink_share = float(np.mean(coarse_ink > INK_CUT))
    paper_share = float(np.mean(coarse_ink < PAPER_CUT))
    stats.ink_share = ink_share
    stats.paper_share = paper_share
    stats.ink_structure = ink_structure(coarse_ink)
    if not ProfileBank(coarse_ink, MAX_SKEW_DEGREES).usable:
        stats.kind = "blank"
        stats.why = "the page carries no ink at all"
        return stats

    degrees = _search(coarse_ink)
    detail_ink = coarse_ink if coarse.shape == plane.shape else ink_plane(plane).values
    detail_bank = ProfileBank(detail_ink, MAX_SKEW_DEGREES)
    geometry = (
        line_geometry(detail_bank.profile_fine(degrees))
        if detail_bank.usable
        else LineGeometry()
    )
    stats.skew_degrees = float(degrees)
    stats.line_count = geometry.line_count
    stats.line_contrast = geometry.line_contrast
    if geometry.text_height is not None and scale > 0.0:
        stats.text_height = float(geometry.text_height) / scale

    has_lines = (
        geometry.line_count >= MIN_TEXT_LINES
        and geometry.line_contrast >= TEXT_LINE_CONTRAST
    )
    if ink_depth < MIN_INK_DEPTH:
        stats.kind = "blank"
        stats.why = (
            "the darkest mark on the page is only {0:.1f} grey levels below the "
            "paper around it, which is grain rather than ink".format(ink_depth)
        )
        stats.skew_degrees = 0.0
        stats.text_height = None
        return stats
    if not has_lines and stats.ink_structure < MIN_INK_STRUCTURE:
        stats.kind = "blank"
        stats.why = (
            "what ink there is lies evenly across the whole sheet rather than "
            "gathered into marks (it measures {0:.1f} against a floor of "
            "{1:g}), which is grain on empty paper".format(
                stats.ink_structure, MIN_INK_STRUCTURE
            )
        )
        stats.skew_degrees = 0.0
        stats.text_height = None
        return stats
    if ink_share <= BLANK_INK_SHARE and not has_lines:
        stats.kind = "blank"
        stats.why = (
            "only {0:.2f}% of the page carries ink, and no lines of text were "
            "found".format(ink_share * 100.0)
        )
        stats.skew_degrees = 0.0
        stats.text_height = None
        return stats
    if paper_share < MIN_PAPER_SHARE and not has_lines:
        stats.kind = "photograph"
        stats.why = (
            "only {0:.0f}% of the page comes out as bare paper and no repeating "
            "lines of text were found, so this reads as a photograph rather "
            "than a document".format(paper_share * 100.0)
        )
        stats.skew_degrees = 0.0
        return stats
    stats.why = (
        "{0} line-shaped bands of text, {1:.0f}% of the page bare paper".format(
            geometry.line_count, paper_share * 100.0
        )
    )
    return stats


def estimate_skew_on_plane(plane: np.ndarray) -> float:
    """Counter-clockwise degrees the text on ``plane`` is off horizontal."""
    coarse, _ = _images.analysis_plane(plane, SEARCH_MAX_SIDE)
    if coarse.size == 0 or min(coarse.shape) < MIN_PLANE_SIDE:
        return 0.0
    ink = ink_plane(coarse).values
    if not ProfileBank(ink, MAX_SKEW_DEGREES).usable:
        return 0.0
    return _search(ink)
