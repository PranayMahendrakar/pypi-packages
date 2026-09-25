"""Skew, orientation and line geometry, all from projection profiles.

Text lines are the only strong periodic structure on a page. Shear the ink plane
by a candidate angle, sum each row, and the profile becomes a comb: tall teeth
where lines of text are, near-zero between them. The comb is sharpest at exactly
one angle, and that angle is the skew.

Every candidate angle is a single ``bincount`` over a sheared index array, so a
few hundred of them cost milliseconds. Nothing is rotated, interpolated or
resampled: shearing by rounded row offsets is faster than bicubic rotation and,
for this measurement, more honest, because rotation smears the very edges being
counted.

Ink is always measured against the local paper surface, never against one
paper level for the whole sheet. A shadow across the top of a page darkens the
paper there, and against a page-wide level that darkened paper would read as
ink, adding a slope to exactly the rows whose shape says how tall the text is
and which way up it reads. Dividing by the surface first is the same
flat-field correction :mod:`document_quality._measures` makes, for the same
reason.

The same profile then gives the line pitch and the height of an inked line,
which is what "is the text big enough to OCR" actually means, and - on a plane
fine enough to resolve a line's ascender and descender zones - which way up the
page reads.

Angles follow ``PIL.Image.rotate``: positive is counter-clockwise. A page whose
text runs downhill to the right has a negative skew, and
``image.rotate(-report.skew_degrees)`` puts it back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import _lighting
from ._images import PagePlanes, percentile_sample
from ._lighting import PaperSurface

logger = logging.getLogger(__name__)

#: Widest skew the search considers, in degrees either side of upright.
MAX_SKEW_DEGREES = 15.0
#: Step of the cheap first sweep across the whole range, on the small plane.
COARSE_STEP = 1.0
#: Refinement steps, on the work plane. Each sweep spans one step of the sweep
#: before it either side of the best angle so far.
REFINE_STEPS = (0.2, 0.05, 0.0125)
#: A profile flatter than this, relative to its own mean, holds no text lines.
MIN_LINE_CONTRAST = 0.04
#: Share of the profile swing that separates a line of text from the gap above.
LINE_THRESHOLD = 0.35
#: Level, as a share of the swing, the profile must stay above between two
#: bands for them to be one line whose body thins out in the middle.
MERGE_LEVEL = 0.12
#: Lower cut used to find the full inked extent, ascender top to descender foot.
#: Ascenders and descenders are sparse - a third of Latin letters rise above
#: the x-height and far fewer drop below it - so their rows carry a small
#: fraction of the ink a line's body does, and the cut has to sit low to keep
#: them.
EXTENT_THRESHOLD = 0.05
#: Cuts tried in turn when the low one lets neighbouring lines run together,
#: which a noisy or show-through page can do by lifting the gaps between them.
EXTENT_FALLBACKS = (0.12, 0.20, LINE_THRESHOLD)
#: Shortest run of rows, in plane pixels, that can be part of a line of text.
MIN_LINE_ROWS = 2
#: Smallest plane either side can have before measuring lines is pointless.
MIN_PLANE_SIDE = 16
#: Profile swing, relative to its mean, below which a page holds no text rows.
#: Text pages measure 2 and up; photographs and blank paper stay under 1.
TEXT_LINE_CONTRAST = 1.0
#: How many times a band must repeat down the page before it counts as text.
#: A photograph's broad tonal bands can out-swing a text comb, but they repeat
#: three or four times across the whole frame where text repeats dozens.
MIN_LINE_REPEATS = 8
#: Least share of the page's columns a band of rows must carry ink in to be a
#: line of text. A line of words runs across the page; a speck of dust, however
#: dark, inks two or three columns, and twenty of them down a blank sheet make
#: a comb that is not text. One short word on a letter page still clears this.
MIN_BAND_COVERAGE = 0.012

#: Ink lighter than this share of the darkest stroke is paper grain, not ink,
#: and is dropped before any profile is built. It carries no line structure,
#: only a uniform floor, and skipping it is most of what makes the search fast.
INK_FLOOR = 0.04
#: The same floor in absolute luminance, 0 to 1. Without it a blank sheet's
#: grain, divided by its own small peak, would pass for a page full of ink.
INK_FLOOR_LEVEL = 0.03
#: Most inked pixels a profile bank keeps. A page of text on the work plane
#: inks tens of thousands; a photograph inks nearly all of it. Past this the
#: bank keeps every n-th column, each standing for n, which leaves the row
#: sums - all a projection profile is - the same in expectation.
MAX_BANK_PIXELS = 50_000
#: The same limit for the single profiles line heights and the up-down
#: balance are read from: thinning by columns can alias with regular type.
GEOMETRY_BANK_PIXELS = 400_000

# -- which way up ----------------------------------------------------------
#: Height, in rows, a line of text is given on the plane that decides which
#: way up the page reads. The ascender and descender zones are each about a
#: fifth of a line, so this leaves them several rows apiece.
DETAIL_TEXT_ROWS = 40.0
#: Below this line height on the work plane, a finer plane is built for the
#: up-versus-down test; at or above it the work plane already resolves it.
DETAIL_MIN_ROWS = 28.0
#: Share of a line's own peak row that marks its dense body, x-height top to
#: baseline. Body rows of real type measure a half to all of the peak; the
#: ascender zone above measures well under a fifth, the descender zone less.
BODY_SHARE = 0.25
#: Ink balance between ascender and descender zones needed before upright and
#: upside-down are told apart. Upright Latin text in Arial, Times and Courier
#: measures 0.4 to 0.8; a page set entirely in capitals or figures has no
#: ascenders or descenders to speak of and measures near zero. Below this the
#: page is left the way it came and the caller is told the 180 question went
#: unanswered, which is better than a coin toss reported as fact.
MIN_BALANCE = 0.2
#: Fewest lines the balance must be measured on, and the share of them that
#: must agree with it, before it is believed.
MIN_BALANCE_LINES = 2
MIN_BALANCE_AGREEMENT = 0.6
#: A weaker balance is still believed on a crisp page when the lines agree on
#: it this many standard deviations beyond chance (see
#: :attr:`Balance.consensus`).
MIN_WEAK_BALANCE = 0.08
MIN_CONSENSUS = 3.5
#: Edge softness (see :attr:`Balance.softness`) at or below which a page is
#: crisp enough for the weaker test, and above which it is soft enough that
#: only a balance of :data:`STRONG_BALANCE` is believed. Crisp scans measure
#: under 0.04; a blur of one and a half pixels on 20 px type measures 0.1,
#: and is where blur starts to drag an upright page's balance below zero.
SHARP_EDGE = 0.05
SOFT_EDGE = 0.07
STRONG_BALANCE = 0.35

#: When the weaker way round combs at least this share as well as the
#: stronger, both are read in full before one is chosen (see :func:`_prefer`).
#: Real pages are nowhere near: the wrong way round combs a fiftieth as well.
CLOSE_COMBS = 0.25
#: Share of lines that must agree before the weaker comb can win on balance.
CHALLENGE_AGREEMENT = 0.95
#: ...while the stronger comb's lines agree no better than this.
HOLDER_AGREEMENT = 0.7

_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


def ink_plane(
    plane: np.ndarray,
    surface: Optional[np.ndarray] = None,
    level: Optional[float] = None,
) -> np.ndarray:
    """Turn luminance into ink: 0 where the paper is, 1 at the darkest stroke.

    The plane is first divided by the local paper ``surface`` (estimated here
    when not given), so a shadow or a lighting ramp leaves paper at paper and
    ink at ink, however unevenly the sheet was lit. ``level`` puts the result
    back on the scale of the page's own paper, so the floors below mean the
    same on a dim scan as on a bright one.

    The paper level is then the 85th percentile of the flattened plane rather
    than its maximum, so one torn white corner or a blown highlight cannot set
    the scale for the whole page.

    The grain floor is applied to how far a pixel falls below its *local*
    paper in plain luminance, before flattening. Scanner noise is roughly the
    same number of grey levels everywhere, so dividing by a dim surface would
    magnify the grain of a shadowed margin into a speckle of ink - and in a
    shadow along one side, into a band of it that reads as lines of text.
    Real ink falls far below even the dimmest paper, and keeps well clear.
    """
    plane = np.asarray(plane, dtype=np.float32)
    if surface is None:
        light = _lighting.paper_surface(plane)
        surface, level = light.surface, light.paper_level
    if level is None:
        level = float(np.median(surface))
    lit = np.maximum(surface, np.float32(1e-3))
    ratio = plane / lit
    paper = float(np.percentile(percentile_sample(ratio), 85.0))
    deficit = lit * np.float32(paper) - plane
    deficit[deficit < np.float32(INK_FLOOR_LEVEL)] = 0.0
    ink = deficit / lit
    ink *= np.float32(max(level, 1e-3))
    peak = float(ink.max()) if ink.size else 0.0
    if peak <= INK_FLOOR_LEVEL:
        return np.zeros_like(ink)
    ink[ink < INK_FLOOR * peak] = 0.0
    ink /= np.float32(peak)
    return ink


def _shear_margin(width: int, degrees: float, aspect: float = 1.0) -> int:
    """Rows of headroom a shear of ``degrees`` needs at each end of the profile."""
    return int(np.ceil(width * 0.5 * aspect * abs(np.tan(np.radians(degrees))))) + 1


class ProfileBank:
    """One ink plane, ready to be sheared to any angle within its margin.

    Only inked pixels are kept: their row, their column offset from the
    centre and their weight, as three flat arrays built once and reused for
    every candidate angle. Paper contributes nothing to a projection profile,
    so leaving it out changes no profile and costs a fraction of the work. A
    plane inked almost everywhere, which is a picture rather than a page, is
    thinned to :data:`MAX_BANK_PIXELS` by keeping every n-th column.

    ``aspect`` is rows per column of the plane relative to the page: a plane
    squeezed across and kept tall, which is how the up-versus-down test sees a
    small typeface in full, shears ``aspect`` times as steeply for one angle.

    The profile covers every row a sheared pixel can land in, the page's own
    rows plus the margin either side, so a line of text in the top margin of
    the sheet is counted at every angle rather than only at some.

    Two shears are offered. :meth:`profile` rounds each column to a whole row,
    which is one integer ``bincount``; it is what the skew search uses at
    every stage, because rounding never smooths a profile, so no angle is
    favoured for landing on whole rows. :meth:`profile_fine` splits each
    column between the two rows it falls between, which is kinder to line
    heights measured on a coarse plane.
    """

    def __init__(
        self, ink: np.ndarray, max_degrees: float, aspect: float = 1.0,
        max_pixels: int = 0,
    ) -> None:
        self.ink = ink
        self.height, self.width = ink.shape
        self.aspect = float(aspect)
        flat = ink.ravel()
        inked = np.flatnonzero(flat > 0.0)
        cap = max_pixels or MAX_BANK_PIXELS
        stride = int(np.ceil(inked.size / float(cap))) if inked.size else 1
        stride = max(stride, 1)
        if stride > 1:
            inked = inked[(inked % self.width) % stride == 0]
        self.stride = stride
        self.weights = flat[inked].astype(np.float64) * float(stride)
        self.total = float(self.weights.sum())
        self.margin = _shear_margin(self.width, max_degrees, self.aspect)
        self._length = self.height + 2 * self.margin
        rows, columns = np.divmod(inked, self.width)
        self.columns = columns.astype(np.int64)
        self._base = rows.astype(np.int64) + self.margin
        self._centred = columns.astype(np.float64) - (self.width - 1) / 2.0

    @property
    def usable(self) -> bool:
        """True when there is ink and enough rows to mean anything."""
        return self.total > 0.0 and self.height >= 8

    @property
    def columns_kept(self) -> int:
        """How many of the plane's columns the bank holds pixels for."""
        return max(1, -(-self.width // self.stride))

    def _offsets(self, degrees: float) -> np.ndarray:
        """Row offset each inked pixel gets when the plane is sheared by ``degrees``."""
        return self._centred * (np.tan(np.radians(degrees)) * self.aspect)

    def rows_at(self, degrees: float) -> np.ndarray:
        """Profile row each inked pixel lands in at ``degrees``, rounded."""
        return self._base + np.rint(self._offsets(degrees)).astype(np.int64)

    def profile(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, each column rounded to a row."""
        summed = np.bincount(self.rows_at(degrees), weights=self.weights,
                             minlength=self._length)
        return summed[:self._length]

    def profile_fine(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, split between adjacent rows."""
        rows = self._base.astype(np.float64) + self._offsets(degrees)
        lower = np.floor(rows)
        upper = self.weights * (rows - lower)
        index = lower.astype(np.int64)
        length = self._length + 1
        summed = np.bincount(index, weights=self.weights - upper, minlength=length)
        summed += np.bincount(index + 1, weights=upper, minlength=length)
        return summed[:self._length]


def comb_score(profile: np.ndarray) -> float:
    """How comb-like a profile is: the energy of its first difference.

    Row variance also peaks at the right angle, but the difference is blind to a
    page-wide lighting ramp, which variance happily rewards.
    """
    if profile.size < 2:
        return 0.0
    step = np.diff(profile)
    return float(np.dot(step, step))


def _sweep(bank: ProfileBank, start: float, stop: float, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Comb scores across ``start..stop`` in ``step`` increments."""
    count = int(round((stop - start) / step)) + 1
    angles = np.linspace(start, stop, max(count, 1))
    scores = np.array([comb_score(bank.profile(angle)) for angle in angles])
    return angles, scores


def _parabola_shift(left: float, middle: float, right: float) -> float:
    """Where the peak of a parabola through three even samples lies, in steps.

    Clamped to half a step either way, which is as far as the true peak can be
    from the sample that won.
    """
    curve = left - 2.0 * middle + right
    if curve >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curve, -0.5, 0.5))


def _plateau_centre(angles: np.ndarray, scores: np.ndarray, polish: bool = True) -> float:
    """The best angle of a sweep, as the middle of its top plateau.

    Rounded shears make the score a staircase near its peak: every angle too
    small to move any column by a whole row scores the same. Taking the first
    of those would lean towards one end - and the next, finer sweep would then
    be centred off the truth - so the middle of the run of top scores is
    taken instead, nudged by a parabola through its neighbours when
    ``polish`` is set and the peak is a single sample.
    """
    top = float(scores.max())
    pick = int(np.argmax(scores))
    tied = np.flatnonzero(scores >= top - 1e-9 * max(abs(top), 1.0))
    run = [pick]
    for index in range(pick + 1, scores.size):
        if index in tied:
            run.append(index)
        else:
            break
    first, last = run[0], run[-1]
    step = float(angles[1] - angles[0]) if angles.size > 1 else 0.0
    best = 0.5 * (float(angles[first]) + float(angles[last]))
    if polish and first == last and 0 < pick < scores.size - 1:
        best += _parabola_shift(scores[pick - 1], scores[pick], scores[pick + 1]) * step
    return best


def coarse_search(coarse: ProfileBank) -> Tuple[float, float]:
    """The whole-degree sweep across the range; returns ``(degrees, strength)``.

    ``strength`` is the best score divided by the page's ink, which compares
    across pages and across the two ways round.
    """
    angles, scores = _sweep(coarse, -MAX_SKEW_DEGREES, MAX_SKEW_DEGREES, COARSE_STEP)
    pick = int(np.argmax(scores))
    return float(angles[pick]), float(scores[pick]) / max(coarse.total, 1e-9)


def search_skew(
    coarse: ProfileBank,
    fine: Optional[ProfileBank] = None,
    refine: bool = True,
    start: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Coarse-to-fine hunt for the sharpest comb; returns ``(degrees, strength)``.

    The first sweep crosses the whole range in whole degrees on the small
    plane, which only has to land within a degree of the answer; ``start``
    is its result when it has already been run. The refinement runs on
    ``fine`` - the same page at the work plane's resolution, where a tenth of
    a degree moves the ends of a line by a whole row - so the final step
    resolves the angle rather than the pixel grid. ``refine=False`` stops
    after the first sweep.
    """
    best, strength = start if start is not None else coarse_search(coarse)
    if not refine:
        return best, strength
    bank = fine if fine is not None and fine.usable else coarse
    span = COARSE_STEP
    for index, step in enumerate(REFINE_STEPS):
        start = max(-MAX_SKEW_DEGREES, best - span)
        stop = min(MAX_SKEW_DEGREES, best + span)
        angles, scores = _sweep(bank, start, stop, step)
        best = _plateau_centre(angles, scores, index == len(REFINE_STEPS) - 1)
        span = step
    return float(np.clip(best, -MAX_SKEW_DEGREES, MAX_SKEW_DEGREES)), strength


@dataclass
class InkPlanes:
    """The ink of one page on the small and the work plane, flat-fielded.

    The profile banks and the whole-degree sweep are kept once made, so
    comparing the two ways round and then refining the winner never builds
    or sweeps the same plane twice.
    """

    fine: np.ndarray
    work: np.ndarray
    #: The paper surface of the work plane, in the orientation of ``work``.
    surface: np.ndarray
    #: The page's paper level, which :func:`ink_plane` scales ink by.
    level: float
    _coarse_bank: Optional[ProfileBank] = field(default=None, repr=False, compare=False)
    _work_bank: Optional[ProfileBank] = field(default=None, repr=False, compare=False)
    _coarse: Optional[Tuple[float, float]] = field(default=None, repr=False, compare=False)

    def coarse_bank(self) -> Optional[ProfileBank]:
        """The small plane's profile bank, or ``None`` when it is too small."""
        if self._coarse_bank is None and self.fine.size and min(self.fine.shape) >= MIN_PLANE_SIDE:
            self._coarse_bank = ProfileBank(self.fine, MAX_SKEW_DEGREES)
        return self._coarse_bank

    def work_bank(self) -> Optional[ProfileBank]:
        """The work plane's profile bank, or ``None`` when it is too small."""
        if self._work_bank is None and self.work.size and min(self.work.shape) >= MIN_PLANE_SIDE:
            self._work_bank = ProfileBank(self.work, MAX_SKEW_DEGREES)
        return self._work_bank

    def coarse(self) -> Optional[Tuple[float, float]]:
        """The whole-degree sweep's ``(degrees, strength)``, or ``None``."""
        bank = self.coarse_bank()
        if bank is None or not bank.usable:
            return None
        if self._coarse is None:
            self._coarse = coarse_search(bank)
        return self._coarse

    def turned(self, quarter_turns: int) -> "InkPlanes":
        """The same ink turned ``quarter_turns`` x 90 degrees counter-clockwise."""
        k = int(quarter_turns) % 4
        if k == 0:
            return self
        return InkPlanes(
            fine=np.ascontiguousarray(np.rot90(self.fine, k)),
            work=np.ascontiguousarray(np.rot90(self.work, k)),
            surface=np.ascontiguousarray(np.rot90(self.surface, k)),
            level=self.level,
        )


def ink_planes(planes: PagePlanes, light: Optional[PaperSurface] = None) -> InkPlanes:
    """Flat-fielded ink for the small and the work plane of ``planes``.

    ``light`` is the work plane's paper surface when the caller already has
    it; it is estimated here otherwise. The small plane is flattened by the
    same surface, resampled, so both see one and the same lighting.
    """
    if light is None or tuple(light.surface.shape) != tuple(planes.work.shape):
        light = _lighting.paper_surface(planes.work)
    level = light.paper_level
    work = ink_plane(planes.work, light.surface, level)
    fine = ink_plane(planes.fine, light.surface_for(planes.fine.shape), level)
    return InkPlanes(fine=fine, work=work, surface=light.surface, level=level)


def skew_of(ink: InkPlanes, refine: bool = True) -> Tuple[float, float]:
    """Skew of a page from its ink planes; returns ``(degrees, line_strength)``."""
    start = ink.coarse()
    coarse = ink.coarse_bank()
    if start is None or coarse is None:
        return 0.0, 0.0
    if not refine:
        return start[0] + 0.0, start[1]
    degrees, strength = search_skew(coarse, ink.work_bank(), True, start)
    # A thousandth of a degree is finer than the search resolves; rounding to
    # it also keeps a straight page from reporting a skew of "-0.00".
    return round(degrees, 3) + 0.0, strength


@dataclass
class LineGeometry:
    """What the deskewed profile says about the lines of text on the page."""

    #: Height of an inked line, ascender top to descender foot, in plane pixels.
    text_height: Optional[float] = None
    #: Median baseline-to-baseline distance, in plane pixels.
    pitch: Optional[float] = None
    #: How many line-shaped bands were found.
    line_count: int = 0
    #: Profile swing relative to its own mean; a page with no lines scores ~0.
    line_contrast: float = 0.0
    #: Ink balance between the ascender and descender zones of the lines, -1
    #: to 1; upright Latin script is positive. See :func:`up_down_balance`.
    head_room: float = 0.0
    #: Share of profile rows that are inside a line band rather than a gap.
    line_fill: float = 0.0
    #: Share of the page's ink that falls inside a line band rather than between.
    band_mass: float = 0.0
    #: Median share of the page's columns a band carries ink in. A line of
    #: words runs across the page; a speck of dust does not.
    band_coverage: float = 0.0


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


def _line_bands(profile: np.ndarray, floor: float, swing: float) -> List[Tuple[int, int]]:
    """The body of each line of text in a profile, one band per line.

    Rows above :data:`LINE_THRESHOLD` of the swing are a line's body. A light
    or monospaced face - Courier is the usual culprit - thins out halfway down
    its body, far enough to dip under that cut and split one line into two or
    three bands. Two bands are therefore one line when the dip between them
    is short - no longer than half the taller of the two - and the profile in
    it never falls below :data:`MERGE_LEVEL` of the swing. The gap between two
    real lines is longer than that, and drops to the few ascenders and
    descenders reaching into it or to nothing at all.
    """
    bands = _runs(profile, floor + LINE_THRESHOLD * swing, MIN_LINE_ROWS - 1)
    if len(bands) < 2:
        return [band for band in bands if band[1] - band[0] >= MIN_LINE_ROWS]
    joined: List[Tuple[int, int]] = [bands[0]]
    bridge = floor + MERGE_LEVEL * swing
    for start, stop in bands[1:]:
        last_start, last_stop = joined[-1]
        dip = start - last_stop
        tallest = max(last_stop - last_start, stop - start)
        if dip <= 0 or (
            dip <= 0.5 * tallest and float(profile[last_stop:start].min()) > bridge
        ):
            joined[-1] = (last_start, stop)
        else:
            joined.append((start, stop))
    return [band for band in joined if band[1] - band[0] >= MIN_LINE_ROWS]


def _profile_levels(profile: np.ndarray) -> Tuple[float, float]:
    """The gap floor and the line ceiling of a profile.

    The floor is a low percentile of every row. The ceiling is a high
    percentile of the rows that carry ink at all, not of every row: on a page
    with three lines of text at the top, nineteen rows in twenty are empty
    paper, and a percentile over all of them would put the ceiling on the
    paper and find no lines.
    """
    floor = float(np.percentile(profile, 5.0))
    peak = float(profile.max()) if profile.size else 0.0
    inked = profile[profile > floor + 0.02 * (peak - floor)]
    if inked.size == 0:
        return floor, floor
    return floor, float(np.percentile(inked, 97.0))


def _band_coverage(
    bank: ProfileBank, degrees: float, bands: Sequence[Tuple[int, int]]
) -> float:
    """Median share of the plane's columns each band carries ink in."""
    if not bands:
        return 0.0
    lookup = np.full(bank._length + 1, -1, dtype=np.int64)
    for index, (start, stop) in enumerate(bands):
        lookup[start:stop] = index
    rows = np.clip(bank.rows_at(degrees), 0, bank._length)
    ids = lookup[rows]
    keep = ids >= 0
    if not keep.any():
        return 0.0
    keys = np.unique(ids[keep] * np.int64(bank.width) + bank.columns[keep])
    counts = np.bincount(keys // np.int64(bank.width), minlength=len(bands))
    return float(np.median(counts)) / float(bank.columns_kept)


def line_geometry_of_ink(
    ink: np.ndarray, degrees: float, bank: Optional[ProfileBank] = None
) -> LineGeometry:
    """Measure line height, pitch and band shape from an ink plane at ``degrees``.

    ``bank`` is that plane's profile bank when one has been built already.
    """
    if ink.size == 0 or min(ink.shape) < MIN_PLANE_SIDE:
        return LineGeometry()
    if bank is None or bank.ink is not ink or bank.stride > 1:
        # Line heights want every column: thinning a page of regular strokes
        # by columns can alias with them and shorten every line.
        bank = ProfileBank(ink, max(abs(degrees), 0.01), max_pixels=GEOMETRY_BANK_PIXELS)
    if not bank.usable:
        return LineGeometry()
    profile = bank.profile_fine(degrees)

    floor, ceiling = _profile_levels(profile)
    swing = ceiling - floor
    mean = float(profile[bank.margin:bank.margin + bank.height].mean())
    contrast = swing / mean if mean > 1e-9 else 0.0
    geometry = LineGeometry(line_contrast=contrast)
    if swing <= 1e-9 or contrast < MIN_LINE_CONTRAST:
        return geometry

    bands = _line_bands(profile, floor, swing)
    geometry.line_count = len(bands)
    if not bands:
        return geometry
    geometry.band_coverage = _band_coverage(bank, degrees, bands)
    total = float(profile.sum())
    if total > 1e-9:
        inside = float(sum(profile[start:stop].sum() for start, stop in bands))
        geometry.band_mass = min(1.0, inside / total)
    if len(bands) >= 2:
        starts = np.array([start for start, _ in bands], dtype=np.float64)
        gaps = np.diff(starts)
        gaps = gaps[gaps > 0]
        if gaps.size:
            geometry.pitch = float(np.median(gaps))

    # The full inked extent needs a lower cut than line counting does: at the
    # counting threshold a band stops at the x-height, missing the ascenders
    # and descenders that make a line of text as tall as it really is.
    smallest = max(MIN_LINE_ROWS, int(round((geometry.pitch or 0.0) * 0.2)))
    extents: List[Tuple[int, int]] = []
    for share in (EXTENT_THRESHOLD,) + EXTENT_FALLBACKS:
        extents = _runs(profile, floor + share * swing, smallest)
        if not extents or geometry.pitch is None:
            break
        if float(np.median([b - a for a, b in extents])) < 0.95 * geometry.pitch:
            break
    if extents:
        heights = [stop - start for start, stop in extents]
        geometry.text_height = float(np.median(heights))
        geometry.line_fill = float(sum(heights)) / float(bank.height)
    else:  # pragma: no cover - a lower cut always finds at least the bands above
        geometry.text_height = float(np.median([b - a for a, b in bands]))
    return geometry


def holds_text_lines(geometry: LineGeometry, rows: int) -> bool:
    """True when a profile looks like rows of text rather than broad tone or specks.

    Three things have to hold at once. The profile must swing hard relative to
    its own mean, which paper between lines of ink does and an evenly exposed
    photograph does not; the band it swings with must be fine enough to
    repeat :data:`MIN_LINE_REPEATS` times down ``rows``, which keeps a
    photograph of sky over land, swinging across three broad bands, from being
    read as a page of text; and the bands must run across the page the way
    words do, which keeps a sprinkling of dust on a blank sheet from being
    read as rows of it.
    """
    if geometry.line_contrast < TEXT_LINE_CONTRAST:
        return False
    if geometry.pitch is None or geometry.pitch <= 0:
        return False
    if geometry.band_coverage < MIN_BAND_COVERAGE:
        return False
    return geometry.pitch * MIN_LINE_REPEATS <= rows


# --------------------------------------------------------------------------
# which way up
# --------------------------------------------------------------------------


@dataclass
class Balance:
    """How the ink of each line splits between ascender and descender zones."""

    #: ``(above - below) / (above + below)`` summed over lines; -1 to 1.
    value: float = 0.0
    #: Lines it was measured on.
    lines: int = 0
    #: Share of those lines whose own balance has the same sign as ``value``.
    agreement: float = 0.0
    #: How soft the lines' edges are: the median rows a line's body takes to
    #: climb from a quarter of its peak to three quarters, as a share of the
    #: line's inked height. A crisp scan measures under 0.04.
    softness: float = 0.0

    @property
    def consensus(self) -> float:
        """How far the line votes are from a coin toss, in standard deviations.

        With no up-or-down information each line would vote either way at
        random, so ``k`` of ``n`` agreeing sits ``(2k - n) / sqrt(n)`` standard
        deviations from chance - a sign test.
        """
        if self.lines <= 0:
            return 0.0
        agreeing = self.agreement * self.lines
        return (2.0 * agreeing - self.lines) / float(np.sqrt(self.lines))

    @property
    def decided(self) -> bool:
        """True when the balance is strong and consistent enough to act on.

        On a crisp page a plainly strong balance decides, and so does a weaker
        one - text short of ascenders - when nearly every line agrees on it.
        A soft page is held to a much higher bar. Blur spills each line's
        dense body into the zones either side of it, and a body is rarely
        equally heavy at its top and its bottom, so blur drags the balance
        towards zero and on descender-heavy text past it. Calling an upright
        page upside down is the one answer worse than no answer.
        """
        if self.lines < MIN_BALANCE_LINES or self.agreement < MIN_BALANCE_AGREEMENT:
            return False
        strength = abs(self.value)
        if self.softness > SOFT_EDGE:
            return strength >= STRONG_BALANCE
        if strength >= MIN_BALANCE:
            return True
        return (
            self.softness <= SHARP_EDGE
            and strength >= MIN_WEAK_BALANCE
            and self.consensus >= MIN_CONSENSUS
        )


def _edge_softness(segment: np.ndarray, peak: float) -> Optional[float]:
    """Rows a line takes to rise from 25% to 75% of its peak, per inked row."""
    high = np.flatnonzero(segment >= 0.75 * peak)
    low = np.flatnonzero(segment >= 0.25 * peak)
    inked = np.flatnonzero(segment >= 0.03 * peak)
    if high.size == 0 or low.size == 0 or inked.size < 2:
        return None
    rise = 0.5 * float((high[0] - low[0]) + (low[-1] - high[-1]))
    return rise / float(inked[-1] - inked[0] + 1)


def up_down_balance(
    profile: np.ndarray,
    pitch: Optional[float] = None,
    text_rows: Optional[float] = None,
) -> Balance:
    """Which way up a deskewed profile reads, from where each line's ink sits.

    This is the only thing on a page that says which way up it is, and it is a
    fact about the script rather than about the ink. A Latin line has a dense
    body from x-height to baseline, and sparse zones above and below it. The
    two zones are about the same height, so their *extents* say nothing; what
    differs is how much ink they carry. A third of lower-case letters rise
    into the zone above - b d f h k l t, the dot of every i and j, and every
    capital - while only g j p q y drop into the zone below. So upright text
    carries several times more ink above its body than below, and upside-down
    text the reverse.

    Lines are found on a copy smoothed over a third of ``text_rows``, so a
    face that thins out halfway down its body - Courier - is still one line.
    Each line is framed halfway to its neighbours, its body is the rows at or
    above :data:`BODY_SHARE` of its own peak row, and the ink outside the body
    is summed on each side. The answer sums every line's ink before dividing,
    so a long line counts for more than a short one, and records how many of
    the lines agree on their own and how soft their edges are.
    """
    if profile.size < 8:
        return Balance()
    floor, ceiling = _profile_levels(profile)
    swing = ceiling - floor
    if swing <= 1e-9:
        return Balance()
    width = int(round((text_rows or 0.0) / 3.0))
    if width >= 3:
        smooth = np.convolve(profile, np.ones(width) / float(width), mode="same")
        s_floor, s_ceiling = _profile_levels(smooth)
        bands = _line_bands(smooth, s_floor, s_ceiling - s_floor)
    else:
        bands = _line_bands(profile, floor, swing)
    if not bands:
        return Balance()
    centres = [0.5 * (start + stop) for start, stop in bands]
    if pitch is None or pitch <= 0:
        if len(centres) >= 2:
            pitch = float(np.median(np.diff(centres)))
        else:
            pitch = 2.0 * float(bands[0][1] - bands[0][0])
    half = max(2.0, 0.5 * pitch)

    above_total = below_total = 0.0
    votes: List[float] = []
    softness: List[float] = []
    for index, centre in enumerate(centres):
        top = centres[index - 1] if index > 0 else centre - 2.0 * half
        bottom = centres[index + 1] if index + 1 < len(centres) else centre + 2.0 * half
        low = int(round(max(0.5 * (top + centre), centre - half)))
        high = int(round(min(0.5 * (centre + bottom), centre + half))) + 1
        low, high = max(0, low), min(profile.size, high)
        segment = np.clip(profile[low:high] - floor, 0.0, None)
        if segment.size < 5:
            continue
        peak = float(segment.max())
        if peak <= 1e-9:
            continue
        body = np.flatnonzero(segment >= BODY_SHARE * peak)
        first, last = int(body[0]), int(body[-1])
        above = float(segment[:first].sum())
        below = float(segment[last + 1:].sum())
        if above + below <= 0.02 * float(segment[first:last + 1].sum()):
            continue
        soft = _edge_softness(segment, peak)
        if soft is not None:
            softness.append(soft)
        above_total += above
        below_total += below
        votes.append((above - below) / (above + below))
    if not votes or above_total + below_total <= 0.0:
        return Balance()
    value = (above_total - below_total) / (above_total + below_total)
    sign = 1.0 if value >= 0 else -1.0
    agreement = float(np.mean([1.0 if vote * sign > 0 else 0.0 for vote in votes]))
    return Balance(
        value=float(value), lines=len(votes), agreement=agreement,
        softness=float(np.median(softness)) if softness else 1.0,
    )


def detail_ink(
    planes: PagePlanes,
    surface: np.ndarray,
    level: float,
    quarter_turns: int,
    text_rows: Optional[float],
) -> Tuple[np.ndarray, float]:
    """An ink plane tall enough to resolve a line's ascender and descender zones.

    The work plane shrinks a 20 px line on a letter page to 6 rows, leaving
    its ascender and descender zones a row each. This plane keeps the work
    plane's columns but gives the rows back, up to native resolution, so each
    line is about :data:`DETAIL_TEXT_ROWS` rows tall. Row sums are all a
    projection profile needs, and squeezing only across keeps them exact at a
    fraction of the native page's pixels.

    ``surface`` is the work plane's paper surface turned the same way, and the
    result is flattened by it. Returns the ink and its aspect, rows per column
    relative to the page, which is what a shear of one angle has to be scaled
    by on this plane.
    """
    k = int(quarter_turns) % 4
    work_rows, work_columns = np.rot90(planes.work, k).shape
    lum = planes.lum if k == 0 else np.ascontiguousarray(np.rot90(planes.lum, k))
    native_rows, native_columns = lum.shape
    row_scale = work_rows / float(native_rows)
    if text_rows and text_rows > 0:
        row_scale = min(1.0, max(row_scale, row_scale * DETAIL_TEXT_ROWS / text_rows))
    rows = max(1, int(round(native_rows * row_scale)))
    columns = max(1, int(work_columns))
    if (rows, columns) == (native_rows, native_columns):
        plane = lum.astype(np.float32) / np.float32(255.0)
    else:
        small = Image.fromarray(np.ascontiguousarray(lum, dtype=np.uint8)).resize(
            (columns, rows), _BOX
        )
        plane = np.asarray(small, dtype=np.float32) / np.float32(255.0)
    ink = ink_plane(plane, _lighting.resample(surface, plane.shape), level)
    aspect = (rows / float(native_rows)) / (columns / float(native_columns))
    return ink, aspect


@dataclass
class Orientation:
    """Which way up the page is, and the line measurements that decided it."""

    #: Counter-clockwise quarter turn that sets the page upright.
    degrees: int = 0
    #: ``"none"``, ``"lines"`` or ``"lines+headroom"``; see
    #: :func:`detect_orientation_on_plane`.
    basis: str = "none"
    #: Skew of the page once turned by the quarter turns (a half turn leaves a
    #: skew angle as it was), so the caller need not search for it again.
    skew: float = 0.0
    #: Line geometry of the page turned the same way, measured on the work plane.
    geometry: LineGeometry = field(default_factory=LineGeometry)
    #: The ascender-versus-descender evidence, when there was any.
    balance: Balance = field(default_factory=Balance)


@dataclass
class _Axis:
    """Everything one way round says about the page."""

    base: int
    ink: InkPlanes
    degrees: float
    geometry: LineGeometry
    lines: bool
    balance: Balance = field(default_factory=Balance)


def _read_axis(planes: PagePlanes, ink: InkPlanes, base: int) -> _Axis:
    """Skew, line geometry and up-down balance of the page turned by ``base``."""
    degrees, _ = skew_of(ink)
    geometry = line_geometry_of_ink(ink.work, degrees, ink.work_bank())
    axis = _Axis(base, ink, degrees, geometry, holds_text_lines(geometry, ink.work.shape[0]))
    if not axis.lines:
        return axis
    native_rows = planes.lum.shape[0] if base % 180 == 0 else planes.lum.shape[1]
    if (geometry.text_height and geometry.text_height >= DETAIL_MIN_ROWS) or (
        native_rows < 1.25 * ink.work.shape[0]
    ):
        # The work plane already has the rows - tall type, or a page hardly
        # bigger than the work plane - so there is nothing finer to build.
        detail, aspect = ink.work, 1.0
        bank = ProfileBank(detail, max(abs(degrees), 0.01), aspect,
                           max_pixels=GEOMETRY_BANK_PIXELS)
    else:
        detail, aspect = detail_ink(
            planes, ink.surface, ink.level, base // 90, geometry.text_height
        )
        bank = ProfileBank(detail, max(abs(degrees), 0.01), aspect,
                           max_pixels=GEOMETRY_BANK_PIXELS)
    if bank.usable:
        stretch = detail.shape[0] / float(ink.work.shape[0])
        pitch = geometry.pitch * stretch if geometry.pitch else None
        text_rows = geometry.text_height * stretch if geometry.text_height else None
        axis.balance = up_down_balance(bank.profile(degrees), pitch, text_rows)
    geometry.head_room = axis.balance.value
    return axis


def _prefer(first: _Axis, second: _Axis) -> _Axis:
    """The better reading of two ways round whose combs came out close.

    Text whose letters line up in columns down the page - a monospaced face,
    or the same sentence set line after line - combs almost as well turned
    sideways as upright, because its columns of letters are as regular as its
    lines. Only the right way round shows lines of text whose letters say
    which way up they read, so a reading with a settled up-down balance beats
    one without; otherwise the stronger comb stands.
    """
    if first.lines != second.lines:
        return first if first.lines else second
    # The weaker comb wins only when nearly every one of its lines leans the
    # same way while the stronger comb's lines do not agree. Lines of text
    # set alike lean together; columns of letters lean every which way - but
    # a monospaced column can lean by chance, so the bar is set very high.
    challenger, holder = second.balance, first.balance
    if (
        challenger.lines >= MIN_BALANCE_LINES
        and challenger.agreement >= CHALLENGE_AGREEMENT
        and abs(challenger.value) >= MIN_WEAK_BALANCE
        and holder.agreement <= HOLDER_AGREEMENT
    ):
        return second
    return first


def orient(planes: PagePlanes, light: Optional[PaperSurface] = None) -> Orientation:
    """Settle which way up the page is, keeping the skew and lines found on the way.

    See :func:`detect_orientation_on_plane` for what the answer means. The two
    ways round are compared with the cheap one-degree sweep; the full reading
    is taken on the way round that won, and on the other as well only when
    the two combs came out within :data:`CLOSE_COMBS` of each other.
    """
    fine = planes.fine
    if fine.size == 0 or min(fine.shape) < MIN_PLANE_SIDE:
        return Orientation()
    ink = ink_planes(planes, light)
    _, upright_score = skew_of(ink, refine=False)
    turned_ink = ink.turned(1)
    _, turned_score = skew_of(turned_ink, refine=False)
    if max(upright_score, turned_score) <= 0.0:
        return Orientation()

    # np.rot90(x, 1) turns the page counter-clockwise. If that is the one whose
    # lines run across, the correction is that same counter-clockwise quarter.
    ranked = [(upright_score, 0, ink), (turned_score, 90, turned_ink)]
    ranked.sort(key=lambda item: -item[0])
    best = _read_axis(planes, ranked[0][2], ranked[0][1])
    if ranked[1][0] >= CLOSE_COMBS * ranked[0][0]:
        best = _prefer(best, _read_axis(planes, ranked[1][2], ranked[1][1]))

    if not best.lines:
        # Neither way round produced a comb, so there are no lines of text here
        # and there is nothing to be upright about. Saying 0 would look like a
        # finding; "none" says the question could not be answered. The
        # upright measurements are kept, because they are what the page is.
        if best.base:
            degrees, _ = skew_of(ink)
            return Orientation(0, "none", degrees, line_geometry_of_ink(ink.work, degrees))
        return Orientation(0, "none", best.degrees, best.geometry)
    balance = best.balance
    if not balance.decided:
        return Orientation(best.base, "lines", best.degrees, best.geometry, balance)
    turn = best.base if balance.value > 0 else (best.base + 180) % 360
    return Orientation(turn, "lines+headroom", best.degrees, best.geometry, balance)


def detect_orientation_on_plane(
    planes: PagePlanes, light: Optional[PaperSurface] = None
) -> Tuple[int, str]:
    """Quarter-turn that sets this page upright; returns ``(degrees, basis)``.

    The question is really two questions, and they are answered by different
    evidence. Which way the lines *run* is settled by the projection comb,
    which is unambiguous: text only combs one way round. Which way up they
    *read* is settled by :func:`up_down_balance`, which is a weaker signal and
    sometimes has nothing to say at all.

    Returns:
        ``(degrees, basis)``. ``degrees`` is counter-clockwise, so
        ``image.rotate(degrees, expand=True)`` is the correction to apply.
        ``basis`` says how much the answer rests on:

        ``"none"``
            the page holds no text lines, and 0 is a default, not a finding
        ``"lines"``
            which way the lines run was clear, but upright versus upside-down
            was not, so the answer may be 180 degrees out
        ``"lines+headroom"``
            both tests had something to work with
    """
    found = orient(planes, light)
    return found.degrees, found.basis
