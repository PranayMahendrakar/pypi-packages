"""Skew, orientation and line geometry, all from one projection profile.

Text lines are the only strong periodic structure on a page. Shear the ink plane
by a candidate angle, sum each row, and the profile becomes a comb: tall teeth
where lines of text are, near-zero between them. The comb is sharpest at exactly
one angle, and that angle is the skew.

Every candidate angle is a single ``bincount`` over a sheared index array, so a
few hundred of them cost milliseconds on the 512-pixel plane. Nothing is
rotated, interpolated or resampled: shearing by rounded row offsets is faster
than bicubic rotation and, for this measurement, more honest, because rotation
smears the very edges being counted.

The same profile then gives the line pitch and the height of an inked line,
which is what "is the text big enough to OCR" actually means.

Angles follow ``PIL.Image.rotate``: positive is counter-clockwise. A page whose
text runs downhill to the right has a negative skew, and
``image.rotate(-report.skew_degrees)`` puts it back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Widest skew the search considers, in degrees either side of upright.
MAX_SKEW_DEGREES = 15.0
#: Coarse-to-fine search steps in degrees; the last one sets the resolution.
SEARCH_STEPS = (1.0, 0.1, 0.02)
#: A profile flatter than this, relative to its own mean, holds no text lines.
MIN_LINE_CONTRAST = 0.04
#: Share of the profile swing that separates a line of text from the gap above.
LINE_THRESHOLD = 0.35
#: Lower cut used to find the full inked extent, ascender top to descender foot.
EXTENT_THRESHOLD = 0.12
#: Upper cut that isolates the dense body of a line, x-height top to baseline.
CORE_THRESHOLD = 0.60
#: Shortest run of rows, in plane pixels, that can be part of a line of text.
MIN_LINE_ROWS = 2
#: Smallest plane either side can have before measuring lines is pointless.
MIN_PLANE_SIDE = 16
#: Profile swing, relative to its mean, below which a page holds no text rows.
#: Text pages measure 2 and up; photographs and blank paper stay under 1.
TEXT_LINE_CONTRAST = 1.0
#: Headroom asymmetry needed before upright and upside-down can be told apart.
#: Representative pages measure 0.05 to 0.13; a page set entirely in capitals or
#: digits has no ascenders or descenders to speak of and measures near zero.
#: Below this the page is left the way it came and the caller is told the 180
#: question went unanswered, which is better than a coin toss reported as fact.
MIN_HEAD_ROOM = 0.03
#: How many times a band must repeat down the page before it counts as text.
#: A photograph's broad tonal bands can out-swing a text comb, but they repeat
#: three or four times across the whole frame where text repeats dozens.
MIN_LINE_REPEATS = 8


def ink_plane(plane: np.ndarray) -> np.ndarray:
    """Turn luminance into ink: 0 where the paper is, 1 at the darkest stroke.

    The paper level is the 85th percentile rather than the maximum, so one torn
    white corner or a blown highlight cannot set the scale for the whole page.
    """
    paper = float(np.percentile(plane, 85.0))
    ink = np.clip(paper - np.asarray(plane, dtype=np.float64), 0.0, None)
    peak = float(ink.max())
    if peak <= 0.0:
        return np.zeros_like(ink)
    return ink / peak


def _shear_margin(width: int, degrees: float) -> int:
    """Rows of headroom a shear of ``degrees`` needs at each end of the profile."""
    return int(np.ceil(width * 0.5 * abs(np.tan(np.radians(degrees))))) + 1


class ProfileBank:
    """One ink plane, ready to be sheared to any angle within its margin.

    The index arithmetic is the expensive part of a projection profile, so the
    row base, the centred column offsets and the scratch buffers are built once
    and reused for every candidate angle.

    Two shears are offered because they cost an order of magnitude apart.
    :meth:`profile` rounds each column to a whole row, which is one integer
    ``bincount`` and fast enough to sweep the whole search range. Rounding
    quantises the answer, though: for a nearly upright page every angle under
    about a tenth of a degree shears to the same integer offsets, so the coarse
    sweep can only ever bracket the truth, never resolve it, and it carries a
    small bias of its own. :meth:`profile_fine` splits each column between the
    two rows it falls between, which makes the score a smooth function of the
    angle at roughly ten times the cost. The search uses the cheap one to find
    the neighbourhood and the smooth one to land inside it.
    """

    def __init__(self, ink: np.ndarray, max_degrees: float) -> None:
        self.ink = ink
        self.height, self.width = ink.shape
        self.weights = ink.ravel()
        self.total = float(self.weights.sum())
        self.margin = _shear_margin(self.width, max_degrees)
        self._length = self.height + 2 * self.margin
        self._base = np.arange(self.height, dtype=np.int64)[:, None] + self.margin
        self._float_base = self._base.astype(np.float64)
        self._centred = np.arange(self.width, dtype=np.float64) - (self.width - 1) / 2.0
        self._rows = np.empty((self.height, self.width), dtype=np.int64)

    @property
    def usable(self) -> bool:
        """True when enough full rows survive the shear margin to mean anything."""
        return self.total > 0.0 and self.height - 2 * self.margin >= 8

    def _offsets(self, degrees: float) -> np.ndarray:
        """Row offset each column gets when the plane is sheared by ``degrees``."""
        return self._centred * np.tan(np.radians(degrees))

    def profile(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, each column rounded to a row."""
        np.add(
            self._base, np.rint(self._offsets(degrees)).astype(np.int64)[None, :],
            out=self._rows,
        )
        summed = np.bincount(
            self._rows.ravel(), weights=self.weights, minlength=self._length
        )
        return summed[2 * self.margin: self.height]

    def profile_fine(self, degrees: float) -> np.ndarray:
        """Row sums after shearing by ``degrees``, split between adjacent rows.

        Each column lands between two rows and gives each its share, so moving
        the angle by a hundredth of a degree moves the profile a little rather
        than not at all. This is what makes sub-tenth-of-a-degree agreement
        with a known rotation possible.
        """
        rows = self._float_base + self._offsets(degrees)[None, :]
        lower = np.floor(rows)
        upper_share = (rows - lower).ravel()
        index = lower.astype(np.int64).ravel()
        length = self._length + 1
        summed = np.bincount(
            index, weights=self.weights * (1.0 - upper_share), minlength=length
        )
        summed += np.bincount(
            index + 1, weights=self.weights * upper_share, minlength=length
        )
        return summed[2 * self.margin: self.height]


def comb_score(profile: np.ndarray) -> float:
    """How comb-like a profile is: the energy of its first difference.

    Row variance also peaks at the right angle, but the difference is blind to a
    page-wide lighting ramp, which variance happily rewards.
    """
    if profile.size < 2:
        return 0.0
    step = np.diff(profile)
    return float(np.dot(step, step))


def _search(bank: ProfileBank) -> Tuple[float, float]:
    """Coarse-to-fine hunt for the sharpest comb; returns ``(degrees, score)``.

    The first sweep is the cheap rounded shear across the whole range, which
    only has to land within a degree of the answer. Every later sweep uses the
    smooth shear, so the final step resolves the angle rather than the pixel
    grid.
    """
    low, high = -MAX_SKEW_DEGREES, MAX_SKEW_DEGREES
    best, best_score = 0.0, -1.0
    for index, step in enumerate(SEARCH_STEPS):
        if index == 0:
            start, stop = low, high
            shear = bank.profile
        else:
            span = SEARCH_STEPS[index - 1]
            start, stop = max(low, best - span), min(high, best + span)
            shear = bank.profile_fine
        count = int(round((stop - start) / step)) + 1
        angles = np.linspace(start, stop, max(count, 1))
        scores = [comb_score(shear(angle)) for angle in angles]
        pick = int(np.argmax(scores))
        best, best_score = float(angles[pick]), float(scores[pick])
    return best, best_score


def estimate_skew_on_plane(plane: np.ndarray) -> Tuple[float, float]:
    """Skew of one prepared plane; returns ``(degrees, line_strength)``.

    ``degrees`` is how far the text has been turned counter-clockwise from
    horizontal. ``line_strength`` is the comb score divided by the page's total
    ink, so it compares across pages and across the two ways round.
    """
    if plane.size == 0 or min(plane.shape) < MIN_PLANE_SIDE:
        return 0.0, 0.0
    bank = ProfileBank(ink_plane(plane), MAX_SKEW_DEGREES)
    if not bank.usable:
        return 0.0, 0.0
    degrees, score = _search(bank)
    return degrees, score / max(bank.total, 1e-9)


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
    #: How much taller the ascender zone is than the descender zone, as a share
    #: of the line's inked height. Upright Latin script is positive.
    head_room: float = 0.0
    #: Share of profile rows that are inside a line band rather than a gap.
    line_fill: float = 0.0
    #: Share of the page's ink that falls inside a line band rather than between.
    band_mass: float = 0.0


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


def _head_room(
    profile: np.ndarray,
    bands: Sequence[Tuple[int, int]],
    pitch: float,
    low_cut: float,
    high_cut: float,
) -> float:
    """How much taller a line's ascender zone is than its descender zone.

    This is the only thing on a page that says which way up it is, and it is a
    fact about the script rather than about the ink: in Latin type the
    ascenders reach roughly three quarters of an em above the baseline while
    the descenders drop only a fifth of an em below it. So the dense body of a
    line - x-height top down to baseline - sits with more empty room above it
    than below.

    Each line is framed by half its own pitch either side of its centre, so the
    frame does not depend on where a threshold happened to cut. Inside the
    frame a low cut gives the full inked span and a high cut gives the dense
    body; the answer is the difference between the room above and the room
    below, as a share of the span. Positive means upright.

    The median across lines is taken, not the mean, so a few headings or a
    table rule cannot swing the page. A page set entirely in capitals or
    figures has neither ascenders nor descenders and honestly measures near
    zero - which is why callers check the result against
    :data:`MIN_HEAD_ROOM` before believing it.
    """
    frame = max(2, int(round(pitch / 2.0)))
    values: List[float] = []
    for start, stop in bands:
        centre = (start + stop) // 2
        low, high = centre - frame, centre + frame + 1
        if low < 0 or high > profile.size:
            continue
        segment = profile[low:high]
        inked = np.flatnonzero(segment > low_cut)
        core = np.flatnonzero(segment > high_cut)
        if inked.size < 3 or core.size < 1:
            continue
        span = float(inked[-1] - inked[0])
        if span <= 0.0:                 # pragma: no cover - inked.size >= 3
            continue
        above = float(core[0] - inked[0])
        below = float(inked[-1] - core[-1])
        values.append((above - below) / span)
    if not values:
        return 0.0
    return float(np.median(values))


def line_geometry(plane: np.ndarray, degrees: float) -> LineGeometry:
    """Measure line height, pitch and headroom from the deskewed profile.

    Give this the work plane rather than the skew plane. The angle is found on
    the small plane because the search visits hundreds of candidates, but the
    geometry is one profile at one angle, and measuring a line only a few
    pixels tall is how a text height comes back a third too big.
    """
    if plane.size == 0 or min(plane.shape) < MIN_PLANE_SIDE:
        return LineGeometry()
    ink = ink_plane(plane)
    bank = ProfileBank(ink, max(abs(degrees), 0.01))
    if not bank.usable:
        return LineGeometry()
    profile = bank.profile_fine(degrees)

    floor = float(np.percentile(profile, 5.0))
    ceiling = float(np.percentile(profile, 97.0))
    swing = ceiling - floor
    mean = float(profile.mean())
    contrast = swing / mean if mean > 1e-9 else 0.0
    geometry = LineGeometry(line_contrast=contrast)
    if swing <= 1e-9 or contrast < MIN_LINE_CONTRAST:
        return geometry

    bands = _runs(profile, floor + LINE_THRESHOLD * swing, MIN_LINE_ROWS)
    geometry.line_count = len(bands)
    if not bands:
        return geometry
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
    floor_lo = floor + EXTENT_THRESHOLD * swing
    smallest = max(MIN_LINE_ROWS, int(round((geometry.pitch or 0.0) * 0.2)))
    extents = _runs(profile, floor_lo, smallest)
    if extents:
        heights = [stop - start for start, stop in extents]
        geometry.text_height = float(np.median(heights))
        geometry.line_fill = float(sum(heights)) / float(profile.size)
    else:  # pragma: no cover - a lower cut always finds at least the bands above
        geometry.text_height = float(np.median([b - a for a, b in bands]))
    if geometry.pitch:
        geometry.head_room = _head_room(
            profile, bands, geometry.pitch, floor_lo,
            floor + CORE_THRESHOLD * swing,
        )
    return geometry


def holds_text_lines(geometry: LineGeometry, rows: int) -> bool:
    """True when a profile looks like rows of text rather than broad tone.

    Two things have to hold at once. The profile must swing hard relative to
    its own mean, which paper between lines of ink does and an evenly exposed
    photograph does not; and the band it swings with must be fine enough to
    repeat :data:`MIN_LINE_REPEATS` times down ``rows``. The second test is
    what keeps a photograph of sky over land, which swings plenty across three
    broad bands, from being read as a page of text.
    """
    if geometry.line_contrast < TEXT_LINE_CONTRAST:
        return False
    if geometry.pitch is None or geometry.pitch <= 0:
        return False
    return geometry.pitch * MIN_LINE_REPEATS <= rows


def detect_orientation_on_plane(
    plane: np.ndarray,
    detail: Optional[np.ndarray] = None,
) -> Tuple[int, str]:
    """Quarter-turn that sets this plane upright; returns ``(degrees, basis)``.

    The question is really two questions, and they are answered by different
    evidence. Which way the lines *run* is settled by the projection comb,
    which is unambiguous: text only combs one way round. Which way up they
    *read* is settled by :func:`_head_room`, which is a weaker signal and
    sometimes has nothing to say at all.

    Args:
        plane: the page, small enough for two skew searches.
        detail: the same page at higher resolution, used for the up-versus-down
            test, which needs more than a few pixels per line to mean anything.
            Defaults to ``plane``.

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
    if plane.size == 0 or min(plane.shape) < MIN_PLANE_SIDE:
        return 0, "none"
    if detail is None or min(detail.shape) < MIN_PLANE_SIDE:
        detail = plane
    _, upright_score = estimate_skew_on_plane(plane)
    turned = np.ascontiguousarray(np.rot90(plane, 1))
    _, turned_score = estimate_skew_on_plane(turned)
    if max(upright_score, turned_score) <= 0.0:
        return 0, "none"

    # np.rot90(x, 1) turns the page counter-clockwise. If that is the one whose
    # lines run across, the correction is that same counter-clockwise quarter.
    if upright_score >= turned_score:
        candidate, fine, base = plane, detail, 0
    else:
        candidate = turned
        fine = np.ascontiguousarray(np.rot90(detail, 1))
        base = 90
    degrees, _ = estimate_skew_on_plane(candidate)
    geometry = line_geometry(fine, degrees)
    if not holds_text_lines(geometry, fine.shape[0]):
        # Neither way round produced a comb, so there are no lines of text here
        # and there is nothing to be upright about. Saying 0 would look like a
        # finding; "none" says the question could not be answered.
        return 0, "none"
    if geometry.line_count < 3 or abs(geometry.head_room) < MIN_HEAD_ROOM:
        return base, "lines"
    if geometry.head_room >= 0:
        return base, "lines+headroom"
    return (base + 180) % 360, "lines+headroom"
