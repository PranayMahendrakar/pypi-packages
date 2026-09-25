"""Lighting as a gradient, and scanner borders as genuinely black regions.

These two are kept in one file because confusing them is the classic mistake.
A page lit from one side is darker along that side, and a page scanned with the
lid up has a black band along that side, and if darkness is judged against one
page-wide paper level the two look alike. They are not alike, and they want
opposite remedies: a shadow wants more light, a border wants cropping.

So they are told apart by what they are rather than by how dark they are.

**A border is black.** Not grey, not dim paper: close to the darkest ink on the
page, filling most of every row or column it occupies, and running in from the
edge of the image. Being that dark is not enough on its own, because a shadow
deep enough to be unreadable gets there too at the very edge of the sheet. So a
border also has to be empty - no ink on it, because there is no paper there -
and has to end in a step, the edge of the sheet, where the level jumps to paper
within a few pixels. A shadow fades back to paper over hundreds of pixels, with
the text still printed under it. :func:`find_border` looks for exactly that and
nothing else, and :func:`find_outside` holds the black corners of a sheet
scanned crooked to the same test before they are painted out.

**Lighting is a surface.** The paper level is estimated everywhere on the page
- block maxima, then a morphological closing that removes anything dark and
narrower than the window, which is every stroke of text - and the page is
judged by how that surface slopes. The size of the problem is how far the
paper falls from its brightest to its darkest; the place is read from a plane
fitted through the surface, so a shadow along the left side is named "the left
edge" however the noise happens to fall. Everything else in the package
measures ink against this local surface rather than one number for the sheet,
so a shadow never turns into low contrast, show-through or a missing line.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from ._images import percentile_sample

logger = logging.getLogger(__name__)

#: Work-plane pixels per block when the paper surface is estimated.
BLOCK_PX = 8
#: Closing radius, in blocks. The window is ``2 * radius + 1`` blocks across,
#: so with 8-pixel blocks anything dark and narrower than 40 work-plane pixels -
#: every stroke of body text and most headings - is lifted out of the surface.
CLOSING_RADIUS = 2
#: Box-smoothing radius, in blocks, applied after the closing.
SMOOTH_RADIUS = 1
#: A pixel or block darker than this share of the page's paper level is
#: genuinely black. A shadow deep enough to reach it - three quarters of the
#: light gone - is darker than any page anyone should OCR, and even then it is
#: judged by its shape below, not by this level alone.
BLACK_SHARE = 0.25
#: Share of a row or column that has to be genuinely black to be border.
BORDER_LINE_SHARE = 0.80
#: Lines, in work-plane pixels, within which the level must jump from the
#: border's black up to paper. The edge of a sheet lying on a black lid is a
#: step a pixel or two wide; a shadow climbs back to paper over hundreds.
BORDER_EDGE_PX = 3
#: How far that jump must climb, as a share of the paper level.
BORDER_STEP = 0.30
#: A pixel differing from its band's own median by more than this share of the
#: paper level is content - ink on dim paper - and a band holding more than
#: :data:`BORDER_MAX_CONTENT` of such pixels is paper in shadow, not a border.
BORDER_CONTENT_LEVEL = 0.10
BORDER_MAX_CONTENT = 0.01
#: Most of the image that may lie outside the sheet before the dark region is
#: taken for part of the picture instead.
OUTSIDE_MAX_SHARE = 0.45
#: Pixels the outside region is grown by, on the work plane, so the soft edge
#: of the sheet goes with it rather than staying behind as a thin grey line.
OUTSIDE_GROW_PX = 2
#: Most of any one side that a border may take. Deeper than this and the dark
#: region is part of the picture, not an artefact of the scanner.
BORDER_MAX_SIDE = 0.25
#: Rows or columns of non-border tolerated inside a border's ragged inner edge.
BORDER_GAP = 2
#: Border narrower than this share of the side is reported as nothing.
BORDER_MIN_SIDE = 0.004
#: Percentiles of the paper surface taken as its darkest and brightest parts.
#: Not the extremes, so a single scratched block cannot set the swing.
DARK_PERCENTILE = 2.0
BRIGHT_PERCENTILE = 98.0

_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

SIDES = ("top", "bottom", "left", "right")


# --------------------------------------------------------------------------
# small separable filters on a block grid
# --------------------------------------------------------------------------


def _slide(
    grid: np.ndarray, radius: int, axis: int, op: Callable[..., np.ndarray]
) -> np.ndarray:
    """Apply ``op`` (np.maximum / np.minimum) over a window along one axis."""
    if radius <= 0 or grid.shape[axis] < 2:
        return grid.copy()
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    padded = np.pad(grid, pad, mode="edge")
    length = grid.shape[axis]
    out = np.take(padded, np.arange(0, length), axis=axis).copy()
    for offset in range(1, 2 * radius + 1):
        op(out, np.take(padded, np.arange(offset, offset + length), axis=axis), out=out)
    return out


def max_filter(grid: np.ndarray, radius: int) -> np.ndarray:
    """Square max filter of ``2 * radius + 1``, edges replicated."""
    return _slide(_slide(grid, radius, 0, np.maximum), radius, 1, np.maximum)


def min_filter(grid: np.ndarray, radius: int) -> np.ndarray:
    """Square min filter of ``2 * radius + 1``, edges replicated."""
    return _slide(_slide(grid, radius, 0, np.minimum), radius, 1, np.minimum)


def box_mean(grid: np.ndarray, radius: int) -> np.ndarray:
    """Square mean filter of ``2 * radius + 1``, edges replicated."""
    out = grid.astype(np.float64)
    for axis in (0, 1):
        if radius <= 0 or out.shape[axis] < 2:
            continue
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        length = out.shape[axis]
        total = np.zeros_like(out)
        for offset in range(2 * radius + 1):
            total += np.take(padded, np.arange(offset, offset + length), axis=axis)
        out = total / float(2 * radius + 1)
    return out


def _half(plane: np.ndarray) -> np.ndarray:
    """Mean of each 2 x 2 cell, the ragged edge padded by replication."""
    height, width = plane.shape
    padded = np.pad(np.asarray(plane, dtype=np.float32),
                    ((0, height % 2), (0, width % 2)), mode="edge")
    rows, columns = padded.shape
    return padded.reshape(rows // 2, 2, columns // 2, 2).mean(axis=(1, 3))


def box_blur(plane: np.ndarray) -> np.ndarray:
    """3 x 3 mean of ``plane``, edges replicated, as float32."""
    if min(plane.shape) < 3:
        return np.asarray(plane, dtype=np.float32)
    padded = np.pad(np.asarray(plane, dtype=np.float32), 1, mode="edge")
    rows = padded[:-2, :] + padded[1:-1, :] + padded[2:, :]
    out = rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]
    out *= np.float32(1.0 / 9.0)
    return out


def block_max(plane: np.ndarray, block: int) -> np.ndarray:
    """Maximum of each ``block`` x ``block`` tile; a ragged last tile is padded."""
    height, width = plane.shape
    rows = -(-height // block)
    columns = -(-width // block)
    padded = np.pad(
        plane, ((0, rows * block - height), (0, columns * block - width)), mode="edge"
    )
    return padded.reshape(rows, block, columns, block).max(axis=(1, 3))


# --------------------------------------------------------------------------
# the paper surface
# --------------------------------------------------------------------------


@dataclass
class PaperSurface:
    """The paper level everywhere on the page, and what it says about the light."""

    #: Paper level per work-plane pixel, 0 to 1. Genuinely black regions are
    #: filled with the page's paper level, so ink measured against the surface
    #: there still reads as ink.
    surface: np.ndarray
    #: The smoothed surface on the block grid, for anyone who wants the map.
    blocks: np.ndarray
    #: Block grid mask of genuinely black regions, which are not paper at all.
    black: np.ndarray
    #: The page's paper level: the typical block, black regions excluded.
    paper_level: float
    #: Brightest and darkest parts of the paper surface.
    brightest: float
    darkest: float
    #: How far the paper falls from brightest to darkest, as a share of the
    #: brightest. The lighting measure.
    swing: float
    #: Where the dark part is, in words: "the left edge", "the top-right corner".
    dark_zone: str
    #: Fitted plane slope across and down the page, in paper-level units from
    #: one side of the page to the other. Positive means brighter to the right
    #: or towards the bottom.
    slope_across: float = 0.0
    slope_down: float = 0.0
    #: Share of blocks that are genuinely black.
    black_share: float = 0.0
    #: The light alone: a quadratic fitted through the paper surface, on the
    #: block grid (one value per :data:`BLOCK_PX` block). Lighting - a shadow down one side, a ramp across the
    #: sheet, a hot spot - is smooth and low-order; the tones of a photograph
    #: are not, so dividing by this evens out the light and leaves a picture a
    #: picture. ``None`` means the surface is used as it is.
    model: Optional[np.ndarray] = None

    def turned(self, quarter_turns: int) -> "PaperSurface":
        """The same light, for the page turned ``quarter_turns`` x 90 degrees.

        Everything about the light but its direction survives a quarter turn,
        so only the maps are turned and the dark side named again.
        """
        k = int(quarter_turns) % 4
        if k == 0:
            return self
        blocks = np.ascontiguousarray(np.rot90(self.blocks, k))
        black = np.ascontiguousarray(np.rot90(self.black, k))
        usable = ~black if (~black).any() else np.ones_like(black)
        zone, across, down = _zone_from_plane(blocks, usable, self.brightest - self.darkest)
        return dataclasses.replace(
            self,
            surface=np.ascontiguousarray(np.rot90(self.surface, k)),
            blocks=blocks,
            black=black,
            model=None if self.model is None else np.ascontiguousarray(np.rot90(self.model, k)),
            dark_zone=zone,
            slope_across=across,
            slope_down=down,
        )

    def surface_for(self, shape: Tuple[int, int]) -> np.ndarray:
        """The same paper surface resampled onto another plane of this page.

        ``shape`` is ``(rows, columns)`` of a plane covering the same sheet at a
        different scale, possibly a different one on each axis. The surface is
        smooth by construction, so a bilinear resample loses nothing.
        """
        rows, columns = int(shape[0]), int(shape[1])
        if (rows, columns) == tuple(self.surface.shape):
            return self.surface
        # Upsampled from the small block grid, not resampled from the full
        # surface: the same smooth map at a fraction of the cost.
        height, width = self.surface.shape
        grid_rows, grid_columns = self.blocks.shape
        full = (max(1, int(round(grid_columns * BLOCK_PX * columns / float(width)))),
                max(1, int(round(grid_rows * BLOCK_PX * rows / float(height)))))
        stretched = Image.fromarray(np.ascontiguousarray(self.blocks, dtype=np.float32)).resize(
            full, _BILINEAR)
        out = np.asarray(stretched, dtype=np.float32)[:rows, :columns]
        if out.shape != (rows, columns):
            out = np.pad(out, ((0, rows - out.shape[0]), (0, columns - out.shape[1])), mode="edge")
        return np.ascontiguousarray(out)


def resample(surface: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """A smooth float plane resampled bilinearly to ``shape`` (rows, columns)."""
    rows, columns = int(shape[0]), int(shape[1])
    if (rows, columns) == tuple(surface.shape):
        return surface
    stretched = Image.fromarray(np.ascontiguousarray(surface, dtype=np.float32)).resize(
        (max(1, columns), max(1, rows)), _BILINEAR
    )
    return np.asarray(stretched, dtype=np.float32)


def _zone_from_plane(
    blocks: np.ndarray, usable: np.ndarray, swing_units: float
) -> Tuple[str, float, float]:
    """Name where the paper is darkest from a plane fitted through the surface."""
    rows, columns = blocks.shape
    ys, xs = np.nonzero(usable)
    if ys.size < 3:
        return "centre of the page", 0.0, 0.0
    x = (xs + 0.5) / float(columns) * 2.0 - 1.0
    y = (ys + 0.5) / float(rows) * 2.0 - 1.0
    z = blocks[ys, xs]
    design = np.column_stack([np.ones_like(x), x, y])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    across, down = float(coef[1]) * 2.0, float(coef[2]) * 2.0
    tilt = abs(across) + abs(down)
    if swing_units <= 1e-9 or tilt < swing_units / 3.0:
        # The light does not slope one way; it rings or dips. Compare the rim
        # of the page with its middle to say which.
        rim = np.zeros_like(usable)
        edge_r = max(1, rows // 5)
        edge_c = max(1, columns // 5)
        rim[:edge_r, :] = rim[-edge_r:, :] = True
        rim[:, :edge_c] = rim[:, -edge_c:] = True
        rim_level = blocks[rim & usable]
        mid_level = blocks[~rim & usable]
        if rim_level.size and mid_level.size and rim_level.mean() < mid_level.mean():
            return "edges of the page", across, down
        return "centre of the page", across, down
    horizontal = "left" if across > 0 else "right"
    vertical = "top" if down > 0 else "bottom"
    if abs(down) < 0.4 * abs(across):
        return "{0} edge".format(horizontal), across, down
    if abs(across) < 0.4 * abs(down):
        return "{0} edge".format(vertical), across, down
    return "{0}-{1} corner".format(vertical, horizontal), across, down


def paper_surface(work: np.ndarray) -> PaperSurface:
    """Estimate the paper level everywhere on ``work`` and describe the light.

    ``work`` is the 0-1 float working plane. Small pages whose block grid would
    be a handful of cells fall back to one level for the whole sheet.
    """
    height, width = work.shape
    # Block maxima are taken over a lightly blurred copy. Scanners sharpen, and
    # a sharpened stroke wears a thin halo brighter than the paper around it;
    # the maximum of every block touching text would land on that halo, and
    # the paper beside the text would then read as a little darker than
    # paper - a faint grey veil over every gap between lines. Blurring by one
    # pixel folds each halo back into the stroke it belongs to. Averaging each
    # 2 x 2 cell does that at a quarter of the cost of a sliding blur.
    grid = block_max(_half(work), BLOCK_PX // 2)
    if min(grid.shape) < 2 * CLOSING_RADIUS + 2:
        level = float(np.percentile(percentile_sample(work), 95.0))
        flat = np.full(work.shape, level, dtype=np.float32)
        return PaperSurface(
            surface=flat, blocks=np.full(grid.shape, level), black=np.zeros(grid.shape, bool),
            paper_level=level, brightest=level, darkest=level, swing=0.0,
            dark_zone="centre of the page",
        )

    closed = min_filter(max_filter(grid, CLOSING_RADIUS), CLOSING_RADIUS)
    level = float(np.percentile(closed, 75.0))
    black = closed < BLACK_SHARE * level
    usable = ~black
    if not usable.any():
        usable = np.ones_like(black)
        black = np.zeros_like(black)
    level = float(np.median(closed[usable]))
    filled = np.where(black, level, closed)
    smooth = box_mean(filled, SMOOTH_RADIUS)

    values = smooth[usable]
    brightest = float(np.percentile(values, BRIGHT_PERCENTILE))
    darkest = float(np.percentile(values, DARK_PERCENTILE))
    swing = (brightest - darkest) / max(brightest, 1e-6)
    zone, across, down = _zone_from_plane(smooth, usable, brightest - darkest)

    rows, columns = smooth.shape
    stretched = Image.fromarray(np.ascontiguousarray(smooth, dtype=np.float32)).resize(
        (columns * BLOCK_PX, rows * BLOCK_PX), _BILINEAR
    )
    surface = np.asarray(stretched, dtype=np.float32)[:height, :width].copy()
    return PaperSurface(
        surface=surface, blocks=smooth, black=black, paper_level=level,
        brightest=brightest, darkest=darkest, swing=float(max(0.0, swing)),
        dark_zone=zone, slope_across=across, slope_down=down,
        black_share=float(black.mean()),
        model=_light_model(smooth, usable),
    )


def _light_model(blocks: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """A quadratic through the block surface, evaluated on the block grid."""
    rows, columns = blocks.shape
    ys, xs = np.nonzero(usable)
    y_all, x_all = np.mgrid[0:rows, 0:columns]

    def terms(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        u = (x + 0.5) / float(columns) * 2.0 - 1.0
        v = (y + 0.5) / float(rows) * 2.0 - 1.0
        return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])

    if ys.size < 12:
        fitted = np.full(blocks.shape, float(np.median(blocks)), dtype=np.float64)
    else:
        coef, *_ = np.linalg.lstsq(terms(ys, xs), blocks[ys, xs], rcond=None)
        fitted = (terms(y_all.ravel(), x_all.ravel()) @ coef).reshape(blocks.shape)
    return np.clip(fitted, 1e-3, None).astype(np.float32)


# --------------------------------------------------------------------------
# scanner borders
# --------------------------------------------------------------------------


@dataclass
class Border:
    """Genuinely black bands running in from the edges of the image.

    Widths are in native pixels. A side with no border has width 0.
    """

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    #: Share of the image area the border covers.
    area_share: float = 0.0
    #: The black level used, 0 to 1.
    black_level: float = 0.0
    sides: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """True when any side carries a border."""
        return bool(self.sides)

    def box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """The page inside the border as ``(left, top, right, bottom)``."""
        return (self.left, self.top, width - self.right, height - self.bottom)

    def describe(self) -> str:
        """The sides and their widths in words, e.g. ``"left (70 px)"``."""
        parts = [
            "{0} ({1} px)".format(side, getattr(self, side)) for side in self.sides
        ]
        if len(parts) <= 1:
            return "".join(parts)
        return ", ".join(parts[:-1]) + " and " + parts[-1]


def _border_depth(dark_share: np.ndarray, limit: int) -> int:
    """How many leading lines of ``dark_share`` are border, allowing a ragged edge."""
    depth = 0
    gap = 0
    for index in range(min(limit, dark_share.size)):
        if dark_share[index] >= BORDER_LINE_SHARE:
            depth = index + 1
            gap = 0
        else:
            gap += 1
            if gap > BORDER_GAP or depth == 0:
                break
    return depth


def _lines_from(work: np.ndarray, side: str) -> np.ndarray:
    """``work`` arranged so row ``i`` is the ``i``-th line in from ``side``."""
    if side == "top":
        return work
    if side == "bottom":
        return work[::-1, :]
    if side == "left":
        return work.T
    return work[:, ::-1].T


def _band_is_border(work: np.ndarray, side: str, depth: int, paper: float) -> bool:
    """Whether the dark band ``depth`` lines in from ``side`` is a real border.

    Being dark is not enough, because a deep shadow is dark too. A border is
    also *flat* - nothing printed on it, because there is no paper there - and
    it *ends in a step*: the edge of the sheet, where the level jumps to paper
    within a few pixels. A shadow fails both. It climbs back to paper
    gradually, and the ink printed on the paper under it is still there, a
    little darker than the dim paper around it. Cropping a shadow as a border
    would cut into the text it is lying on.
    """
    lines = _lines_from(work, side)
    reach = min(lines.shape[0], depth + BORDER_EDGE_PX + 1)
    if depth < 1 or reach <= depth:
        return False
    levels = np.median(lines[:reach], axis=1)
    inner = float(levels[depth - 1])
    after = float(levels[depth:reach].max())
    if after - inner < BORDER_STEP * paper:
        return False
    band = lines[:max(1, depth - 1)]
    middle = float(np.median(band))
    content = np.abs(band - np.float32(middle)) > np.float32(BORDER_CONTENT_LEVEL * paper)
    return float(content.mean()) <= BORDER_MAX_CONTENT


def find_border(work: np.ndarray, scale: float) -> Border:
    """Find genuinely black bands along the edges of the work plane.

    Only near-black counts: a pixel has to be darker than :data:`BLACK_SHARE`
    of the page's paper level, and a row or column has to be that dark along
    :data:`BORDER_LINE_SHARE` of its length. A line of text fails this, being
    mostly gaps between letters. A band that passes must then also be flat and
    end in a sharp step up to paper (see :func:`_band_is_border`), which a
    lighting shadow - however deep it looks against the bright middle of the
    page - never does.

    ``scale`` maps native pixels to work-plane pixels, so the widths reported
    are native pixels.
    """
    height, width = work.shape
    paper = float(np.percentile(percentile_sample(work), 90.0))
    black_level = BLACK_SHARE * paper
    result = Border(black_level=black_level)
    if paper <= 0.0 or min(height, width) < 16:
        return result
    dark = work < black_level
    row_share = dark.mean(axis=1)
    column_share = dark.mean(axis=0)
    depths = {
        "top": _border_depth(row_share, int(height * BORDER_MAX_SIDE)),
        "bottom": _border_depth(row_share[::-1], int(height * BORDER_MAX_SIDE)),
        "left": _border_depth(column_share, int(width * BORDER_MAX_SIDE)),
        "right": _border_depth(column_share[::-1], int(width * BORDER_MAX_SIDE)),
    }
    depths = {
        side: (depth if depth and _band_is_border(work, side, depth, paper) else 0)
        for side, depth in depths.items()
    }
    minimum = {
        "top": BORDER_MIN_SIDE * height, "bottom": BORDER_MIN_SIDE * height,
        "left": BORDER_MIN_SIDE * width, "right": BORDER_MIN_SIDE * width,
    }
    depths = {side: (d if d >= max(1.0, minimum[side]) else 0) for side, d in depths.items()}
    top, bottom = depths["top"], height - depths["bottom"]
    left, right = depths["left"], width - depths["right"]
    if bottom - top < 8 or right - left < 8:
        return result
    if not any(depths.values()):
        return result
    if float(dark[top:bottom, left:right].mean()) >= BORDER_LINE_SHARE:
        # Dark all the way through: that is a dark picture, not a dark frame
        # around a sheet of paper.
        return result
    inverse = 1.0 / max(scale, 1e-9)
    for side in SIDES:
        if depths[side]:
            # One work-plane pixel of feather so the soft inner edge goes too.
            setattr(result, side, int(np.ceil((depths[side] + 1) * inverse)))
            result.sides.append(side)
    inner = (bottom - top) * (right - left)
    result.area_share = float(1.0 - inner / float(height * width))
    return result


# --------------------------------------------------------------------------
# outside the sheet
# --------------------------------------------------------------------------


def _leading(mask: np.ndarray, axis: int) -> np.ndarray:
    """How many leading ``True`` values each line of ``mask`` has along ``axis``."""
    clear = ~mask
    first = np.argmax(clear, axis=axis)
    first[~clear.any(axis=axis)] = mask.shape[axis]
    return first


def outside_mask(
    plane: np.ndarray,
    black_level: float,
    grow: int,
    step: float = 0.0,
    edge_px: int = BORDER_EDGE_PX,
) -> np.ndarray:
    """Genuinely black pixels reachable in a straight line from the image edge.

    This is the black lid showing around a sheet scanned crooked: a wedge in
    each corner, running in from the edges. Every point of such a wedge can be
    reached from the nearest image edge along its row or its column without
    leaving the black, so the leading black run of every row and column,
    taken from all four sides, covers it - no flood fill needed. Runs shorter
    than two pixels (or ``grow``) are left alone, so a stroke of ink touching
    the image edge is not taken for the lid, and every run kept is extended by
    ``grow`` so the soft edge of the sheet goes with it.

    With ``step`` set, a run only counts when it ends at the edge of a sheet:
    within ``edge_px`` pixels past its end the level has to climb by ``step``.
    A shadow deep enough to pass for black at the very edge of the page fades
    back to paper gradually and never makes that jump, so it stays a shadow.
    """
    height, width = plane.shape
    dark = plane < np.float32(black_level)
    outside = np.zeros((height, width), dtype=bool)
    if not dark.any():
        return outside
    minimum = max(2, int(grow), 1 - int(grow))
    reach_px = max(1, int(edge_px))
    for axis, flip in ((1, False), (1, True), (0, False), (0, True)):
        values = plane if axis == 1 else plane.T
        mask = dark if axis == 1 else dark.T
        if flip:
            values, mask = values[:, ::-1], mask[:, ::-1]
        run = _leading(mask, 1)
        keep = run >= minimum
        if step > 0.0 and keep.any():
            length = values.shape[1]
            lines = np.flatnonzero(keep)
            ends = run[lines]
            before = values[lines, np.clip(ends - 1, 0, length - 1)].astype(np.float64)
            after = before.copy()
            for offset in range(reach_px + 1):
                column = np.clip(ends + offset, 0, length - 1)
                np.maximum(after, values[lines, column], out=after)
            keep[lines[(after - before) < step]] = False
        run = np.where(keep, run + int(grow), 0)
        if not run.any():
            continue
        span = np.arange(values.shape[1])[None, :] < run[:, None]
        if flip:
            span = span[:, ::-1]
        outside |= span if axis == 1 else span.T
    return outside


def find_outside(work: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """The part of the work plane that is black lid, not sheet.

    Returns ``(mask, share, black_level)``. The mask is empty unless the region
    is genuinely black, featureless and no more than
    :data:`OUTSIDE_MAX_SHARE` of the image: a dark photograph also runs in
    from the edges, but it has something in it and usually fills the frame.
    """
    empty = np.zeros(work.shape, dtype=bool)
    if min(work.shape) < 16:
        return empty, 0.0, 0.0
    paper = float(np.percentile(percentile_sample(work), 90.0))
    black_level = BLACK_SHARE * paper
    if paper <= 0.0:
        return empty, 0.0, 0.0
    step = BORDER_STEP * paper
    core = outside_mask(work, black_level, 0, step)
    share = float(core.mean())
    if share <= 0.0 or share > OUTSIDE_MAX_SHARE:
        return empty, 0.0, black_level
    # The content test skips the last pixels before the sheet, where the edge
    # of the paper blurs the black into grey; that is an edge, not content.
    inner = outside_mask(work, black_level, -OUTSIDE_GROW_PX, step)
    values = work[inner if inner.any() else core]
    middle = float(np.median(values))
    content = np.abs(values - np.float32(middle)) > np.float32(BORDER_CONTENT_LEVEL * paper)
    if float(content.mean()) > BORDER_MAX_CONTENT:
        return empty, 0.0, black_level
    mask = outside_mask(work, black_level, OUTSIDE_GROW_PX, step)
    return mask, float(mask.mean()), black_level
