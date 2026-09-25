"""The built-in classical counter. It counts blobs, not objects.

Steps, all plain numpy:

1. Estimate the background: the median colour of the counted area, then a
   smooth quadratic surface fitted to the pixels that look like background, so
   a gentle gradient or vignette is not mistaken for a blob.
2. Measure how far every pixel is from that background (the largest difference
   over the channels) and threshold it: Otsu's method, but never below a floor
   set by the image noise and a minimum contrast.
3. Label the connected foreground regions and drop the ones under ``min_area``.
4. Split a region where a narrow neck joins two rounder parts. This works on a
   distance-to-edge map: two lobes are kept apart when the neck between them is
   narrower than ``NECK_RATIO`` of the smaller lobe's width. Things that touch
   along a broad edge have no neck and stay one blob.
5. Drop what is over ``max_area``.

None of this knows what an object is. It is dependable for high-contrast,
separable things on a plain background and has no idea about anything else.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._labels import Components, label

logger = logging.getLogger(__name__)

#: A pixel must differ from the background by at least this much (0-255 scale).
MIN_CONTRAST = 10.0
#: ... and by at least this many noise standard deviations.
NOISE_SIGMAS = 5.0
#: Split two lobes when the neck is narrower than this fraction of the smaller lobe.
NECK_RATIO = 0.75
#: A lobe must rise at least this many pixels above the neck to count.
MIN_PERSISTENCE = 2
#: A lobe must be at least this many pixels deep (about its radius) to count.
MIN_LOBE_DEPTH = 3
#: At most this many distance levels are examined per blob, to bound the time.
MAX_LEVELS = 48
#: Blobs wider than this many pixels have their necks found on a reduced copy.
MAX_ANALYSIS_SIDE = 160
#: Statistics are taken from about this many pixels, however large the image.
_MAX_STAT_PIXELS = 1_000_000
#: Default min_area: this fraction of the counted area, but never under the floor.
DEFAULT_MIN_AREA_FRACTION = 2e-5
DEFAULT_MIN_AREA_FLOOR = 9
#: Fewer background samples than this and the surface fit is skipped.
_MIN_FIT_SAMPLES = 64
_MAX_FIT_SAMPLES = 60000


@dataclass
class Blob:
    """One counted blob. Box edges are pixel indices, right and bottom exclusive."""

    left: int
    top: int
    right: int
    bottom: int
    area: int
    cx: float
    cy: float
    contrast: float
    split: bool = False
    touches_edge: bool = False
    touches_region_edge: bool = False


@dataclass
class ClassicalOutcome:
    """Everything the classical counter found and decided."""

    blobs: List[Blob]
    threshold: float
    otsu: float
    floor: float
    noise: float
    background_level: float
    background_polarity: str
    background_model: str
    background_variation: float
    specks: int
    too_large: int
    split_blobs: int
    split_into: int
    foreground_fraction: float
    min_area: float
    max_area: Optional[float]
    counted_pixels: int
    notes: List[str] = field(default_factory=list)


def default_min_area(counted_pixels: int) -> int:
    """The ``min_area`` used when none is given."""
    return int(max(DEFAULT_MIN_AREA_FLOOR, round(DEFAULT_MIN_AREA_FRACTION * counted_pixels)))


# --------------------------------------------------------------------------- background

def _noise_sigma(data: np.ndarray, usable: np.ndarray) -> float:
    """Robust noise level from neighbour differences, so objects barely affect it."""
    sigmas = []
    h, w = usable.shape
    stride = max(1, int(np.ceil(h * w / _MAX_STAT_PIXELS)))
    for axis in (1, 0):
        if data.shape[axis] < 2:
            continue
        if axis == 1:
            diff = data[::stride, 1:, :] - data[::stride, :-1, :]
            both = usable[::stride, 1:] & usable[::stride, :-1]
        else:
            top = np.arange(0, h - 1, stride)
            diff = data[top + 1] - data[top]
            both = usable[top + 1] & usable[top]
        if not both.any():
            continue
        for ch in range(data.shape[2]):
            med = float(np.median(np.abs(diff[:, :, ch][both])))
            sigmas.append(1.4826 * med / np.sqrt(2.0))
    return float(max(sigmas)) if sigmas else 0.0


def _distance(data: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Largest absolute difference from a flat ``background`` colour (c,) over channels."""
    h, w, c = data.shape
    out = np.zeros((h, w), dtype=np.float32)
    tmp = np.empty((h, w), dtype=np.float32)
    for ch in range(c):
        np.subtract(data[:, :, ch], np.float32(background[ch]), out=tmp)
        np.abs(tmp, out=tmp)
        np.maximum(out, tmp, out=out)
    return out


def _otsu(values: np.ndarray) -> float:
    """Otsu's threshold on 0-255 values; a flat optimum resolves to its middle."""
    if values.size == 0:
        return 0.0
    bins = np.clip(values, 0, 255).astype(np.int64)
    hist = np.bincount(bins, minlength=256).astype(np.float64)
    total = hist.sum()
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - total * m0) ** 2 / (w0 * w1 * total)
    between[~np.isfinite(between)] = 0.0
    best = between.max()
    if best <= 0:
        return 0.0
    ties = np.flatnonzero(between >= best * (1.0 - 1e-9))
    k = int(ties[len(ties) // 2])
    return float(k + 1)


def _surface_terms(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(w) + 0.5) / max(w, 1) * 2.0 - 1.0
    ys = (np.arange(h) + 0.5) / max(h, 1) * 2.0 - 1.0
    return xs, ys


def _design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _fit_surface(data: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares quadratic surface per channel through the sample pixels: (6, c) coefficients."""
    h, w, c = data.shape
    xs, ys = _surface_terms(h, w)
    design = _design(xs[cols], ys[rows])
    target = data[rows, cols, :].astype(np.float64)
    try:
        coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 6:
        # Too few distinct positions for a quadratic (a thin strip): use a plane.
        try:
            coef3, _, rank3, _ = np.linalg.lstsq(design[:, :3], target, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if rank3 < 3:
            return None
        coef = np.vstack([coef3, np.zeros((3, c))])
    return coef


def _surface_at(coef: np.ndarray, rows: np.ndarray, cols: np.ndarray, h: int, w: int) -> np.ndarray:
    xs, ys = _surface_terms(h, w)
    return _design(xs[cols], ys[rows]) @ coef


def _distance_to_surface(data: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Like :func:`_distance`, against the fitted surface, without building it whole."""
    h, w, c = data.shape
    xs, ys = _surface_terms(h, w)
    out = np.zeros((h, w), dtype=np.float32)
    tmp = np.empty((h, w), dtype=np.float32)
    for ch in range(c):
        a0, a1, a2, a3, a4, a5 = coef[:, ch]
        per_row = (a0 + a2 * ys + a5 * ys * ys).astype(np.float32)      # terms in y only
        slope = (a1 + a4 * ys).astype(np.float32)                        # multiplies x
        per_col = (a3 * xs * xs).astype(np.float32)                      # terms in x only
        np.multiply(slope[:, None], xs.astype(np.float32)[None, :], out=tmp)
        tmp += per_row[:, None]
        tmp += per_col[None, :]
        np.subtract(data[:, :, ch], tmp, out=tmp)
        np.abs(tmp, out=tmp)
        np.maximum(out, tmp, out=out)
    return out


def _border_ring(usable: np.ndarray) -> np.ndarray:
    """Usable pixels on the edge of the usable area (the image edge counts as outside)."""
    h, w = usable.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = usable
    inner = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
             & padded[1:-1, :-2] & padded[1:-1, 2:])
    return usable & ~inner


def _threshold(dist: np.ndarray, usable: np.ndarray, floor: float) -> Tuple[float, float]:
    otsu = _otsu(dist[usable])
    return max(otsu, floor), otsu


def _background(data: np.ndarray, usable: np.ndarray, floor: float,
                start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str, float, np.ndarray]:
    """Return ``(distance map, model name, surface variation, background colour)``.

    ``start`` is the first guess at the background colour; by default the median.
    """
    rows_all, cols_all = np.nonzero(usable)
    step = max(1, int(np.ceil(rows_all.size / _MAX_FIT_SAMPLES)))
    rows_s, cols_s = rows_all[::step], cols_all[::step]
    flat = np.median(data[rows_s, cols_s], axis=0) if start is None else start
    dist = _distance(data, flat)
    model, variation, colour = "flat", 0.0, flat
    best = dist
    for _ in range(2):
        cut, _ = _threshold(best, usable, floor)
        quiet = best[rows_s, cols_s] < cut
        if int(quiet.sum()) < _MIN_FIT_SAMPLES:
            break
        coef = _fit_surface(data, rows_s[quiet], cols_s[quiet])
        if coef is None:
            break
        best = _distance_to_surface(data, coef)
        at_samples = _surface_at(coef, rows_s, cols_s, data.shape[0], data.shape[1])
        variation = float((at_samples.max(axis=0) - at_samples.min(axis=0)).max())
        colour = np.median(at_samples, axis=0)
        model = "surface"
    if model == "surface" and variation < 2.0:
        model = "flat"
    return best, model, variation, colour


# --------------------------------------------------------------------------- splitting

def _shift_max(img: np.ndarray) -> np.ndarray:
    """Largest value among the 8 neighbours (the border must be zero padding)."""
    out = np.zeros_like(img)
    core = out[1:-1, 1:-1]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            np.maximum(core, img[1 + dy:img.shape[0] - 1 + dy, 1 + dx:img.shape[1] - 1 + dx], out=core)
    return out


def _depth(mask: np.ndarray) -> np.ndarray:
    """Distance to the edge by repeated erosion (alternating square and cross: octagonal)."""
    depth = np.zeros(mask.shape, dtype=np.int32)
    cur = mask.copy()
    k = 0
    while cur.any():
        k += 1
        depth[cur] = k
        inner = (cur[1:-1, 1:-1] & cur[:-2, 1:-1] & cur[2:, 1:-1]
                 & cur[1:-1, :-2] & cur[1:-1, 2:])
        if k % 2 == 1:
            inner &= cur[:-2, :-2] & cur[:-2, 2:] & cur[2:, :-2] & cur[2:, 2:]
        cur = np.zeros_like(cur)
        cur[1:-1, 1:-1] = inner
    return depth


def _regional_maxima(depth: np.ndarray, min_depth: int) -> int:
    """How many plateaus of ``depth`` have no higher neighbour, among those >= ``min_depth``."""
    h, w = depth.shape
    lower = (depth == 0) | (_shift_max(depth) > depth)
    while True:
        spread = lower.copy()
        core = spread[1:-1, 1:-1]
        centre = depth[1:-1, 1:-1]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb_lower = lower[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
                nb_depth = depth[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
                core |= nb_lower & (nb_depth == centre)
        if np.array_equal(spread, lower):
            break
        lower = spread
    return label(~lower & (depth >= min_depth)).count


def _levels(max_depth: int) -> List[int]:
    step = max(1, int(np.ceil(max_depth / MAX_LEVELS)))
    levels = list(range(max_depth, 0, -step))
    if levels[-1] != 1:
        levels.append(1)
    return levels


@dataclass
class _Lobe:
    peak: int
    row: int
    col: int
    level: int
    comp: int
    death: Optional[int] = None


def _lobes(depth: np.ndarray, levels: List[int]) -> Tuple[List[_Lobe], Dict[int, Components]]:
    lobes: List[_Lobe] = []
    alive: List[int] = []
    kept: Dict[int, Components] = {}
    for level in levels:
        comps = label(depth >= level)
        if comps.count == 0:
            continue
        lab = comps.label_image()
        owners: Dict[int, List[int]] = {}
        for i in alive:
            owners.setdefault(int(lab[lobes[i].row, lobes[i].col]), []).append(i)
        born = False
        for k in range(1, comps.count + 1):
            if k not in owners:
                lobes.append(_Lobe(level, int(comps.first_rows[k - 1]), int(comps.first_cols[k - 1]),
                                   level, k))
                alive.append(len(lobes) - 1)
                born = True
        if born:
            kept[level] = comps
        for members in owners.values():
            if len(members) > 1:
                members.sort(key=lambda i: (-lobes[i].peak, i))
                for i in members[1:]:
                    lobes[i].death = level
                    alive.remove(i)
    return lobes, kept


def _marker(lobe: _Lobe, kept: Dict[int, Components]) -> Tuple[np.ndarray, np.ndarray]:
    comps = kept[lobe.level]
    sel = comps.run_labels == lobe.comp
    rows, starts, ends = comps.run_rows[sel], comps.run_starts[sel], comps.run_ends[sel]
    lengths = ends - starts
    rr = np.repeat(rows, lengths)
    cc = np.repeat(starts, lengths) + (np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths))
    return rr, cc


def _assign(mask: np.ndarray, geometry: List[Tuple[float, float, float]]) -> np.ndarray:
    """Give every pixel to the lobe whose centre is nearest relative to the lobe's radius.

    ``geometry`` holds ``(row, col, radius)`` per lobe. For two round lobes this
    cut passes through the neck whatever their sizes (it is the circle of
    Apollonius through the point where they touch).
    """
    rows, cols = np.nonzero(mask)
    cy = np.array([g[0] for g in geometry])[None, :]
    cx = np.array([g[1] for g in geometry])[None, :]
    radii = np.array([g[2] for g in geometry])[None, :]
    weighted = np.hypot(rows[:, None] - cy, cols[:, None] - cx) / radii
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[rows, cols] = np.argmin(weighted, axis=1).astype(np.int32) + 1
    return labels


def _reduce(mask: np.ndarray, factor: int) -> np.ndarray:
    """Majority vote over ``factor`` x ``factor`` blocks, keeping a False border."""
    inner = mask[1:-1, 1:-1]
    h, w = inner.shape
    rows, cols = -(-h // factor), -(-w // factor)
    padded = np.zeros((rows * factor, cols * factor), dtype=np.float64)
    padded[:h, :w] = inner
    votes = padded.reshape(rows, factor, cols, factor).mean(axis=(1, 3))
    small = np.zeros((rows + 2, cols + 2), dtype=bool)
    small[1:-1, 1:-1] = votes >= 0.5
    return small


def split_blob(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Split a single connected blob at narrow necks.

    ``mask`` must have a one-pixel False border. Returns an int label image
    (0 background, 1..n pieces) when the blob splits into two or more pieces
    that each reach ``min_area``, otherwise None. Blobs wider than
    ``MAX_ANALYSIS_SIDE`` are analysed on a reduced copy to bound the time; the
    cut itself is always made on the full-size mask.
    """
    side = max(mask.shape[0], mask.shape[1]) - 2
    factor = max(1, int(np.ceil(side / MAX_ANALYSIS_SIDE)))
    work = mask if factor == 1 else _reduce(mask, factor)
    depth = _depth(work)
    max_depth = int(depth.max())
    if max_depth < MIN_LOBE_DEPTH:
        return None
    # Only a regional maximum at least MIN_LOBE_DEPTH deep can become a second lobe.
    if _regional_maxima(depth, MIN_LOBE_DEPTH) <= 1:
        return None
    levels = _levels(max_depth)
    lobes, kept = _lobes(depth, levels)
    chosen = [
        lobe for lobe in lobes
        if lobe.death is None
        or (lobe.peak >= MIN_LOBE_DEPTH
            and lobe.peak - lobe.death >= MIN_PERSISTENCE
            and lobe.death <= NECK_RATIO * lobe.peak)
    ]
    geometry: List[Tuple[float, float, float]] = []
    for lobe in chosen:
        rr, cc = _marker(lobe, kept)
        row, col = float(rr.mean()), float(cc.mean())
        if factor > 1:
            row = (row - 1) * factor + (factor - 1) / 2.0 + 1
            col = (col - 1) * factor + (factor - 1) / 2.0 + 1
        geometry.append((row, col, float(lobe.peak * factor)))
    while len(geometry) >= 2:
        labels = _assign(mask, geometry)
        areas = np.bincount(labels.reshape(-1), minlength=len(geometry) + 1)[1:]
        if areas.min() >= min_area:
            return labels
        geometry.pop(int(np.argmin(areas)))     # too small to count alone: give it back
    return None


# --------------------------------------------------------------------------- counting

def count_blobs(data: np.ndarray, usable: np.ndarray, region_mask: Optional[np.ndarray],
                min_area: Optional[float], max_area: Optional[float]) -> ClassicalOutcome:
    """Run the classical pipeline on ``data`` (h, w, c; 0-255) where ``usable`` is True."""
    h, w, _ = data.shape
    counted = int(usable.sum())
    min_used = float(default_min_area(counted) if min_area is None else min_area)
    notes: List[str] = []
    empty = ClassicalOutcome(
        blobs=[], threshold=0.0, otsu=0.0, floor=MIN_CONTRAST, noise=0.0,
        background_level=0.0, background_polarity="none", background_model="flat",
        background_variation=0.0, specks=0, too_large=0, split_blobs=0, split_into=0,
        foreground_fraction=0.0, min_area=min_used, max_area=max_area,
        counted_pixels=counted, notes=notes,
    )
    if counted == 0:
        notes.append("no usable pixels to count (all NaN, or none inside the region)")
        return empty

    noise = _noise_sigma(data, usable)
    floor = max(MIN_CONTRAST, NOISE_SIGMAS * noise)
    dist, model, variation, colour = _background(data, usable, floor)
    threshold, otsu = _threshold(dist, usable, floor)

    # The median is the background only while things cover less than half the
    # picture. If the "foreground" covers most of the border, the two were
    # probably swapped: try the other class as background and keep whichever
    # reading leaves the border clearer.
    ring = _border_ring(usable)
    fg_try = (dist >= threshold) & usable
    if ring.any() and fg_try.sum() > 0.3 * counted and fg_try[ring].mean() > 0.5:
        rr, cc = np.nonzero(fg_try)
        step = max(1, int(np.ceil(rr.size / _MAX_FIT_SAMPLES)))
        other = np.median(data[rr[::step], cc[::step]], axis=0)
        dist_b, model_b, variation_b, colour_b = _background(data, usable, floor, start=other)
        threshold_b, otsu_b = _threshold(dist_b, usable, floor)
        fg_b = (dist_b >= threshold_b) & usable
        if fg_b.any() and fg_b[ring].mean() < fg_try[ring].mean():
            dist, model, variation, colour = dist_b, model_b, variation_b, colour_b
            threshold, otsu = threshold_b, otsu_b
            notes.append(
                "the things cover about half the picture, so the background was taken to be "
                "what fills the border rather than the most common colour"
            )

    fg = (dist >= threshold) & usable
    level = float(np.mean(colour[: max(1, colour.size - (1 if data.shape[2] in (2, 4) else 0))]))
    signed = data[fg].mean(axis=0) - colour if fg.any() else np.zeros_like(colour)
    colour_part = signed[: max(1, signed.size - (1 if data.shape[2] in (2, 4) else 0))]
    main = float(colour_part.mean())
    if not fg.any():
        polarity = "none"
    elif colour_part.size > 1 and abs(main) < 0.3 * float(np.abs(colour_part).max()):
        polarity = "colour"
    else:
        polarity = "light" if main < 0 else "dark"


    outcome = ClassicalOutcome(
        blobs=[], threshold=float(threshold), otsu=float(otsu), floor=float(floor),
        noise=float(noise), background_level=level, background_polarity=polarity,
        background_model=model, background_variation=variation, specks=0, too_large=0,
        split_blobs=0, split_into=0, foreground_fraction=float(fg.sum()) / counted,
        min_area=min_used, max_area=max_area, counted_pixels=counted, notes=notes,
    )
    if not fg.any():
        return outcome

    comps = label(fg)
    lab = comps.label_image()
    region_edge = None
    if region_mask is not None and not region_mask.all():
        inside = region_mask
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = inside
        eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
        region_edge = inside & ~eroded

    blobs: List[Blob] = []
    for k in range(1, comps.count + 1):
        area = int(comps.areas[k - 1])
        if area < min_used:
            outcome.specks += 1
            continue
        top, left = int(comps.tops[k - 1]), int(comps.lefts[k - 1])
        bottom, right = int(comps.bottoms[k - 1]), int(comps.rights[k - 1])
        crop = lab[top:bottom, left:right] == k
        pieces: List[Tuple[np.ndarray, bool]] = []
        if area >= 2 * min_used:
            padded = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), dtype=bool)
            padded[1:-1, 1:-1] = crop
            parts = split_blob(padded, min_used)
            if parts is not None:
                parts = parts[1:-1, 1:-1]
                n_parts = int(parts.max())
                outcome.split_blobs += 1
                outcome.split_into += n_parts
                pieces = [(parts == i, True) for i in range(1, n_parts + 1)]
        if not pieces:
            pieces = [(crop, False)]
        for piece, was_split in pieces:
            rr, cc = np.nonzero(piece)
            p_area = int(rr.size)
            if p_area < min_used:
                outcome.specks += 1
                continue
            if max_area is not None and p_area > max_area:
                outcome.too_large += 1
                continue
            r0, r1 = int(rr.min()) + top, int(rr.max()) + top + 1
            c0, c1 = int(cc.min()) + left, int(cc.max()) + left + 1
            abs_r, abs_c = rr + top, cc + left
            blobs.append(Blob(
                left=c0, top=r0, right=c1, bottom=r1, area=p_area,
                cx=float(abs_c.mean() + 0.5), cy=float(abs_r.mean() + 0.5),
                contrast=float(np.median(dist[abs_r, abs_c])),
                split=was_split,
                touches_edge=(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
                touches_region_edge=bool(region_edge is not None and region_edge[abs_r, abs_c].any()),
            ))
    blobs.sort(key=lambda b: (b.top, b.left, b.bottom, b.right))
    outcome.blobs = blobs
    return outcome
