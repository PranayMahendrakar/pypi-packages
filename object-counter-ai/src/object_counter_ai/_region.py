"""Regions to count inside: a box, a polygon or a boolean mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class Region:
    """A normalised region. ``polygon`` holds the (x, y) corners in pixels."""

    kind: str                          # "box", "polygon" or "mask"
    polygon: Tuple[Point, ...] = ()
    mask: Optional[np.ndarray] = None

    def describe(self) -> str:
        """One line for summaries."""
        if self.kind == "mask":
            assert self.mask is not None
            return f"a mask of {int(self.mask.sum())} pixels"
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.polygon
            return f"box ({_num(x0)}, {_num(y0)}, {_num(x1)}, {_num(y1)})"
        return f"polygon with {len(self.polygon)} corners"

    def to_json(self) -> Any:
        """A JSON-safe description."""
        if self.kind == "mask":
            assert self.mask is not None
            return {"kind": "mask", "pixels": int(self.mask.sum())}
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.polygon
            return {"kind": "box", "box": [_num(x0), _num(y0), _num(x1), _num(y1)]}
        return {"kind": "polygon", "points": [[_num(x), _num(y)] for x, y in self.polygon]}

    def pixel_mask(self, height: int, width: int) -> np.ndarray:
        """Pixels whose centre lies inside the region."""
        if self.kind == "mask":
            assert self.mask is not None
            if self.mask.shape != (height, width):
                raise ValueError(
                    f"the region mask is {self.mask.shape[1]} x {self.mask.shape[0]} pixels "
                    f"but the image is {width} x {height}"
                )
            return self.mask.copy()
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.polygon
            xs = np.arange(width) + 0.5
            ys = np.arange(height) + 0.5
            inside_x = (xs >= x0) & (xs < x1)
            inside_y = (ys >= y0) & (ys < y1)
            return inside_y[:, None] & inside_x[None, :]
        return _rasterise(self.polygon, height, width)

    def contains(self, xs: Any, ys: Any) -> np.ndarray:
        """Which of the points ``(xs[i], ys[i])`` lie inside the region."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if self.kind == "mask":
            assert self.mask is not None
            h, w = self.mask.shape
            cols = np.floor(xs).astype(np.int64)
            rows = np.floor(ys).astype(np.int64)
            ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
            out = np.zeros(xs.shape, dtype=bool)
            out[ok] = self.mask[rows[ok], cols[ok]]
            return out
        if self.kind == "box":
            (x0, y0), _, (x1, y1), _ = self.polygon
            return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        return _points_in_polygon(self.polygon, xs, ys)


def _num(value: float) -> Any:
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


_REGION_HELP = (
    "region must be a box (left, top, right, bottom), a polygon [(x, y), (x, y), (x, y), ...] "
    "or a boolean mask the size of the image"
)


def parse_region(region: Any) -> Optional[Region]:
    """Accept None, a box ``(left, top, right, bottom)``, a polygon ``[(x, y), ...]`` or a mask."""
    if region is None:
        return None
    if isinstance(region, Region):
        return region
    if isinstance(region, np.ndarray) and region.dtype == bool:
        if region.ndim != 2:
            raise ValueError("a region mask must be a 2-D boolean array")
        return Region("mask", mask=np.array(region, dtype=bool, copy=True))
    try:
        arr = np.array(region, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(_REGION_HELP) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("region coordinates must be finite numbers")
    if arr.ndim == 1 and arr.size == 4:
        x0, y0, x1, y1 = (float(v) for v in arr)
        return _box(x0, y0, x1, y1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        if arr.shape[0] == 2:
            (x0, y0), (x1, y1) = arr.tolist()
            return _box(x0, y0, x1, y1)
        points = [(float(x), float(y)) for x, y in arr]
        if len(points) > 3 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) < 3:
            raise ValueError("a polygon region needs at least 3 corners")
        if _signed_area(points) == 0.0:
            raise ValueError("the region polygon has no area")
        return Region("polygon", polygon=tuple(points))
    raise ValueError(f"{_REGION_HELP}; got an array of shape {arr.shape}")


def _box(x0: float, y0: float, x1: float, y1: float) -> Region:
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= left or bottom <= top:
        raise ValueError("the region box has no area")
    return Region("box", polygon=((left, top), (right, top), (right, bottom), (left, bottom)))


def _signed_area(points: Sequence[Point]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, list(points[1:]) + [points[0]]):
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _edges(polygon: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.array(polygon, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return pts[:, 0], pts[:, 1], nxt[:, 0], nxt[:, 1]


def _rasterise(polygon: Sequence[Point], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = _edges(polygon)
    out = np.zeros((height, width), dtype=bool)
    xs = np.arange(width) + 0.5
    lo = max(0, int(np.floor(min(y0.min(), y1.min()))))
    hi = min(height, int(np.ceil(max(y0.max(), y1.max()))) + 1)
    for row in range(lo, hi):
        yc = row + 0.5
        crosses = (y0 <= yc) != (y1 <= yc)
        if not crosses.any():
            continue
        xa, ya, xb, yb = x0[crosses], y0[crosses], x1[crosses], y1[crosses]
        xint = np.sort(xa + (yc - ya) * (xb - xa) / (yb - ya))
        out[row] = (np.searchsorted(xint, xs, side="right") % 2) == 1
    return out


def _points_in_polygon(polygon: Sequence[Point], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = _edges(polygon)
    px = xs.reshape(-1, 1)
    py = ys.reshape(-1, 1)
    crosses = (y0 <= py) != (y1 <= py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    hits = crosses & (xint <= px)
    return (hits.sum(axis=1) % 2 == 1).reshape(xs.shape)
