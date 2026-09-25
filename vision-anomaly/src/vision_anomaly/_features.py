"""The hand-built features. Nothing here is learned, and nothing is downloaded.

Every number an image is judged on is computed by this module, from the pixels,
with numpy. There are six families:

===============  ===============================================================
``colour``       per-channel histograms of R, G and B over the whole frame, plus
                 the mean chroma of each grid cell so a colour that moves can be
                 located as well as noticed.
``edges``        per cell, the mean gradient magnitude and the share of pixels
                 whose gradient clears :data:`EDGE_THRESHOLD` - how much edge
                 detail that part of the picture holds.
``orientation``  per cell, a gradient-orientation histogram weighted by gradient
                 magnitude, plus a finer one over the whole frame - which way
                 the detail runs.
``brightness``   per cell, mean luminance.
``contrast``     per cell, the standard deviation of luminance.
``texture``      per cell, the mean absolute Laplacian - fine detail, which
                 separates a smooth surface from a rough one at the same
                 brightness.
===============  ===============================================================

Two properties matter and are deliberate:

* **Every feature is a 0.0 to 1.0 quantity** - a fraction of pixels, a mean of a
  0..1 plane, or an energy on that plane. That is what lets one absolute noise
  floor in :mod:`vision_anomaly._profile` apply to all of them.
* **Most features belong to one grid cell**, recorded in
  :attr:`FeatureSpace.cell_of`. That is the whole reason a result can say
  *where* an image departed and not only *that* it did.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._loading import ANALYSIS_SIZE, LoadedImage

#: Cells on a side of the coarse grid. 4 gives 16 cells, each 64x64 pixels at
#: the default analysis size: coarse enough that a cell's statistics are stable,
#: fine enough that "the top right corner changed" is useful to a person.
DEFAULT_GRID = 4

#: Bins per channel in the colour histograms.
DEFAULT_COLOUR_BINS = 16

#: Orientation bins per cell. Gradient direction is unsigned - an edge running
#: north-south is the same edge either way round - so these span 0 to 180
#: degrees, 45 degrees each.
DEFAULT_ORIENT_BINS = 4

#: Orientation bins for the whole-frame histogram, which can afford to be finer
#: because it pools every pixel.
DEFAULT_GLOBAL_ORIENT_BINS = 8

#: Gradient magnitude, on a 0..1 luminance plane, above which a pixel counts as
#: sitting on an edge. Roughly a 6% step between neighbours - visible, but well
#: under the contrast of a real object boundary.
EDGE_THRESHOLD = 0.06

#: The six feature families, in the order they appear in a vector.
GROUP_NAMES = ("colour", "edges", "orientation", "brightness", "contrast", "texture")

#: One plain sentence per family, used to build :attr:`AnomalyResult.reasons`.
GROUP_DESCRIPTIONS = {
    "colour": "the mix of colours in the frame",
    "edges": "how much edge detail there is, and where",
    "orientation": "the directions the edges run in",
    "brightness": "how bright each part of the frame is",
    "contrast": "local contrast within each part of the frame",
    "texture": "fine texture within each part of the frame",
}

_CHANNEL_NAMES = ("red", "green", "blue")


@dataclass(frozen=True)
class FeatureConfig:
    """How the feature vector is laid out.

    Saved with a profile and checked on load, so a profile fitted with one
    layout can never be used to score against another.

    Attributes:
        analysis_size: edge of the square grid images are resampled onto.
        grid: cells on a side of the coarse grid.
        colour_bins: bins per channel in the colour histograms.
        orient_bins: orientation bins per cell.
        global_orient_bins: orientation bins for the whole-frame histogram.
        edge_threshold: gradient magnitude above which a pixel counts as an edge.
    """

    analysis_size: int = ANALYSIS_SIZE
    grid: int = DEFAULT_GRID
    colour_bins: int = DEFAULT_COLOUR_BINS
    orient_bins: int = DEFAULT_ORIENT_BINS
    global_orient_bins: int = DEFAULT_GLOBAL_ORIENT_BINS
    edge_threshold: float = EDGE_THRESHOLD

    def validate(self) -> "FeatureConfig":
        """Raise a clear :class:`ValueError` if any setting is unusable."""
        if self.grid < 1:
            raise ValueError("grid must be at least 1, got {0}".format(self.grid))
        if self.analysis_size < self.grid * 4:
            raise ValueError(
                "analysis_size must be at least 4 pixels per cell: {0} is too small "
                "for a {1}x{1} grid".format(self.analysis_size, self.grid)
            )
        if self.analysis_size % self.grid:
            raise ValueError(
                "analysis_size {0} must divide evenly by grid {1}".format(
                    self.analysis_size, self.grid
                )
            )
        for name in ("colour_bins", "orient_bins", "global_orient_bins"):
            if int(getattr(self, name)) < 2:
                raise ValueError(
                    "{0} must be at least 2, got {1}".format(name, getattr(self, name))
                )
        if not 0.0 < float(self.edge_threshold) < 1.0:
            raise ValueError(
                "edge_threshold must be between 0 and 1, got {0}".format(
                    self.edge_threshold
                )
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the layout."""
        return {
            "analysis_size": int(self.analysis_size),
            "grid": int(self.grid),
            "colour_bins": int(self.colour_bins),
            "orient_bins": int(self.orient_bins),
            "global_orient_bins": int(self.global_orient_bins),
            "edge_threshold": float(self.edge_threshold),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        """Rebuild a layout from :meth:`to_dict` output."""
        known = cls().to_dict()
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(
                "unknown feature settings in profile: {0}".format(", ".join(unknown))
            )
        return cls(
            analysis_size=int(data.get("analysis_size", ANALYSIS_SIZE)),
            grid=int(data.get("grid", DEFAULT_GRID)),
            colour_bins=int(data.get("colour_bins", DEFAULT_COLOUR_BINS)),
            orient_bins=int(data.get("orient_bins", DEFAULT_ORIENT_BINS)),
            global_orient_bins=int(
                data.get("global_orient_bins", DEFAULT_GLOBAL_ORIENT_BINS)
            ),
            edge_threshold=float(data.get("edge_threshold", EDGE_THRESHOLD)),
        ).validate()


@dataclass(frozen=True)
class FeatureSpace:
    """The names, families and cell of every slot in a feature vector.

    Attributes:
        config: the layout these names were built from.
        names: one readable name per feature, same order as the vector.
        groups: the family each feature belongs to.
        cell_of: the flat cell index each feature describes, or ``-1`` for the
            whole-frame features that belong to no single cell.
    """

    config: FeatureConfig
    names: Tuple[str, ...]
    groups: Tuple[str, ...]
    cell_of: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def cells(self) -> int:
        """How many grid cells there are."""
        return int(self.config.grid) * int(self.config.grid)

    def group_index(self, group: str) -> np.ndarray:
        """Positions of every feature in one family."""
        return np.array(
            [i for i, name in enumerate(self.groups) if name == group], dtype=np.intp
        )

    def cell_index(self, cell: int) -> np.ndarray:
        """Positions of every feature belonging to one grid cell."""
        return np.array(
            [i for i, owner in enumerate(self.cell_of) if owner == cell], dtype=np.intp
        )

    def cell_box(self, cell: int) -> Tuple[int, int, int, int]:
        """``(x, y, width, height)`` of a cell on the analysis grid."""
        grid = int(self.config.grid)
        step = int(self.config.analysis_size) // grid
        row, col = divmod(int(cell), grid)
        return (col * step, row * step, step, step)


def build_space(config: FeatureConfig) -> FeatureSpace:
    """Work out the name, family and cell of every feature for a layout."""
    config = config.validate()
    names: List[str] = []
    groups: List[str] = []
    cells: List[int] = []

    def add(name: str, group: str, cell: int) -> None:
        names.append(name)
        groups.append(group)
        cells.append(cell)

    grid = int(config.grid)

    for channel in _CHANNEL_NAMES:
        for index in range(int(config.colour_bins)):
            low = index / float(config.colour_bins)
            high = (index + 1) / float(config.colour_bins)
            add(
                "colour.{0}_hist[{1:.2f}-{2:.2f}]".format(channel, low, high),
                "colour",
                -1,
            )
    for cell in range(grid * grid):
        add("colour.chroma{0}".format(_cell_tag(cell, grid)), "colour", cell)

    for cell in range(grid * grid):
        add("edges.magnitude{0}".format(_cell_tag(cell, grid)), "edges", cell)
    for cell in range(grid * grid):
        add("edges.density{0}".format(_cell_tag(cell, grid)), "edges", cell)

    step = 180.0 / float(config.orient_bins)
    for cell in range(grid * grid):
        for index in range(int(config.orient_bins)):
            add(
                "orientation.{0}[{1:.0f}-{2:.0f}deg]".format(
                    _cell_tag(cell, grid).strip("."), index * step, (index + 1) * step
                ),
                "orientation",
                cell,
            )
    global_step = 180.0 / float(config.global_orient_bins)
    for index in range(int(config.global_orient_bins)):
        add(
            "orientation.frame[{0:.0f}-{1:.0f}deg]".format(
                index * global_step, (index + 1) * global_step
            ),
            "orientation",
            -1,
        )

    for group in ("brightness", "contrast", "texture"):
        for cell in range(grid * grid):
            add("{0}{1}".format(group, _cell_tag(cell, grid)), group, cell)

    return FeatureSpace(
        config=config,
        names=tuple(names),
        groups=tuple(groups),
        cell_of=tuple(cells),
    )


def _cell_tag(cell: int, grid: int) -> str:
    row, col = divmod(int(cell), int(grid))
    return ".r{0}c{1}".format(row, col)


def describe_features(config: Optional[FeatureConfig] = None) -> str:
    """A table of the feature families and how many slots each one holds.

    Handy for answering "what is this actually looking at?" without opening the
    source.
    """
    space = build_space(config or FeatureConfig())
    lines = [
        "vision-anomaly features: {0} numbers per image, {1}x{1} grid on a "
        "{2}x{2} analysis image".format(
            len(space), space.config.grid, space.config.analysis_size
        )
    ]
    for group in GROUP_NAMES:
        count = int(len(space.group_index(group)))
        lines.append(
            "  {0:<12} {1:>4}  {2}".format(group, count, GROUP_DESCRIPTIONS[group])
        )
    lines.append(
        "Each is standardised against the fitted set with its median and MAD, "
        "so all of them are comparable in robust sigmas."
    )
    return "\n".join(lines)


def _blocks(plane: np.ndarray, grid: int) -> np.ndarray:
    """Reshape an NxN plane into ``(grid*grid, cell_pixels)``, cells in row order."""
    size = plane.shape[0]
    step = size // grid
    return (
        plane.reshape(grid, step, grid, step)
        .transpose(0, 2, 1, 3)
        .reshape(grid * grid, step * step)
    )


def _laplacian(plane: np.ndarray) -> np.ndarray:
    """Second-difference energy, edge pixels padded by repetition.

    ``4*L - (up + down + left + right)``: zero on any smooth ramp, large wherever
    detail changes fast, which is what separates a rough surface from a smooth
    one of the same brightness.
    """
    padded = np.pad(plane, 1, mode="edge")
    out = plane * np.float32(4.0)
    out = out - padded[:-2, 1:-1]
    out = out - padded[2:, 1:-1]
    out = out - padded[1:-1, :-2]
    out = out - padded[1:-1, 2:]
    return out


def extract(image: LoadedImage, space: FeatureSpace) -> np.ndarray:
    """The full feature vector for one loaded image.

    Args:
        image: an image already on the analysis grid.
        space: the layout to compute, from :func:`build_space`.

    Returns:
        A float32 vector of ``len(space)`` numbers, every one of them in the
        0.0 to 1.0 range the profile's noise floor assumes.

    Raises:
        ValueError: the image is not on the analysis grid this space expects.
    """
    config = space.config
    size = int(config.analysis_size)
    if image.rgb.shape[0] != size or image.rgb.shape[1] != size:
        raise ValueError(
            "image is {0}x{1} on the analysis grid but the profile was fitted at "
            "{2}x{2}".format(image.rgb.shape[1], image.rgb.shape[0], size)
        )
    grid = int(config.grid)
    rgb = image.rgb
    luminance = image.luminance
    parts: List[np.ndarray] = []

    # --- colour ------------------------------------------------------------
    for channel in range(3):
        parts.append(_soft_histogram(rgb[:, :, channel].ravel(), int(config.colour_bins)))
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    parts.append(_blocks(chroma, grid).mean(axis=1).astype(np.float32))

    # --- gradients ---------------------------------------------------------
    gy, gx = np.gradient(luminance.astype(np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
    block_magnitude = _blocks(magnitude, grid)
    parts.append(block_magnitude.mean(axis=1).astype(np.float32))
    over = (magnitude > np.float32(config.edge_threshold)).astype(np.float32)
    parts.append(_blocks(over, grid).mean(axis=1).astype(np.float32))

    # Unsigned orientation: an edge running north-south is the same edge either
    # way round, so the direction is folded onto 0..180 degrees.
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    parts.append(_orientation_cells(angle, magnitude, grid, int(config.orient_bins)))
    parts.append(_orientation_global(angle, magnitude, int(config.global_orient_bins)))

    # --- block statistics --------------------------------------------------
    block_luma = _blocks(luminance, grid)
    parts.append(block_luma.mean(axis=1).astype(np.float32))
    parts.append(block_luma.std(axis=1).astype(np.float32))
    detail = np.abs(_laplacian(luminance))
    # A Laplacian of a 0..1 plane can reach 4; dividing by 4 keeps every feature
    # on the one 0..1 scale the noise floor is defined against.
    parts.append((_blocks(detail, grid).mean(axis=1) / np.float32(4.0)).astype(np.float32))

    vector = np.concatenate([np.asarray(part, dtype=np.float32).ravel() for part in parts])
    if vector.size != len(space):
        raise ValueError(
            "internal feature layout mismatch: built {0} numbers, expected {1}".format(
                vector.size, len(space)
            )
        )
    return np.clip(vector, 0.0, 1.0, out=vector)


def _soft_bins(
    values: np.ndarray, bins: int, *, circular: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each value between its two nearest bins, by distance.

    Hard binning is the quiet killer of histogram features. A value sitting a
    hair below a bin edge lands entirely in one bin; brighten the picture by one
    grey level and it lands entirely in the next. Two photographs of the same
    scene then differ by the whole bin, and the profile learns a spread that has
    nothing to do with the subject.

    Splitting each value linearly between the two bins whose centres straddle it
    - the same trick SIFT uses on its orientation bins - makes every histogram a
    continuous function of the pixels, so a one-grey-level shift moves the
    histogram by one grey level's worth.

    Args:
        values: the numbers to bin, already scaled so a bin is one unit wide.
        bins: how many bins.
        circular: whether the last bin wraps round to the first, which
            orientation does and brightness does not.

    Returns:
        The lower bin index, the upper bin index, and the weight going to the
        upper one.
    """
    position = values - 0.5
    lower = np.floor(position).astype(np.intp)
    upper_weight = (position - lower).astype(np.float64)
    upper = lower + 1
    if circular:
        lower = np.mod(lower, bins)
        upper = np.mod(upper, bins)
    else:
        # Outside the first and last centres there is nothing to share with, so
        # the end bins keep the whole weight.
        lower = np.clip(lower, 0, bins - 1)
        upper = np.clip(upper, 0, bins - 1)
    return lower, upper, upper_weight


def _soft_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    """Interpolated histogram of 0..1 values, normalised to sum to 1."""
    scaled = np.asarray(values, dtype=np.float64).ravel() * float(bins)
    lower, upper, weight = _soft_bins(scaled, bins, circular=False)
    totals = np.zeros(bins, dtype=np.float64)
    np.add.at(totals, lower, 1.0 - weight)
    np.add.at(totals, upper, weight)
    whole = float(totals.sum())
    if whole <= 1e-12:                      # pragma: no cover - an empty image
        return np.full(bins, 1.0 / float(bins), dtype=np.float32)
    return (totals / whole).astype(np.float32)


def _orientation_cells(
    angle: np.ndarray, magnitude: np.ndarray, grid: int, bins: int
) -> np.ndarray:
    """Magnitude-weighted orientation histogram per cell, each cell summing to 1.

    Directions are interpolated between neighbouring bins, and the bins wrap:
    179 degrees and 1 degree are nearly the same edge, and a histogram that did
    not know that would report a huge change every time an edge drifted past
    horizontal.

    A cell with no gradient at all - a patch of flat sky, a blank label - has no
    direction to report. Its histogram is left flat rather than zero, so it reads
    as "no preferred direction" instead of looking like every direction vanished.
    """
    scaled = (angle / np.pi) * float(bins)
    lower, upper, weight = _soft_bins(scaled.ravel(), bins, circular=True)
    weights = magnitude.ravel().astype(np.float64)
    shape = angle.shape
    per_cell = np.empty((grid * grid, bins), dtype=np.float64)
    for slot in range(bins):
        share = np.where(lower == slot, 1.0 - weight, 0.0)
        share += np.where(upper == slot, weight, 0.0)
        plane = (share * weights).reshape(shape).astype(np.float32)
        per_cell[:, slot] = _blocks(plane, grid).sum(axis=1)
    total = per_cell.sum(axis=1, keepdims=True)
    flat = total <= 1e-8
    per_cell = np.where(flat, 1.0 / float(bins), per_cell / np.maximum(total, 1e-8))
    return per_cell.astype(np.float32).ravel()


def _orientation_global(angle: np.ndarray, magnitude: np.ndarray, bins: int) -> np.ndarray:
    """Magnitude-weighted orientation histogram over the whole frame."""
    scaled = (angle / np.pi) * float(bins)
    lower, upper, weight = _soft_bins(scaled.ravel(), bins, circular=True)
    weights = magnitude.ravel().astype(np.float64)
    totals = np.zeros(bins, dtype=np.float64)
    np.add.at(totals, lower, weights * (1.0 - weight))
    np.add.at(totals, upper, weights * weight)
    whole = float(totals.sum())
    if whole <= 1e-8:
        return np.full(bins, 1.0 / float(bins), dtype=np.float32)
    return (totals / whole).astype(np.float32)


def extract_many(
    images: Sequence[LoadedImage], space: FeatureSpace
) -> np.ndarray:
    """Feature vectors for a sequence of loaded images, one row each."""
    if not images:
        return np.zeros((0, len(space)), dtype=np.float32)
    rows = [extract(image, space) for image in images]
    return np.vstack(rows).astype(np.float32)
