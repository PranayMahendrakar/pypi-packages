"""Every number the checks use, in one place, so all of them can be changed.

``Thresholds()`` is the default set. Pass your own to ``check``, ``check_stream``
or ``Monitor`` when your cameras are unusual - a low-light street camera and a
bright factory line do not want the same darkness limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict


@dataclass(frozen=True)
class Thresholds:
    """Limits and cut-offs for the camera health checks.

    Luminance values are on the 0-255 scale. Fractions are 0-1.
    """

    # --- working resolution -------------------------------------------------
    analysis_pixels: int = 409_600
    """Pixel budget for the decimated working image (640x640). 0 keeps full size."""

    # --- darkness -----------------------------------------------------------
    dark_mean: float = 35.0
    """Mean luminance at or below which the frame counts as dark."""
    dark_mean_critical: float = 12.0
    """Mean luminance at or below which darkness is critical."""
    dark_bright_pixel: float = 40.0
    """A frame whose 99th percentile stays under this has nothing usable in it."""

    # --- overexposure -------------------------------------------------------
    bright_mean: float = 225.0
    """Mean luminance at or above which the frame counts as overexposed."""
    bright_mean_critical: float = 245.0
    """Mean luminance at or above which overexposure is critical."""
    clipped_fraction: float = 0.15
    """Share of pixels at 250 or above that counts as blown highlights."""
    clipped_fraction_critical: float = 0.40
    """Share of blown pixels at which overexposure is critical."""

    # --- focus --------------------------------------------------------------
    focus_ratio: float = 0.085
    """Edge energy over scene contrast below which the lens reads as soft."""
    focus_ratio_critical: float = 0.035
    """Focus ratio below which defocus is critical."""
    min_content_contrast: float = 5.0
    """Scene contrast (std) under which focus cannot be judged at all."""

    # --- obstruction --------------------------------------------------------
    tile_grid: int = 16
    """The view is split into at most this many tiles per side.

    This is the resolution the blocked area is measured at, so a coarse grid
    under-reports it: a square covering half the view lands on whole tiles only
    in its middle, and its edge tiles still carry scene detail. 16 keeps each
    tile large enough for a stable detail estimate while measuring the area to
    within a few per cent."""
    dead_tile_ratio: float = 0.14
    """A tile with less than this share of the frame's typical detail is dead."""
    dead_tile_floor: float = 0.9
    """Absolute detail level (mean |Laplacian|) under which a tile is dead."""
    live_tile_detail: float = 2.0
    """The rest of the view must reach this much detail to call a blob a blockage."""
    obstruction_area: float = 0.10
    """Smallest share of the view a dead blob must cover to be reported.

    Measured on whole tiles, so it is a lower bound: the tiles straddling the
    edge of a blob still hold scene detail and are not counted."""
    obstruction_area_critical: float = 0.30
    """Share of the view at which an obstruction is critical."""
    obstruction_area_max: float = 0.93
    """Above this the whole view is dead, which is defocus or a cover, not a blob."""
    obstruction_noise_margin: float = 1.10
    """Tile detail within this multiple of the noise floor is grain, not scene detail."""

    # --- noise --------------------------------------------------------------
    noise_sigma: float = 6.0
    """Estimated sensor noise (grey levels) above which noise is reported."""
    noise_sigma_critical: float = 12.0
    """Noise level at which the fault is critical."""

    # --- colour cast --------------------------------------------------------
    colour_cast: float = 0.18
    """Relative channel imbalance above which a cast is reported."""
    colour_cast_critical: float = 0.35
    """Channel imbalance at which the cast is critical."""
    mono_channel_spread: float = 0.5
    """Below this the frame is effectively monochrome and the cast check is skipped."""

    # --- frozen feed --------------------------------------------------------
    freeze_mean_diff: float = 0.05
    """Mean absolute pixel change at or below which two frames are the same image."""
    freeze_changed_fraction: float = 0.01
    """Share of pixels that must move for a live sensor; below this the feed repeats."""
    static_mean_diff: float = 2.5
    """Mean change under this is a static scene, provided enough pixels did move."""
    static_changed_fraction: float = 0.20
    """Share of moving pixels that proves the sensor is alive on a still scene."""

    # --- tampering and drift ------------------------------------------------
    tamper_correlation: float = 0.55
    """Structural correlation with the reference below which the scene has changed."""
    tamper_correlation_critical: float = 0.25
    """Correlation below which tampering is critical."""
    drift_correlation: float = 0.85
    """Correlation below which a slow shift away from the reference counts as drift."""
    drift_delta: float = 0.10
    """How far correlation must fall across the history window to call it drift."""
    drift_min_frames: int = 6
    """Frames that must have been compared with the reference before drift is judged."""
    signature_size: int = 32
    """Side of the thumbnail the scene signature is built from."""
    min_signature_contrast: float = 2.0
    """A reference flatter than this carries no structure to compare against."""

    # --- scoring ------------------------------------------------------------
    penalty_critical: float = 45.0
    """Points a critical fault can take off the 0-100 score."""
    penalty_warning: float = 15.0
    """Points a warning can take off the score."""
    penalty_info: float = 3.0
    """Points an informational finding can take off the score."""

    def replace(self, **changes: Any) -> "Thresholds":
        """A copy with some values changed: ``thresholds.replace(dark_mean=20)``."""
        unknown = sorted(set(changes) - {f.name for f in fields(self)})
        if unknown:
            raise ValueError(
                "unknown threshold(s): {}; known names: {}".format(
                    ", ".join(unknown), ", ".join(f.name for f in fields(self))
                )
            )
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """Every threshold as a JSON-safe mapping."""
        return asdict(self)


DEFAULT_THRESHOLDS = Thresholds()


def resolve(thresholds: Any) -> Thresholds:
    """Accept a Thresholds, a mapping of overrides, or None for the defaults."""
    if thresholds is None:
        return DEFAULT_THRESHOLDS
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, dict):
        return DEFAULT_THRESHOLDS.replace(**thresholds)
    raise TypeError(
        "thresholds must be a Thresholds, a dict of overrides or None; got {}".format(
            type(thresholds).__name__
        )
    )
