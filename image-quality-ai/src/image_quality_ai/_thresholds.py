"""Every number the judgement depends on, in one place.

Nothing in this package compares against a literal buried in a function. Each
measure asks :class:`Thresholds` for its boundaries, so a user who disagrees
with a default can override it and get a different verdict from the same pixels::

    import image_quality_ai
    report = image_quality_ai.assess(photo, thresholds={"sharpness_blurry": 40.0})

The defaults were chosen for ordinary photographs going into a vision model.
Scanned documents, microscopy, thermal frames and astrophotography all have
different notions of "normal", so expect to move them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple, Union

#: Long edge, in pixels, that spatial measures are computed at. Sharpness and
#: framing are resolution dependent: the same lens blur covers four times as many
#: pixels in a 4000px frame as in a 1000px one. Every image is resampled to this
#: long edge first - area-averaged coming down, pixel-replicated going up - so the
#: numbers mean the same thing for a phone photo, for a thumbnail of it and for a
#: 224px crop of it.
ANALYSIS_LONG_EDGE = 512

#: Shortest side, in pixels, an image must have before it is worth putting on the
#: analysis grid at all. A plane with fewer than three pixels on a side has no
#: interior, so there is nothing to measure and nothing to normalise: it is left
#: at its own size and the measures say they could not judge it.
ANALYSIS_MIN_EDGE = 3

#: Cells per side of the coarse detail grid the framing measure works on.
FRAMING_GRID = 16

#: Side, in native pixels, of each tile the noise measure samples. Grain is a
#: property of the pixels the camera produced, not of the scene, so it is the one
#: measure that must NOT be taken from the area-averaged analysis plane: shrinking
#: a 12-megapixel frame to 512px averages away exactly the grain being looked for.
NOISE_TILE = 128

#: Tiles per side of the grid the noise measure samples at native resolution.
#: 16 tiles of 128px cover a quarter of a million pixels wherever the frame came
#: from, which is plenty for a median, and costs the same on a phone snap as on a
#: 50-megapixel raw conversion.
NOISE_GRID = 4


@dataclass(frozen=True)
class Thresholds:
    """The boundaries every measure is judged against.

    Luminance values are on a 0.0 (black) to 1.0 (white) scale unless the field
    name ends in ``_255``, and every share or fraction runs 0.0 to 1.0.

    A score of 50 always sits exactly on the "borderline" constant for that
    measure, 100 on its "good" constant and 0 on its "bad" one, so a score can
    be read back to a number without reading the source.
    """

    # -- sharpness -----------------------------------------------------------
    #: Variance of the Laplacian, on a 0-255 luminance scale at the analysis
    #: size, below which a photo reads as blurry. This is the number
    #: :func:`image_quality_ai.is_blurry` compares against.
    sharpness_blurry: float = 100.0
    #: Laplacian variance at or above which a photo is called properly sharp.
    sharpness_good: float = 600.0
    #: Laplacian variance at or below which sharpness scores zero: no usable
    #: edge detail anywhere in the frame.
    sharpness_floor: float = 2.0

    # -- exposure ------------------------------------------------------------
    #: Mean luminance at or below which a frame is fully underexposed.
    exposure_mean_dark: float = 0.06
    #: Bottom of the band of mean luminance that scores full marks.
    exposure_mean_low: float = 0.30
    #: Top of the band of mean luminance that scores full marks.
    exposure_mean_high: float = 0.70
    #: Mean luminance at or above which a frame is fully overexposed.
    exposure_mean_bright: float = 0.94
    #: Luminance at or below which a pixel counts as crushed to black.
    black_level: float = 2.0 / 255.0
    #: Luminance at or above which a pixel counts as blown out to white.
    white_level: float = 253.0 / 255.0
    #: Share of clipped pixels that is normal and costs nothing. Real photos
    #: nearly always clip a few specular highlights.
    clipping_ok: float = 0.005
    #: Share of clipped pixels that costs the full clipping penalty.
    clipping_bad: float = 0.15
    #: Points taken off the exposure score when clipping is at its worst.
    clipping_penalty: float = 40.0

    # -- contrast ------------------------------------------------------------
    #: 5th-to-95th percentile luminance spread at or below which contrast is nil.
    contrast_spread_floor: float = 0.02
    #: Percentile spread on the borderline between flat and acceptable.
    contrast_spread_flat: float = 0.15
    #: Percentile spread at or above which contrast scores full marks.
    contrast_spread_good: float = 0.45
    #: Luminance standard deviation at or below which contrast scores zero.
    contrast_std_floor: float = 0.005
    #: Luminance standard deviation on the borderline between flat and acceptable.
    contrast_std_flat: float = 0.05
    #: Luminance standard deviation at or above which contrast scores full marks.
    contrast_std_good: float = 0.18
    #: Share of the contrast score that comes from the percentile spread; the
    #: rest comes from the standard deviation.
    contrast_spread_weight: float = 0.6

    # -- noise ---------------------------------------------------------------
    #: Residual sigma after a 3x3 median filter, on a 0-255 scale, at or below
    #: which the frame counts as clean.
    noise_clean_255: float = 1.0
    #: Residual sigma on the borderline: grain is visible, the frame is usable.
    noise_noticeable_255: float = 3.0
    #: Residual sigma at or above which noise scores zero.
    noise_bad_255: float = 10.0

    # -- framing -------------------------------------------------------------
    #: Distance of the detail centroid from the frame centre, as a share of the
    #: half-diagonal, that still scores full marks. Rule-of-thirds framing lands
    #: near 0.25, so the default is deliberately forgiving.
    framing_offset_ok: float = 0.28
    #: Centroid offset at or above which framing scores zero: the subject is
    #: jammed into a corner or half out of frame.
    framing_offset_bad: float = 0.70
    #: Share of the frame the detail region must fill before the subject stops
    #: counting as small and distant.
    framing_fill_low: float = 0.04
    #: Share of the frame at which fill scores full marks.
    framing_fill_good: float = 0.20
    #: Share of the strongest cell's detail that a cell must reach to join the
    #: high-detail region.
    framing_detail_fraction: float = 0.30
    #: Detail (mean absolute Laplacian, 0-255 scale) the strongest cell must
    #: reach before framing is judged at all. Below it there is no subject to
    #: locate, and the measure steps aside rather than inventing a verdict.
    framing_min_detail_255: float = 0.5
    #: Share of the framing score that comes from the centroid offset; the rest
    #: comes from how much of the frame the subject fills.
    framing_offset_weight: float = 0.65

    # -- verdict -------------------------------------------------------------
    #: Score at or above which one measure is reported as ``ok``.
    measure_ok_score: float = 60.0
    #: Overall score at or above which a photo is reported as ``usable``.
    usable_score: float = 60.0

    #: Weight of each measure in the overall score. They are normalised over the
    #: measures that actually applied, so a frame whose framing could not be
    #: judged is scored on the other four rather than punished for it.
    weight_sharpness: float = 0.30
    weight_exposure: float = 0.25
    weight_contrast: float = 0.20
    weight_noise: float = 0.10
    weight_framing: float = 0.15

    def weights(self) -> Dict[str, float]:
        """The per-measure weights, keyed by measure name."""
        return {
            "sharpness": float(self.weight_sharpness),
            "exposure": float(self.weight_exposure),
            "contrast": float(self.weight_contrast),
            "noise": float(self.weight_noise),
            "framing": float(self.weight_framing),
        }

    def replace(self, **overrides: float) -> "Thresholds":
        """A copy with some fields changed, e.g. ``t.replace(sharpness_blurry=40)``."""
        _check_names(overrides)
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, float]:
        """JSON-safe view of every threshold."""
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        """Every overridable name, in declaration order."""
        return tuple(item.name for item in fields(cls))


#: The defaults, ready to read or copy from.
DEFAULT_THRESHOLDS = Thresholds()

ThresholdsLike = Union[Thresholds, Mapping[str, float], None]


def _check_names(overrides: Mapping[str, Any]) -> None:
    known = set(Thresholds.field_names())
    unknown = [name for name in overrides if name not in known]
    if unknown:
        raise ValueError(
            "unknown threshold(s): "
            + ", ".join(sorted(unknown))
            + ". Known thresholds: "
            + ", ".join(Thresholds.field_names())
        )


def resolve_thresholds(thresholds: ThresholdsLike = None) -> Thresholds:
    """Accept ``None``, a :class:`Thresholds`, or a dict of overrides.

    Args:
        thresholds: ``None`` for the defaults, a :class:`Thresholds` instance, or
            a mapping of field name to value applied on top of the defaults.

    Returns:
        The :class:`Thresholds` to judge against.

    Raises:
        TypeError: if the argument is none of those.
        ValueError: if a mapping names a threshold that does not exist, or gives
            a value that is not a finite number.
    """
    if thresholds is None:
        return DEFAULT_THRESHOLDS
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        _check_names(thresholds)
        clean: Dict[str, float] = {}
        for name, value in thresholds.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "threshold {0!r} must be a number, got {1!r}".format(name, value)
                ) from None
            if not math.isfinite(number):
                raise ValueError(
                    "threshold {0!r} must be finite, got {1!r}".format(name, value)
                )
            clean[name] = number
        return replace(DEFAULT_THRESHOLDS, **clean)
    raise TypeError(
        "thresholds must be None, a Thresholds instance or a dict of overrides, got "
        + type(thresholds).__name__
    )


def describe_thresholds(thresholds: Optional[Thresholds] = None) -> str:
    """A printable table of the thresholds in force. Handy from a REPL."""
    active = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    width = max(len(name) for name in Thresholds.field_names())
    lines = ["threshold".ljust(width) + "  value"]
    for name in Thresholds.field_names():
        lines.append(name.ljust(width) + "  " + "{0:g}".format(getattr(active, name)))
    return "\n".join(lines)


#: Motion blur is recognised by TWO conditions together, because either alone is wrong.
#: A ratio test alone flags any anisotropic subject - a fence, a building facade, a page
#: of text all hold far more detail across one axis than the other while being perfectly
#: sharp. An absolute test alone flags any softly-lit photograph. Requiring both means
#: the weaker axis has to be nearly featureless AND far weaker than its partner, which is
#: what a smear does and what a striped subject does not. Measured on a structured test
#: scene: sharp holds 369 on its weaker axis, a 31-pixel smear holds 19.
DIRECTIONAL_BLUR_RATIO = 0.35
#: Edge energy, in squared grey levels, below which an axis counts as featureless.
DIRECTIONAL_DETAIL_FLOOR = 120.0
