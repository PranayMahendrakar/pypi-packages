"""The decision: which rectangle of an image is worth keeping.

One energy map from :mod:`smart_crop_ai._energy`, one summed-area table, one
exhaustive scan of every window of the wanted size. No randomness, no model, no
OpenCV: the same image and arguments always produce the same box.

The part worth reading twice is the confidence number. It is not a vague score:
it is the share of the chosen window's detail that a centre crop of the same
size would have thrown away. When it comes back at 0, cropping here bought the
caller nothing over cutting out the middle, and the notes say why.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ._energy import ENERGY_BUILDERS, integral_image
from ._images import apply_exif_orientation, open_image, resize_exact, working_arrays
from ._result import CropResult

logger = logging.getLogger(__name__)

#: Every strategy :func:`crop` accepts.
STRATEGIES: Tuple[str, ...] = ("auto", "saliency", "entropy", "edges", "center")

#: What ``strategy="auto"`` reaches for first.
AUTO_STRATEGY = "saliency"

DEFAULT_STRATEGY = "auto"
DEFAULT_PADDING = 0.05

#: Padding above this would leave the scored interior smaller than a tenth of
#: the window, which stops meaning "breathing room" and starts meaning "ignore
#: the crop I asked for".
MAX_PADDING = 0.45

#: Longest side the energy map is built at. Big enough to place a subject to
#: within a few source pixels, small enough that a 24 megapixel photo is still
#: a fraction of a second.
ANALYSIS_MAX_SIDE = 320

#: ``auto`` falls back to the centre when the energy map never rises above this.
FLAT_PEAK = 0.03

#: ...or when the best and worst windows differ by less than this share of the
#: best one, which means every placement is much the same placement.
FLAT_WINDOW_SPREAD = 0.01

#: Below this, window scores are equal as far as float64 is concerned.
_DEGENERATE = 1e-9

_RATIO_TYPES = (int, float, str, tuple, list)


def crop(
    image: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    ratio: Optional[Union[float, int, str, Sequence[float]]] = None,
    strategy: str = DEFAULT_STRATEGY,
    padding: float = DEFAULT_PADDING,
) -> CropResult:
    """Crop ``image`` to the interesting part of it.

    Args:
        image: a ``PIL.Image.Image``, a path to an image file, or a numpy array
            shaped ``(h, w)``, ``(h, w, 3)`` or ``(h, w, 4)``. Never modified.
        width: target width in pixels. On its own, the height follows the
            source's aspect ratio.
        height: target height in pixels. On its own, the width follows the
            source's aspect ratio.
        ratio: an aspect ratio instead of a size - ``(16, 9)``, ``1.0`` or
            ``"16:9"``. Takes the largest window of that shape that fits. Cannot
            be combined with ``width`` or ``height``.
        strategy: ``"auto"`` (saliency, falling back to the centre when the
            energy map is nearly flat), ``"saliency"`` (gradient magnitude plus
            colour variance), ``"entropy"`` (local entropy, better for texture),
            ``"edges"`` (Sobel density, better for product shots on plain
            backgrounds) or ``"center"`` (a plain centre crop, for comparison).
        padding: 0 to 0.45. Breathing room: this fraction of the window's width
            and height at each edge is left out of the score, so the subject is
            framed inside the crop rather than pressed against its border.

    Returns:
        A :class:`~smart_crop_ai.CropResult` carrying the cropped image, the box,
        the strategy that actually chose it, a 0 to 1 confidence and notes.

    Raises:
        ValueError: if no size and no ratio is given, if ``ratio`` is combined
            with ``width`` or ``height``, or if a size, ratio, strategy or
            padding is out of range.
        TypeError: if ``image`` or ``ratio`` is of a type this cannot read.
        FileNotFoundError: if ``image`` is a path that does not exist.

    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> flat = Image.fromarray(np.full((40, 80, 3), 128, dtype=np.uint8))
        >>> result = crop(flat, ratio=1.0)
        >>> result.strategy_used, result.confidence
        ('center', 0.0)
    """
    strategy = _check_strategy(strategy)
    padding = _check_padding(padding)

    source = open_image(image)
    oriented, was_rotated = apply_exif_orientation(source)
    source_width, source_height = oriented.size
    if source_width < 1 or source_height < 1:
        raise ValueError("image has no pixels")

    notes: List[str] = []
    if was_rotated:
        notes.append(
            "EXIF orientation was applied before anything was measured, so the box "
            "refers to the upright image, not to the stored pixel order."
        )

    target_width, target_height = _target_size(
        source_width, source_height, width, height, ratio
    )

    common: Dict[str, Any] = {
        "strategy_requested": strategy,
        "source_size": (source_width, source_height),
        "target_size": (target_width, target_height),
        "padding": padding,
        "source": _source_label(image),
    }

    if target_width > source_width or target_height > source_height:
        notes.append(
            "The {0} x {1} crop you asked for is larger than the {2} x {3} source, so the "
            "whole image was returned at its own size. Nothing was upscaled; resize it "
            "yourself if you want those pixels.".format(
                target_width, target_height, source_width, source_height
            )
        )
        return _build(
            oriented,
            (0, 0, source_width, source_height),
            strategy_used="whole_image",
            confidence=0.0,
            scores={},
            notes=notes,
            **common,
        )

    centre_box = _centre_box(source_width, source_height, target_width, target_height)

    if strategy == "center":
        notes.append(
            "Strategy 'center' takes the middle of the image without looking at it, so "
            "the confidence is 0 by definition. Use 'auto' to let the content decide."
        )
        return _build(
            oriented,
            centre_box,
            strategy_used="center",
            confidence=0.0,
            scores={},
            notes=notes,
            **common,
        )

    map_name = AUTO_STRATEGY if strategy == "auto" else strategy
    rgb, alpha, scale = working_arrays(oriented, ANALYSIS_MAX_SIDE)
    energy = ENERGY_BUILDERS[map_name](rgb)
    if alpha is not None:
        # Fully transparent pixels hold nothing worth framing.
        energy = energy * alpha
        notes.append(
            "The alpha channel was used as a mask, so fully transparent pixels count "
            "as empty rather than as detail."
        )

    search = _search(
        energy,
        max(1, int(round(target_width * scale))),
        max(1, int(round(target_height * scale))),
        padding,
    )
    scores = {
        "best_window": search["best_mean"],
        "center_window": search["centre_mean"],
        "energy_peak": float(energy.max()),
        "energy_spread": float(energy.std()),
        "window_spread": search["window_spread"],
    }

    degenerate = (
        search["best_mean"] <= _DEGENERATE or search["window_spread"] <= _DEGENERATE
    )
    nearly_flat = (
        scores["energy_peak"] < FLAT_PEAK
        or search["window_spread"] < FLAT_WINDOW_SPREAD
    )

    if degenerate or (strategy == "auto" and nearly_flat):
        notes.append(
            _fallback_note(
                map_name,
                energy_peak=scores["energy_peak"],
                exact=degenerate,
                covers=(target_width * target_height)
                / float(source_width * source_height),
            )
        )
        return _build(
            oriented,
            centre_box,
            strategy_used="center",
            confidence=0.0,
            scores=scores,
            notes=notes,
            **common,
        )

    box = _to_source_box(
        search["best_x"],
        search["best_y"],
        scale,
        source_width,
        source_height,
        target_width,
        target_height,
    )
    confidence = _confidence(search["best_mean"], search["centre_mean"])
    if confidence <= 0.0:
        notes.append(
            "The best window scores no better than the centre one, so this crop is "
            "worth no more than a plain centre crop of the same size."
        )
    return _build(
        oriented,
        box,
        strategy_used=map_name,
        confidence=confidence,
        scores=scores,
        notes=notes,
        **common,
    )


def crop_to_file(src: Any, dst: str, **kw: Any) -> CropResult:
    """Crop ``src`` and write the result to ``dst``. Returns the CropResult.

    Args:
        src: anything :func:`crop` accepts, usually a path.
        dst: where to write the crop. The format comes from the suffix, missing
            parent directories are created, and transparency is composited onto
            white when the format cannot store it.
        **kw: passed straight to :func:`crop`.

    Example:
        >>> import os, tempfile, numpy as np
        >>> from PIL import Image
        >>> folder = tempfile.mkdtemp()
        >>> source = os.path.join(folder, "in.png")
        >>> Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(source)
        >>> crop_to_file(source, os.path.join(folder, "out.png"), ratio=1.0).size
        (40, 40)
    """
    result = crop(src, **kw)
    if isinstance(src, (str, os.PathLike)):
        result.source = os.fspath(src)
    result.save(dst)
    return result


def thumbnail(image: Any, size: Union[int, Sequence[int]], **kw: Any) -> Image.Image:
    """Crop to the subject, then resize to exactly ``size``.

    Unlike ``PIL.Image.thumbnail``, which shrinks the whole frame and lets the
    subject fall wherever it falls, this picks the largest window of the wanted
    shape that actually holds the subject, then scales that window down.

    Args:
        image: anything :func:`crop` accepts.
        size: ``(width, height)``, or one integer for a square.
        **kw: passed to :func:`crop`. Give ``width``, ``height`` or ``ratio``
            yourself to override the default, which is to crop at the aspect
            ratio of ``size``.

    Returns:
        A new ``PIL.Image.Image`` of exactly ``size``.

    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> photo = Image.fromarray(np.zeros((200, 400, 3), dtype=np.uint8))
        >>> thumbnail(photo, 64).size
        (64, 64)
    """
    thumb_width, thumb_height = _thumbnail_size(size)
    options = dict(kw)
    if not any(key in options for key in ("ratio", "width", "height")):
        options["ratio"] = (thumb_width, thumb_height)
    result = crop(image, **options)
    return resize_exact(result.image, (thumb_width, thumb_height))


# ---------------------------------------------------------------- the search


def _window_sums(energy: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """Sum of ``energy`` over every ``win_w`` x ``win_h`` window, via one SAT."""
    table = integral_image(energy)
    return (
        table[win_h:, win_w:]
        - table[:-win_h, win_w:]
        - table[win_h:, :-win_w]
        + table[:-win_h, :-win_w]
    )


def _search(
    energy: np.ndarray, target_w: int, target_h: int, padding: float
) -> Dict[str, float]:
    """Best window of ``target_w`` x ``target_h``, in working-map pixels.

    Padding is applied by scoring only the interior of each candidate window, so
    a subject hugging the window's border loses to the same subject framed with
    room around it. Confidence is always measured on the *whole* window, so the
    number the caller sees compares like with like.
    """
    map_h, map_w = energy.shape
    win_w = int(min(max(1, target_w), map_w))
    win_h = int(min(max(1, target_h), map_h))

    inner_w = max(1, win_w - 2 * int(round(win_w * padding)))
    inner_h = max(1, win_h - 2 * int(round(win_h * padding)))
    pad_x = (win_w - inner_w) // 2
    pad_y = (win_h - inner_h) // 2

    last_x = map_w - win_w
    last_y = map_h - win_h

    inner = _window_sums(energy, inner_w, inner_h)
    scored = inner[pad_y : pad_y + last_y + 1, pad_x : pad_x + last_x + 1]
    best_index = int(np.argmax(scored))
    best_y, best_x = divmod(best_index, scored.shape[1])

    whole = _window_sums(energy, win_w, win_h)
    area = float(win_w * win_h)
    best_mean = float(whole[best_y, best_x]) / area
    centre_mean = float(whole[last_y // 2, last_x // 2]) / area

    highest = float(whole.max())
    lowest = float(whole.min())
    window_spread = (highest - lowest) / highest if highest > 0.0 else 0.0

    return {
        "best_x": float(best_x),
        "best_y": float(best_y),
        "best_mean": best_mean,
        "centre_mean": centre_mean,
        "window_spread": float(window_spread),
    }


def _fallback_note(
    map_name: str,
    *,
    energy_peak: float,
    exact: bool,
    covers: float,
) -> str:
    """Say why the centre was used, naming the cause instead of guessing at it.

    Two very different situations end up here and they must not be described
    the same way. Either the image really holds nothing - a blank wall, a sky -
    or it holds plenty but the window asked for is so large that every place it
    could sit contains the same detail. Telling a user their photo of an
    obvious subject has a "flat energy map" is how a good tool loses trust.
    """
    if energy_peak < FLAT_PEAK:
        return (
            "The {0} map is {1} (peak {2:.3f}): there is no detail anywhere in this "
            "image to crop to, so the plain centre was used rather than maximising "
            "noise and calling it a subject.".format(
                map_name, "flat" if exact else "nearly flat", energy_peak
            )
        )
    return (
        "This image has detail (peak {0:.3f}), but the crop covers {1:.0%} of it, so "
        "every window of this size holds {2} the same detail and there is no better "
        "placement to find. The plain centre was used. Ask for a smaller crop if you "
        "want the subject framed.".format(
            energy_peak, covers, "exactly" if exact else "nearly"
        )
    )


def _confidence(best_mean: float, centre_mean: float) -> float:
    """Share of the chosen window's detail a centre crop would have missed."""
    if best_mean <= 0.0:
        return 0.0
    return float(np.clip((best_mean - centre_mean) / best_mean, 0.0, 1.0))


def _to_source_box(
    best_x: float,
    best_y: float,
    scale: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[int, int, int, int]:
    """Map a working-map top-left back to a source box, clamped inside bounds."""
    divisor = scale if scale > 0.0 else 1.0
    left = int(round(best_x / divisor))
    top = int(round(best_y / divisor))
    left = max(0, min(left, source_width - target_width))
    top = max(0, min(top, source_height - target_height))
    return (left, top, left + target_width, top + target_height)


def _centre_box(
    source_width: int, source_height: int, target_width: int, target_height: int
) -> Tuple[int, int, int, int]:
    """The plain centre crop of the wanted size."""
    left = (source_width - target_width) // 2
    top = (source_height - target_height) // 2
    return (left, top, left + target_width, top + target_height)


def _build(
    oriented: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    strategy_used: str,
    confidence: float,
    scores: Dict[str, float],
    notes: List[str],
    source_size: Tuple[int, int],
    **rest: Any,
) -> CropResult:
    """Cut the box out and wrap everything in a :class:`CropResult`.

    The box is clamped to the image one last time here, so that no arithmetic
    above can ever hand the caller a rectangle that falls outside the source.
    """
    width, height = source_size
    left, top, right, bottom = (int(value) for value in box)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    safe = (left, top, right, bottom)
    return CropResult(
        image=oriented.crop(safe),
        box=safe,
        strategy_used=strategy_used,
        confidence=float(confidence),
        scores=dict(scores),
        notes=list(notes),
        source_size=source_size,
        **rest,
    )


# ------------------------------------------------------------- the arguments


def _check_strategy(strategy: Any) -> str:
    """Validate ``strategy``, naming the alternatives when it is wrong."""
    if not isinstance(strategy, str):
        raise TypeError(
            "strategy must be a string, not {0}".format(type(strategy).__name__)
        )
    name = strategy.strip().lower()
    if name in ("centre", "middle"):
        name = "center"
    if name not in STRATEGIES:
        raise ValueError(
            "unknown strategy {0!r}; choose one of: {1}".format(
                strategy, ", ".join(STRATEGIES)
            )
        )
    return name


def _check_padding(padding: Any) -> float:
    """Validate ``padding`` as a fraction from 0 to :data:`MAX_PADDING`."""
    try:
        value = float(padding)
    except (TypeError, ValueError):
        raise ValueError(
            "padding must be a number from 0 to {0}, not {1!r}".format(
                MAX_PADDING, padding
            )
        ) from None
    if not np.isfinite(value) or value < 0.0 or value > MAX_PADDING:
        raise ValueError(
            "padding must be a fraction from 0 to {0}, got {1!r}".format(
                MAX_PADDING, padding
            )
        )
    return value


def _check_length(name: str, value: Any) -> int:
    """Validate a pixel length as a positive whole number."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "{0} must be a whole number of pixels, not {1}".format(
                name, type(value).__name__
            )
        )
    number = float(value)
    if not np.isfinite(number) or number < 1:
        raise ValueError("{0} must be at least 1 pixel, got {1!r}".format(name, value))
    return int(round(number))


def _parse_ratio(ratio: Any) -> float:
    """Turn ``(16, 9)``, ``1.5`` or ``"16:9"`` into a width-over-height float."""
    if isinstance(ratio, bool) or not isinstance(ratio, _RATIO_TYPES + (np.number,)):
        raise TypeError(
            "ratio must be a number, a (width, height) pair or a string like "
            "'16:9', not {0}".format(type(ratio).__name__)
        )
    if isinstance(ratio, str):
        text = ratio.strip().replace("/", ":").replace("x", ":")
        parts = text.split(":") if ":" in text else [text, "1"]
        if len(parts) != 2:
            raise ValueError(
                "ratio string must look like '16:9' or '1.5', got {0!r}".format(ratio)
            )
        try:
            numerator, denominator = float(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError(
                "ratio string must look like '16:9' or '1.5', got {0!r}".format(ratio)
            ) from None
    elif isinstance(ratio, (tuple, list)):
        if len(ratio) != 2:
            raise ValueError(
                "ratio pair must be (width, height), got {0!r}".format(ratio)
            )
        try:
            numerator, denominator = float(ratio[0]), float(ratio[1])
        except (TypeError, ValueError):
            raise ValueError(
                "ratio pair must hold two numbers, got {0!r}".format(ratio)
            ) from None
    else:
        numerator, denominator = float(ratio), 1.0

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("ratio must be finite, got {0!r}".format(ratio))
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("ratio must be positive, got {0!r}".format(ratio))
    return numerator / denominator


def _target_size(
    source_width: int,
    source_height: int,
    width: Optional[int],
    height: Optional[int],
    ratio: Any,
) -> Tuple[int, int]:
    """Work out the wanted ``(width, height)`` in source pixels.

    Raises:
        ValueError: if nothing was asked for, or if a ratio was asked for at the
            same time as an explicit width or height.
    """
    if ratio is not None:
        given = [
            "{0}={1!r}".format(name, value)
            for name, value in (("width", width), ("height", height))
            if value is not None
        ]
        if given:
            raise ValueError(
                "give a size or a ratio, not both: ratio={0!r} was passed together with "
                "{1}. Use crop(image, width, height) for an exact size, or "
                "crop(image, ratio={0!r}) to keep an aspect ratio and let the size "
                "follow.".format(ratio, " and ".join(given))
            )
        aspect = _parse_ratio(ratio)
        if source_width / float(source_height) > aspect:
            target_height = source_height
            target_width = int(round(source_height * aspect))
        else:
            target_width = source_width
            target_height = int(round(source_width / aspect))
        target_width = max(1, min(target_width, source_width))
        target_height = max(1, min(target_height, source_height))
        return target_width, target_height

    if width is None and height is None:
        raise ValueError(
            "nothing to crop to: give width and height, or a ratio such as "
            "ratio=(16, 9) or ratio=1.0"
        )
    if width is not None and height is not None:
        return _check_length("width", width), _check_length("height", height)
    if width is not None:
        target_width = _check_length("width", width)
        return target_width, max(
            1, int(round(target_width * source_height / float(source_width)))
        )
    target_height = _check_length("height", height)
    return (
        max(1, int(round(target_height * source_width / float(source_height)))),
        target_height,
    )


def _thumbnail_size(size: Any) -> Tuple[int, int]:
    """``64`` or ``(64, 48)`` into a ``(width, height)`` pair."""
    if isinstance(size, bool):
        raise TypeError("size must be an integer or a (width, height) pair, not bool")
    if isinstance(size, (int, float, np.number)):
        side = _check_length("size", size)
        return side, side
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError(
                "size must be (width, height) or one integer, got {0!r}".format(size)
            )
        return _check_length("width", size[0]), _check_length("height", size[1])
    raise TypeError(
        "size must be an integer or a (width, height) pair, not {0}".format(
            type(size).__name__
        )
    )


def _source_label(image: Any) -> str:
    """What to call the input in the report."""
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    if isinstance(image, Image.Image):
        name = getattr(image, "filename", "") or ""
        if name:
            return str(name)
    return "<image>"
