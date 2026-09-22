"""The public functions: :func:`assess`, :func:`assess_batch`, :func:`is_blurry`.

Everything a caller needs in one line lives here. :class:`ImageAssessor` is the
same thing with the thresholds fixed once instead of passed every call.
"""
from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from ._loading import LoadedImage, label_for, load_image
from ._measures import MEASURE_NAMES, measure_all, measure_sharpness
from ._report import BatchReport, Metric, QualityReport, grade_for
from ._thresholds import DEFAULT_THRESHOLDS, Thresholds, ThresholdsLike, resolve_thresholds

logger = logging.getLogger(__name__)

#: Errors a single image can raise that a batch survives.
_IMAGE_ERRORS = (FileNotFoundError, ValueError, TypeError, OSError)

#: The measures a photo cannot fail and still be handed to a model. A frame that
#: is out of focus or has no exposure left to recover carries no information,
#: however tidy the rest of its numbers look, so failing either one caps the
#: overall score below ``thresholds.usable_score``. Everything else is traded off
#: in the weighted average.
CRITICAL_MEASURES = ("sharpness", "exposure")


def failed_gates(metrics: Dict[str, Metric]) -> List[str]:
    """Which of :data:`CRITICAL_MEASURES` this image failed, in report order."""
    return [
        name
        for name in CRITICAL_MEASURES
        if name in metrics and metrics[name].applies and not metrics[name].ok
    ]


def weighted_mean(metrics: Dict[str, Metric], thresholds: Thresholds) -> float:
    """Weighted average of the measures that applied, 0 to 100.

    Measures that could not be judged are dropped from both the top and the
    bottom of the average, so a photo is never punished for a question that
    could not be asked of it.
    """
    weights = thresholds.weights()
    total = 0.0
    weight_sum = 0.0
    for name, metric in metrics.items():
        if not metric.applies:
            continue
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        total += weight * float(metric.score)
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return max(0.0, min(100.0, total / weight_sum))


def overall_score(metrics: Dict[str, Metric], thresholds: Thresholds) -> float:
    """The headline 0-100 score: the weighted mean, with the gates applied.

    Without the gate a blurred frame with pleasant exposure, contrast and noise
    averages its way into the sixties and reads as a pass, which is exactly the
    photo this package exists to catch. So when a measure in
    :data:`CRITICAL_MEASURES` fails, the mean is scaled by
    ``usable_score / 100``: the ranking between two failing images is kept, and
    neither can present a number that contradicts ``usable``.
    """
    mean = weighted_mean(metrics, thresholds)
    if failed_gates(metrics):
        mean *= max(0.0, min(1.0, float(thresholds.usable_score) / 100.0))
    return max(0.0, min(100.0, mean))


def _issue(score: float, name: str, text: str) -> Tuple[float, int, str]:
    """Sort key plus text. Ties between equally bad measures break by measure
    order, so two zero scores always come out the same way round."""
    order = MEASURE_NAMES.index(name) if name in MEASURE_NAMES else len(MEASURE_NAMES)
    return (float(score), order, text)


def build_issues(metrics: Dict[str, Metric], thresholds: Thresholds) -> List[str]:
    """Plain-language problems, worst first.

    An issue is raised for every measure that did not clear
    ``thresholds.measure_ok_score``, plus one for clipping bad enough to destroy
    detail even when the overall exposure is fine.
    """
    found: List[Tuple[float, int, str]] = []

    sharpness = metrics.get("sharpness")
    if sharpness is not None and not sharpness.ok:
        if sharpness.details.get("measured_pixels") == 0:
            text = (
                "The image is too small to judge: at least 3x3 pixels are needed before any "
                "edge detail can be measured."
            )
        elif sharpness.value <= thresholds.sharpness_floor:
            # A blank frame has nothing in it to be in or out of focus, so
            # diagnosing a focus or motion problem would name a cause that
            # never happened. sharpness_floor is where the package already
            # documents "no usable edge detail anywhere in the frame".
            text = (
                "There is no detail in this frame at all; it is blank or featureless, so "
                "there is nothing to be in or out of focus (sharpness scored {0:.0f} of 100, "
                "Laplacian variance {1:.1f}, at or below the floor of {2:g})."
            ).format(sharpness.score, sharpness.value, thresholds.sharpness_floor)
        else:
            if sharpness.value < thresholds.sharpness_blurry * 0.25:
                verdict = "The photo is badly out of focus or motion blurred"
            else:
                verdict = "The photo is soft; edges are not crisp"
            text = (
                "{0} (sharpness scored {1:.0f} of 100, Laplacian variance {2:.1f} against a "
                "limit of {3:g})."
            ).format(
                verdict, sharpness.score, sharpness.value, thresholds.sharpness_blurry
            )
        found.append(_issue(sharpness.score, "sharpness", text))

    exposure = metrics.get("exposure")
    if exposure is not None:
        black = float(exposure.details.get("clipped_black", 0.0))
        white = float(exposure.details.get("clipped_white", 0.0))
        penalty = float(exposure.details.get("clipping_penalty", 0.0))
        mean = float(exposure.value)
        if not exposure.ok:
            if penalty >= 100.0 - exposure.score - penalty and penalty > 0.0:
                text = (
                    "Too much of the frame is clipped: {0:.1f}% is pure black and {1:.1f}% is "
                    "pure white, and no detail can be recovered from either"
                ).format(black * 100.0, white * 100.0)
            elif mean < thresholds.exposure_mean_low:
                text = (
                    "The photo is underexposed; the average pixel sits at {0:.2f} of 1.0 where "
                    "{1:g} is the bottom of normal"
                ).format(mean, thresholds.exposure_mean_low)
            else:
                text = (
                    "The photo is overexposed; the average pixel sits at {0:.2f} of 1.0 where "
                    "{1:g} is the top of normal"
                ).format(mean, thresholds.exposure_mean_high)
            found.append(
                _issue(
                    exposure.score,
                    "exposure",
                    "{0} (exposure scored {1:.0f} of 100).".format(text, exposure.score),
                )
            )
        elif black >= thresholds.clipping_bad or white >= thresholds.clipping_bad:
            found.append(
                _issue(
                    exposure.score + 0.5,
                    "exposure",
                    "Exposure is acceptable overall, but {0:.1f}% of the frame is crushed to "
                    "black and {1:.1f}% is blown out to white; detail in those areas is "
                    "gone.".format(black * 100.0, white * 100.0),
                )
            )

    contrast = metrics.get("contrast")
    if contrast is not None and not contrast.ok:
        found.append(
            _issue(
                contrast.score,
                "contrast",
                "The photo is flat; the middle 90% of pixels span only {0:.2f} of the "
                "luminance range against a limit of {1:g} (contrast scored {2:.0f} of "
                "100).".format(contrast.value, thresholds.contrast_spread_flat, contrast.score),
            )
        )

    noise = metrics.get("noise")
    if noise is not None and noise.applies and not noise.ok:
        found.append(
            _issue(
                noise.score,
                "noise",
                "The photo is grainy; the residual after a median filter is {0:.1f} of 255 "
                "against a limit of {1:g} (noise scored {2:.0f} of 100).".format(
                    noise.value, thresholds.noise_noticeable_255, noise.score
                ),
            )
        )

    framing = metrics.get("framing")
    if framing is not None and framing.applies and not framing.ok:
        fill = float(framing.details.get("subject_fill", 0.0))
        if fill < thresholds.framing_fill_low:
            text = (
                "The subject fills only {0:.1f}% of the frame; crop in or move closer"
            ).format(fill * 100.0)
        else:
            text = (
                "The subject is off centre, {0:.2f} of the way from the middle to a corner "
                "where {1:g} is the limit; it may be clipped at the edge"
            ).format(framing.value, thresholds.framing_offset_ok)
        found.append(
            _issue(
                framing.score,
                "framing",
                "{0} (framing scored {1:.0f} of 100).".format(text, framing.score),
            )
        )

    found.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in found]


def assess(image: Any, *, thresholds: ThresholdsLike = None) -> QualityReport:
    """Measure one image and explain the result.

    Args:
        image: a path (``str`` or ``os.PathLike``), a ``PIL.Image.Image``, or a
            numpy array shaped HxW, HxWx3 or HxWx4. Greyscale, RGB, RGBA and
            16-bit inputs all work, and EXIF orientation is applied first.
        thresholds: ``None`` for the defaults, a :class:`Thresholds`, or a dict
            of overrides such as ``{"sharpness_blurry": 40}``.

    Returns:
        A :class:`~image_quality_ai.QualityReport` carrying an overall score and
        grade, one :class:`~image_quality_ai.Metric` per measure, and the issues
        in plain language.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is corrupt or truncated, or the array is not an
            image. The message always names the file.
        TypeError: the argument is not a path, a PIL image or an array.

    Example:
        >>> import numpy as np, image_quality_ai
        >>> image_quality_ai.assess(np.zeros((8, 8), dtype=np.uint8)).usable
        False
    """
    active = resolve_thresholds(thresholds)
    loaded = load_image(image)
    metrics = measure_all(loaded, active)
    score = overall_score(metrics, active)
    gates = failed_gates(metrics)
    return QualityReport(
        score=score,
        grade=grade_for(score),
        usable=not gates and score >= active.usable_score,
        metrics=metrics,
        issues=build_issues(metrics, active),
        image=loaded.info(),
        thresholds=active,
        failed_gates=gates,
    )


def _is_single_image(images: Any) -> bool:
    return isinstance(images, (str, bytes, os.PathLike, np.ndarray, Image.Image, LoadedImage))


def assess_batch(
    images: Iterable[Any], *, workers: int = 1, thresholds: ThresholdsLike = None
) -> BatchReport:
    """Measure many images and rank them.

    Unlike :func:`assess`, a batch does not stop for one bad file: anything that
    cannot be read is recorded in ``report.failures`` with its path and the
    reason, and the rest of the batch is still assessed.

    Args:
        images: an iterable of paths, PIL images or numpy arrays. A single image
            is accepted too and treated as a batch of one.
        workers: threads to read and measure with. The default of 1 keeps
            everything on the calling thread. More helps when the images come
            from disk, because decoding releases the GIL.
        thresholds: as for :func:`assess`.

    Returns:
        A :class:`~image_quality_ai.BatchReport`.

    Raises:
        ValueError: if ``workers`` is below 1.
        TypeError: if ``images`` is not iterable.
    """
    if workers < 1:
        raise ValueError("workers must be 1 or more, got {0}".format(workers))
    active = resolve_thresholds(thresholds)

    if _is_single_image(images):
        items: List[Any] = [images]
    else:
        try:
            items = list(images)
        except TypeError:
            raise TypeError(
                "images must be an iterable of paths, PIL images or arrays, got "
                + type(images).__name__
            ) from None

    slots: List[Optional[QualityReport]] = [None] * len(items)
    failures: Dict[int, Dict[str, str]] = {}

    def run(index: int) -> None:
        item = items[index]
        try:
            slots[index] = assess(item, thresholds=active)
        except _IMAGE_ERRORS as exc:
            source = label_for(item)
            message = str(exc) or type(exc).__name__
            logger.debug("skipping %s: %s", source, message)
            failures[index] = {"source": source, "error": message}

    if workers == 1 or len(items) < 2:
        for index in range(len(items)):
            run(index)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            list(pool.map(run, range(len(items))))

    results = [report for report in slots if report is not None]
    ordered_failures = [failures[index] for index in sorted(failures)]
    return BatchReport(results=results, failures=ordered_failures)


def is_blurry(image: Any, threshold: Optional[float] = None) -> bool:
    """Is this photo too blurry to use?

    Args:
        image: as for :func:`assess`.
        threshold: Laplacian variance below which the answer is ``True``.
            Defaults to ``Thresholds.sharpness_blurry`` (100.0).

    Returns:
        ``True`` when the photo is blurrier than the threshold.

    Raises:
        ValueError: if ``threshold`` is not a finite number, or the image cannot
            be read.

    Example:
        >>> import numpy as np, image_quality_ai
        >>> image_quality_ai.is_blurry(np.full((64, 64), 128, dtype=np.uint8))
        True
    """
    if threshold is None:
        limit = float(DEFAULT_THRESHOLDS.sharpness_blurry)
    else:
        try:
            limit = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(
                "threshold must be a number, got {0!r}".format(threshold)
            ) from None
        if not math.isfinite(limit):
            raise ValueError("threshold must be finite, got {0!r}".format(threshold))
    loaded = load_image(image)
    metric = measure_sharpness(loaded, DEFAULT_THRESHOLDS)
    return float(metric.value) < limit


class ImageAssessor:
    """The same three calls with the thresholds set once.

    Args:
        thresholds: ``None`` for the defaults, a :class:`Thresholds`, or a dict
            of overrides.

    Example:
        >>> import numpy as np, image_quality_ai
        >>> strict = image_quality_ai.ImageAssessor({"usable_score": 80})
        >>> strict.assess(np.zeros((16, 16), dtype=np.uint8)).usable
        False
    """

    def __init__(self, thresholds: ThresholdsLike = None) -> None:
        self.thresholds: Thresholds = resolve_thresholds(thresholds)

    def assess(self, image: Any) -> QualityReport:
        """Measure one image. See :func:`assess`."""
        return assess(image, thresholds=self.thresholds)

    def assess_batch(self, images: Iterable[Any], *, workers: int = 1) -> BatchReport:
        """Measure many images. See :func:`assess_batch`."""
        return assess_batch(images, workers=workers, thresholds=self.thresholds)

    def is_blurry(self, image: Any, threshold: Optional[float] = None) -> bool:
        """Blur check using this object's ``sharpness_blurry``. See :func:`is_blurry`."""
        limit = self.thresholds.sharpness_blurry if threshold is None else threshold
        return is_blurry(image, limit)

    def measures(self) -> Tuple[str, ...]:
        """The measure names, in report order."""
        return MEASURE_NAMES

    def __repr__(self) -> str:                     # pragma: no cover - cosmetic
        changed = {
            name: getattr(self.thresholds, name)
            for name in Thresholds.field_names()
            if getattr(self.thresholds, name) != getattr(DEFAULT_THRESHOLDS, name)
        }
        return "ImageAssessor({0})".format(changed or "defaults")
