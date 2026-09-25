"""The pipeline: :func:`clean`, :func:`clean_file` and :func:`estimate_skew`.

This module owns the decisions. Every step asks one question before it runs -
is there anything here to fix? - and writes the answer into the result either
way, so a page that comes back looking much as it went in says which steps
stood down and why.

Two pages never go through the pipeline at all. A blank sheet has nothing to
threshold but its own grain, and thresholding it produces a field of speckles
that looks, to an OCR engine, exactly like text. A photograph has no paper
level, no lines and no skew, and binarising one destroys it. Both are
recognised first, reported as what they are, and returned as they came.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from . import _analysis, _images, _steps
from ._result import CleanResult, Step

logger = logging.getLogger(__name__)

#: Longest side the page is measured on. Skew, line height and the blank and
#: photograph verdicts all come off a plane this size; every step is then
#: applied at full resolution.
ANALYSIS_MAX_SIDE = 1200

#: The thresholding modes :func:`clean` accepts.
THRESHOLD_MODES = _steps.THRESHOLD_MODES
#: Default thresholding mode. Local, because a page lit unevenly is the normal
#: case and one global cut cannot serve both ends of it.
DEFAULT_THRESHOLD = "adaptive"


def _check_threshold(mode: Any) -> str:
    """Validate the ``threshold`` argument and return it."""
    if mode is None:
        return "none"
    text = str(mode).strip().lower()
    if text not in THRESHOLD_MODES:
        raise ValueError(
            "threshold must be one of {0}, not {1!r}".format(
                ", ".join(repr(name) for name in THRESHOLD_MODES), mode
            )
        )
    return text


def _check_dpi(value: Any, name: str) -> Optional[float]:
    """Validate a dpi argument and return it as a float, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{0} must be a number, not {1!r}".format(name, value)) from None
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("{0} must be a positive number, got {1!r}".format(name, value))
    return number


def _prepare(image: Any) -> Tuple[Image.Image, Step]:
    """Open ``image`` and reduce it to luminance. Returns the grayscale step too."""
    opened = _images.open_image(image)
    turned, was_turned = _images.apply_exif_orientation(opened)
    grey, how = _images.to_grayscale(turned)
    if grey.size[0] < 1 or grey.size[1] < 1:
        raise ValueError("image has no pixels")
    detail = how
    if was_turned:
        detail = "turned upright from its EXIF orientation tag, then " + how
    applied = was_turned or turned.mode != "L"
    return grey, Step("grayscale", applied, detail)


def _source_name(image: Any) -> str:
    """A readable name for whatever was passed in."""
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    if isinstance(image, np.ndarray):
        return "<array {0}x{1}>".format(image.shape[1], image.shape[0])
    return "<image>"


def clean(
    image: Any,
    *,
    deskew: bool = True,
    denoise: bool = True,
    threshold: str = DEFAULT_THRESHOLD,
    border: bool = True,
    upscale_to_dpi: Optional[float] = None,
    dpi: Optional[float] = None,
) -> CleanResult:
    """Prepare a scanned page so an OCR engine reads it better.

    The page is measured once, then taken through six steps in order -
    grayscale, deskew, border, denoise, threshold, upscale - each of which
    reports itself in :attr:`CleanResult.steps` whether it did anything or not.

    Args:
        image: a path, a ``PIL.Image.Image``, or a numpy array shaped
            ``(h, w)``, ``(h, w, 1)``, ``(h, w, 3)`` or ``(h, w, 4)``. Greyscale
            and colour are both fine, and your image is never modified.
        deskew: estimate the angle of the text and rotate the page back to
            horizontal, filling the new corners with the page's paper colour.
        denoise: apply a median filter sized to the height of the text, but only
            if the paper is actually grainy or specked.
        threshold: ``"adaptive"`` for a local mean with a window taken from the
            text height, ``"otsu"`` for one global cut, ``"none"`` to leave the
            page in greyscale.
        border: trim the black bands a scanner leaves down the sides of a page
            smaller than its glass.
        upscale_to_dpi: enlarge the page to this resolution. Needs ``dpi`` as
            well, because nothing can be scaled to 300 dpi without knowing what
            it is now.
        dpi: the resolution of the input, if you know it.

    Returns:
        A :class:`CleanResult`. ``.image`` is the cleaned page, ``.steps`` is
        what happened to it, and ``.summary()`` says so in words.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``image`` is none of the accepted kinds.
        ValueError: on an unknown ``threshold`` mode, a dpi that is not a
            positive number, or an image with no pixels.
    """
    mode = _check_threshold(threshold)
    source_dpi = _check_dpi(dpi, "dpi")
    target_dpi = _check_dpi(upscale_to_dpi, "upscale_to_dpi")

    current, grayscale_step = _prepare(image)
    source_size = current.size
    plane = _images.to_array(current)
    small, scale = _images.analysis_plane(plane, ANALYSIS_MAX_SIDE)
    stats = _analysis.measure(small, scale)

    result = CleanResult(
        image=current,
        steps=[grayscale_step],
        estimated_text_height_px=stats.text_height,
        page_kind=stats.kind,
        page_kind_detail=stats.why,
        estimated_skew_degrees=stats.skew_degrees,
        source_size=source_size,
        output_size=source_size,
        dpi=source_dpi,
        output_dpi=source_dpi,
        threshold_mode=mode,
        source=_source_name(image),
    )

    if not stats.is_document:
        _stand_down(result, stats.kind)
        result.output_size = result.image.size
        return result

    scanned_size, scanned_plane = current.size, plane
    current, plane = _run_deskew(result, current, plane, stats, deskew)
    current, plane = _run_border(
        result, current, plane, scanned_plane, scanned_size, border
    )
    current, plane = _run_denoise(result, current, plane, stats, denoise)
    current, _ = _run_threshold(result, current, plane, stats, mode)
    current = _run_upscale(result, current, source_dpi, target_dpi)

    result.image = current
    result.output_size = current.size
    return result


def _stand_down(result: CleanResult, kind: str) -> None:
    """Record every step as skipped on a blank page or a photograph.

    The verdict outranks the arguments: ``threshold="otsu"`` on a blank sheet is
    still a request to turn grain into speckles, so the kind decides and the
    note says what to check.
    """
    if kind == "blank":
        reason = (
            "the page is blank, so there is nothing to {0}; thresholding a "
            "blank sheet turns its grain into speckles an OCR engine reads as "
            "text, which is worse than leaving it alone"
        )
        note = (
            "This page is blank. It comes back exactly as it went in, in "
            "greyscale. Check result.is_blank before sending it to an OCR "
            "engine - there is nothing on it to read."
        )
    else:
        reason = (
            "this is a photograph rather than a document page, so there is "
            "nothing to {0}; a photograph has no paper level, no lines of text "
            "and no skew, and binarising one destroys it"
        )
        note = (
            "This does not look like a document page, so it comes back "
            "unchanged in greyscale. Check result.is_photograph. If it really "
            "is a page of text, it is an unusual one: pass threshold='otsu' "
            "and handle the steps yourself."
        )
    for name, wording in (
        ("deskew", "straighten"),
        ("border", "trim"),
        ("denoise", "filter"),
        ("threshold", "threshold"),
        ("upscale", "enlarge"),
    ):
        result.steps.append(Step(name, False, reason.format(wording)))
    result.skew_corrected_degrees = 0.0
    result.notes.append(note)


def _run_deskew(
    result: CleanResult,
    current: Image.Image,
    plane: np.ndarray,
    stats: _analysis.PageStats,
    wanted: bool,
) -> Tuple[Image.Image, np.ndarray]:
    """The deskew step, whether or not it ends up rotating anything."""
    if not wanted:
        result.steps.append(
            Step("deskew", False, "turned off by deskew=False; the page measures "
                 "{0:+.2f} degrees off horizontal".format(stats.skew_degrees))
        )
        return current, plane
    if abs(stats.skew_degrees) < _steps.MIN_SKEW_TO_CORRECT:
        result.steps.append(
            Step("deskew", False, "the page is already straight at {0:+.2f} degrees, "
                 "under the {1:g} degree floor worth an interpolation pass".format(
                     stats.skew_degrees, _steps.MIN_SKEW_TO_CORRECT))
        )
        return current, plane
    turned, detail = _steps.deskew(current, stats.skew_degrees, stats.paper_level)
    result.steps.append(Step("deskew", True, detail))
    result.skew_corrected_degrees = float(stats.skew_degrees)
    return turned, _images.to_array(turned)


def _run_border(
    result: CleanResult,
    current: Image.Image,
    plane: np.ndarray,
    scanned_plane: np.ndarray,
    scanned_size: Tuple[int, int],
    wanted: bool,
) -> Tuple[Image.Image, np.ndarray]:
    """The border step: cut the scanner's black bands off the sides.

    A black band is square to the image when the scanner's lid left it there and
    square to the page when the page itself carries it, and the deskew that has
    just run turns each of those into the other. So the band is looked for twice
    - on ``scanned_plane``, the page as it arrived, and on the page as it is now
    - and whatever either reading found comes off.
    """
    if not wanted:
        result.steps.append(Step("border", False, "turned off by border=False"))
        return current, plane
    turned = result.skew_corrected_degrees
    box = _steps.find_content_box(scanned_plane, _images.paper_level(scanned_plane))
    if turned:
        if box is not None:
            box = _steps.widen_for_rotation(box, scanned_size, current.size, turned)
        sine = abs(float(np.sin(np.radians(turned))))
        gap = (
            int(np.ceil(scanned_size[0] * sine)) + _steps.BORDER_FEATHER,
            int(np.ceil(scanned_size[1] * sine)) + _steps.BORDER_FEATHER,
        )
        box = _steps.union_boxes(
            box,
            _steps.find_content_box(plane, _images.paper_level(plane), gap),
            current.size,
        )
    if box is None:
        result.steps.append(
            Step("border", False, "no scanner edge or black margin found; every side "
                 "of the page is already paper")
        )
        return current, plane
    cropped = current.crop(box)
    detail = _steps.describe_border(box, current.size)
    if turned:
        detail += (
            ", measured before the {0:+.2f} degree turn and widened by how far "
            "that turn pushed the band".format(-turned)
        )
    result.steps.append(Step("border", True, detail))
    return cropped, _images.to_array(cropped)


def _run_denoise(
    result: CleanResult,
    current: Image.Image,
    plane: np.ndarray,
    stats: _analysis.PageStats,
    wanted: bool,
) -> Tuple[Image.Image, np.ndarray]:
    """The denoise step: a median filter, but only on a page that needs one."""
    if not wanted:
        result.steps.append(Step("denoise", False, "turned off by denoise=False"))
        return current, plane
    paper = _images.paper_level(plane)
    levels, specks = _steps.estimate_noise(plane, paper)
    size = _steps.median_window_for(stats.text_height)
    measured = "the paper measures {0:.1f} grey levels of grain and {1:.2f}% specks".format(
        levels, specks * 100.0
    )
    if not _steps.is_noisy(levels, specks):
        result.steps.append(
            Step("denoise", False, "the page is already clean: {0}, under the "
                 "{1:g} level floor - filtering it would only soften the "
                 "text".format(measured, _steps.NOISE_FLOOR))
        )
        return current, plane
    filtered = _steps.median_filter(current, size)
    height = stats.text_height
    sized = (
        "no text height was measured, so the smallest window was used"
        if height is None
        else "sized for text about {0:.0f} px tall".format(height)
    )
    result.steps.append(
        Step("denoise", True, "median filter {0} x {0}, {1}; {2}".format(size, sized, measured))
    )
    return filtered, _images.to_array(filtered)


def _run_threshold(
    result: CleanResult,
    current: Image.Image,
    plane: np.ndarray,
    stats: _analysis.PageStats,
    mode: str,
) -> Tuple[Image.Image, np.ndarray]:
    """The threshold step: adaptive, otsu, or left in greyscale."""
    if mode == "none":
        result.steps.append(
            Step("threshold", False, "turned off by threshold='none'; the page comes "
                 "back in greyscale")
        )
        return current, plane
    paper = _images.paper_level(plane)
    if mode == "otsu":
        level = _steps.otsu_level(plane)
        binary = _steps.global_threshold(plane, level)
        result.threshold_level = level
        detail = (
            "one global cut at grey level {0}, chosen by Otsu's method; right "
            "for a page lit evenly end to end, and only for that".format(level)
        )
    else:
        window = _steps.adaptive_window_for(stats.text_height)
        contrast = _steps.contrast_of(plane, paper)
        offset = max(_steps.ADAPTIVE_MIN_OFFSET, _steps.ADAPTIVE_OFFSET_SHARE * contrast)
        binary = _steps.adaptive_threshold(plane, window, offset)
        sized = (
            "the default, no text height to size one from"
            if stats.text_height is None
            else "3x the {0:.0f} px text height".format(stats.text_height)
        )
        detail = (
            "local mean over a {0} x {0} window ({1}), ink is {2:.0f} levels "
            "below its surroundings".format(window, sized, offset)
        )
    image = _images.from_array(binary)
    result.steps.append(Step("threshold", True, detail))
    return image, binary


def _run_upscale(
    result: CleanResult,
    current: Image.Image,
    source_dpi: Optional[float],
    target_dpi: Optional[float],
) -> Image.Image:
    """The upscale step, which needs both resolutions or it will not guess."""
    if target_dpi is None:
        result.steps.append(
            Step("upscale", False, "no upscale_to_dpi was asked for, so the page "
                 "keeps its own resolution")
        )
        return current
    if source_dpi is None:
        result.steps.append(
            Step("upscale", False, "upscale_to_dpi={0:g} was asked for, but the page's "
                 "own dpi is unknown, and enlarging blind makes a page worse rather "
                 "than better - pass dpi= as well".format(target_dpi))
        )
        result.notes.append(
            "upscale_to_dpi={0:g} did nothing because dpi was not given. Pass both, "
            "or neither.".format(target_dpi)
        )
        return current
    if source_dpi >= target_dpi:
        result.steps.append(
            Step("upscale", False, "the page is already {0:g} dpi, at or above the "
                 "{1:g} dpi asked for".format(source_dpi, target_dpi))
        )
        return current
    factor = target_dpi / source_dpi
    bigger, detail = _steps.upscale(current, factor, result.binary)
    result.steps.append(
        Step("upscale", True, "{0} dpi to {1} dpi: {2}".format(
            "{0:g}".format(source_dpi), "{0:g}".format(target_dpi), detail))
    )
    result.output_dpi = float(target_dpi)
    return bigger


def clean_file(src: Any, dst: Any, **kw: Any) -> CleanResult:
    """Clean the page at ``src`` and write the result to ``dst``.

    Args:
        src: path to the scanned page.
        dst: path to write the cleaned page to. Missing parent directories are
            created, and the format follows the suffix.
        **kw: passed straight through to :func:`clean`.

    Returns:
        The same :class:`CleanResult` :func:`clean` would give you, with
        ``.destination`` set to the file that was written.

    Raises:
        FileNotFoundError: if ``src`` does not exist.
        OSError: if ``dst`` cannot be written.
        ValueError: on a bad argument, as :func:`clean`.
    """
    result = clean(src, **kw)
    result.source = os.fspath(src) if isinstance(src, (str, os.PathLike)) else result.source
    result.save(dst)
    return result


def estimate_skew(image: Any) -> float:
    """How far the text on ``image`` is turned from horizontal, in degrees.

    Positive is counter-clockwise, matching ``PIL.Image.rotate``, so
    ``image.rotate(-estimate_skew(image))`` puts the page straight. Accurate to
    a fraction of a degree on a page of text, and ``0.0`` on a page that has
    none to measure.

    Args:
        image: a path, a ``PIL.Image.Image``, or a numpy array, as
            :func:`clean` takes.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``image`` is none of the accepted kinds.
        ValueError: if the image has no pixels.
    """
    grey, _ = _prepare(image)
    plane = _images.to_array(grey)
    small, _ = _images.analysis_plane(plane, ANALYSIS_MAX_SIDE)
    return _analysis.estimate_skew_on_plane(small)
