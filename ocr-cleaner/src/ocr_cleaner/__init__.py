"""ocr-cleaner: prepare a scanned page so OCR reads it better.

Deskew, denoise and threshold, in one call, with every step reporting what it
did to the page and why - including the steps that looked and decided to do
nothing.

    >>> import numpy as np, ocr_cleaner
    >>> page = np.full((1100, 850), 246, dtype=np.uint8)   # a sheet of paper
    >>> for top in range(120, 1000, 34):                   # rows of text on it
    ...     page[top:top + 11, 90:760] = 50
    >>> result = ocr_cleaner.clean(page)
    >>> result.page_kind
    'document'
    >>> "threshold" in result.applied
    True

A blank sheet is reported blank and handed back untouched rather than
thresholded into a field of speckles, and a photograph is reported as a
photograph rather than cleaned as a bad scan.

Pure numpy and Pillow. No OpenCV, no OCR engine, no model download, nothing
touches the network, and the same page always gives the same result.
"""
from __future__ import annotations

from ._analysis import MAX_SKEW_DEGREES, PAGE_KINDS
from ._core import (
    ANALYSIS_MAX_SIDE,
    DEFAULT_THRESHOLD,
    THRESHOLD_MODES,
    clean,
    clean_file,
    estimate_skew,
)
from ._images import IMAGE_SUFFIXES, open_image
from ._result import STEP_NAMES, CleanResult, Step

__version__ = "0.1.0"

__all__ = [
    "clean",
    "clean_file",
    "estimate_skew",
    "CleanResult",
    "Step",
    "STEP_NAMES",
    "THRESHOLD_MODES",
    "DEFAULT_THRESHOLD",
    "PAGE_KINDS",
    "MAX_SKEW_DEGREES",
    "ANALYSIS_MAX_SIDE",
    "IMAGE_SUFFIXES",
    "open_image",
    "__version__",
]
