"""Decide whether a scanned page is good enough to OCR before you pay to OCR it.

OCR is billed per page and a bad scan costs the same as a good one, then has to
be caught, re-scanned and re-run. This package reads the page first and answers
in one line::

    import document_quality

    report = document_quality.assess("invoice.png")
    print(report.summary())

    if not report.ocr_ready:
        for issue in report.issues:
            print(issue.kind, "->", issue.fix)

Every verdict carries the number behind it and the boundary it was compared
against, and every problem carries the concrete remedy: "rescan at 300 dpi",
"deskew by 2.3 degrees clockwise", "increase lighting on the left edge",
"crop the black border before OCR". Nothing here
downloads a model, calls out to a network, or needs OpenCV - it is numpy and
Pillow measuring a page.

What gets measured: effective resolution, skew angle, ink-to-paper contrast,
sharpness, uneven lighting (as a gradient of the paper level, so a shadow is
never mistaken for a scanner border), show-through from the reverse side, black
and white clipping, genuinely black scanner borders, text line height in
pixels, and how much of the page looks like text.
A blank sheet is reported as blank rather than as eight failures, and a
photograph is reported as not a document page rather than as a bad one.
"""
from __future__ import annotations

from ._assess import (
    SCORE_WEIGHTS,
    assess,
    assess_batch,
    detect_orientation,
    estimate_skew,
)
from ._report import PAGE_KINDS, SEVERITIES, BatchReport, Issue, Measure, PageReport
from ._thresholds import (
    DEFAULT_THRESHOLDS,
    PASS_SCORE,
    Thresholds,
    describe_thresholds,
)

__version__ = "0.1.0"

__all__ = [
    "assess",
    "assess_batch",
    "estimate_skew",
    "detect_orientation",
    "PageReport",
    "BatchReport",
    "Issue",
    "Measure",
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "describe_thresholds",
    "PAGE_KINDS",
    "SEVERITIES",
    "SCORE_WEIGHTS",
    "PASS_SCORE",
    "__version__",
]
