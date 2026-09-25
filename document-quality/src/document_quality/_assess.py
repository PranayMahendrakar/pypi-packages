"""The four public entry points, and the order the work happens in.

One page goes through the same seven steps every time:

1. :func:`document_quality._images.prepare` opens it once and builds the planes.
2. :func:`document_quality._lighting.find_border` looks for genuinely black
   bands along the image edges, and the planes are cropped to inside them;
   the black corners of a sheet scanned crooked on a dark lid are painted out.
3. :func:`document_quality._measures.analyse` computes what every measure
   shares, against a paper surface estimated everywhere on the page.
4. :func:`document_quality._skew.orient` settles which way up the page is,
   from ink flat-fielded by that same surface. A page lying on its side is
   turned upright, so that every line measure describes the page it will be
   once rotated.
5. The projection profile gives the skew and the line geometry.
6. :func:`document_quality._measures.classify` decides whether this is a
   document, a blank sheet or a photograph.
7. Each measure turns its number into a score and, when something is wrong, an
   :class:`~document_quality._report.Issue` carrying the remedy.

Two of those steps exist to stop the report from lying. Classification comes
before the issue list because a blank sheet should be told it is blank, not
handed eight complaints about text it does not have, and a photograph should be
told it is not a document page rather than advised to rescan at 300 dpi.
Resolution is scored only when a dpi is actually known, because a pixel count
alone cannot tell a 300 dpi letter page from a 600 dpi receipt.

The caller's image is never modified. Arrays are copied on the way in, planes
are freshly allocated, and nothing here writes through a view of the original.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import _images, _lighting, _measures, _skew
from ._lighting import Border
from ._images import PagePlanes
from ._measures import PageStats
from ._report import BatchReport, Issue, Measure, PageReport
from ._skew import LineGeometry
from ._thresholds import ThresholdLike, Thresholds, resolve_thresholds

logger = logging.getLogger(__name__)

#: How much each measure counts towards the overall score. Measures that did
#: not apply drop out and the rest are re-weighted, so a file with no dpi is
#: not punished for the resolution measure it could not take.
SCORE_WEIGHTS: Dict[str, float] = {
    "resolution": 1.0,
    "text_size": 1.5,
    "skew": 1.0,
    "contrast": 1.5,
    "sharpness": 1.5,
    "lighting": 0.8,
    "show_through": 0.8,
    "clipping": 0.5,
}

def _prepare(image: Any, dpi: Any) -> Tuple[PagePlanes, Border]:
    """Open ``image``, crop to inside any black border, and paint out the lid.

    A border is a black band along a whole side, and is cropped off and
    reported. What is left of the black after that - the wedges in the corners
    of a sheet scanned crooked on a dark lid - is not part of the page either,
    but it cannot be cropped without cropping the page, so it is painted over
    with paper on every plane. Left in, a blank sheet scanned crooked would
    read as a page of solid black triangles, and their edges as lines of text.
    """
    planes = _images.prepare(image, dpi)
    border = _lighting.find_border(planes.work, planes.work_scale)
    if border.found:
        planes = planes.cropped(border.box(planes.width, planes.height))
    mask, share, black_level = _lighting.find_outside(planes.work)
    if share > 0.0:
        planes = _paint_outside(planes, mask, share, black_level)
    return planes, border


def _paint_outside(
    planes: PagePlanes, mask: np.ndarray, share: float, black_level: float
) -> PagePlanes:
    """``planes`` with the black lid outside the sheet painted as paper."""
    paper = float(np.percentile(planes.work[~mask], 90.0)) if (~mask).any() else 1.0
    work = planes.work.copy()
    work[mask] = np.float32(paper)
    grow = _lighting.OUTSIDE_GROW_PX
    edge = _lighting.BORDER_EDGE_PX
    step = _lighting.BORDER_STEP * float(np.percentile(planes.work, 90.0))
    fine = planes.fine.copy()
    fine_ratio = planes.fine_scale / max(planes.work_scale, 1e-9)
    fine[_lighting.outside_mask(
        fine, black_level, int(np.ceil(grow * fine_ratio)), step,
        int(np.ceil(edge * fine_ratio)),
    )] = np.float32(paper)
    lum = planes.lum.copy()
    native_ratio = 1.0 / max(planes.work_scale, 1e-9)
    lum[_lighting.outside_mask(
        lum, black_level * 255.0, int(np.ceil(grow * native_ratio)), step * 255.0,
        int(np.ceil(edge * native_ratio)),
    )] = int(round(paper * 255.0))
    return dataclasses.replace(planes, lum=lum, work=work, fine=fine, outside_share=share)


def _skew_and_lines(
    planes: PagePlanes, light: Optional[_lighting.PaperSurface] = None
) -> Tuple[float, LineGeometry, bool]:
    """The skew angle, the line geometry it implies, and whether there are lines.

    Both planes are flat-fielded against the page's paper surface first. The
    angle is swept on the small plane, where hundreds of candidate shears cost
    milliseconds, and refined on the work plane; the geometry it implies is
    measured once on the work plane, where a line of text is enough pixels
    tall to measure.
    """
    ink = _skew.ink_planes(planes, light)
    degrees, _ = _skew.skew_of(ink)
    geometry = _skew.line_geometry_of_ink(ink.work, degrees, ink.work_bank())
    return degrees, geometry, _skew.holds_text_lines(geometry, planes.work.shape[0])


def _describe_source(image: Any, given: Optional[str]) -> str:
    """A short name for whatever the caller passed in."""
    if given is not None:
        return str(given)
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    if isinstance(image, np.ndarray):
        return "<array {0}>".format("x".join(str(n) for n in image.shape))
    if isinstance(image, Image.Image):
        name = getattr(image, "filename", None)
        return str(name) if name else "<image {0}x{1}>".format(*image.size)
    return "<image>"


def _overall_score(measures: Dict[str, Measure]) -> float:
    """Weighted mean of the measures that applied and produced a score."""
    total = 0.0
    weight = 0.0
    for name, measure in measures.items():
        if measure.score is None or not measure.applies:
            continue
        share = SCORE_WEIGHTS.get(name, 1.0)
        total += share * float(measure.score)
        weight += share
    if weight <= 0.0:
        return 0.0
    return float(np.clip(total / weight, 0.0, 100.0))


def _blank_issue(stats: PageStats, reasons: Sequence[str]) -> Issue:
    """The one thing worth saying about a sheet with nothing on it.

    Severity is ``info``, not a failure: a blank page is a fact about the
    stack of paper, not a fault in the scan. Nothing else is reported, because
    every other complaint would be about text that is not there.
    """
    why = reasons[0] if reasons else "nothing was found on the sheet"
    return Issue(
        "blank",
        "info",
        "This sheet is blank: {0}. There is nothing on it to OCR.".format(why),
        "skip this page, or check the scanner fed the printed side of the sheet",
    )


def _photograph_issue(reasons: Sequence[str]) -> Issue:
    """Why this is not a document page, and what to do with it instead."""
    why = "; ".join(reasons) if reasons else "it does not look like ink on paper"
    return Issue(
        "not_a_document",
        "warning",
        "This does not look like a document page: {0}.".format(why),
        "send this to an image pipeline rather than an OCR one; if it really is "
        "a page, crop to the sheet and rescan it as a document",
    )


def _orientation_issue(turn: int) -> Optional[Issue]:
    """The page is not the right way up, and every measure below it is sideways."""
    if turn == 0:
        return None
    if turn == 180:
        return Issue(
            "orientation", "failure",
            "The text on this page reads upside down.",
            "rotate the page 180 degrees before OCR",
        )
    direction = "counter-clockwise" if turn == 90 else "clockwise"
    return Issue(
        "orientation", "failure",
        "The text on this page runs sideways, so the page is lying on its side.",
        "rotate the page 90 degrees {0} before OCR".format(direction),
    )


def _measure_page(
    planes: PagePlanes,
    stats: PageStats,
    geometry: LineGeometry,
    degrees: float,
    thresholds: Thresholds,
    kind: str = "document",
) -> Tuple[Dict[str, Measure], List[Issue], Optional[float]]:
    """Run every measure; return them by name, their issues, and the text height."""
    measures: Dict[str, Measure] = {}
    issues: List[Issue] = []

    def add(built: Tuple[Measure, Optional[Issue]]) -> Measure:
        measure, issue = built
        measures[measure.name] = measure
        if issue is not None:
            issues.append(issue)
        return measure

    add(_measures.resolution(planes, thresholds))

    size_measure, size_issue, text_height_px = _measures.text_size(
        planes, geometry, planes.work_scale, thresholds, document=kind == "document"
    )
    measures[size_measure.name] = size_measure
    if size_issue is not None:
        issues.append(size_issue)

    add(_measures.skew(degrees, thresholds, applies=geometry.text_height is not None))
    add(_measures.contrast(stats, thresholds))
    add(_measures.sharpness(stats, text_height_px, thresholds))
    add(_measures.lighting(stats, thresholds))
    add(_measures.show_through(stats, thresholds))
    add(_measures.clipping(stats, thresholds))
    add(_measures.border(stats))
    measures["text_coverage"] = _measures.text_coverage(stats)
    return measures, issues, text_height_px


def assess(
    image: Any,
    *,
    dpi: Optional[float] = None,
    thresholds: ThresholdLike = None,
    source: Optional[str] = None,
    check_orientation: bool = True,
) -> PageReport:
    """Decide whether one scanned page is worth sending to an OCR engine.

    Args:
        image: a path, a ``PIL.Image.Image``, or a numpy array shaped
            ``(h, w)``, ``(h, w, 1)``, ``(h, w, 3)`` or ``(h, w, 4)``. Greyscale
            and colour are both fine, and the caller's image is never modified.
        dpi: the scan resolution, if you know it. When it is ``None`` the
            file's own resolution tag is used, and when the file has none
            either, resolution advice is left out of the report rather than
            guessed from the pixel count.
        thresholds: a :class:`~document_quality.Thresholds` or a dict of
            overrides, for pages that are not 300 dpi office scans.
        source: a name for this page in the report. Defaults to the path, or a
            description of the in-memory image.
        check_orientation: whether to also test the three quarter-turns, and
            measure a sideways page as the upright page it will be. Costs a
            fraction of a skew search; pass ``False`` for pages you know are
            the right way up.

    Returns:
        A :class:`~document_quality.PageReport`. Start with ``.ocr_ready``,
        ``.score`` and ``.issues``; every issue carries its own ``.fix``.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``image`` is none of the accepted kinds.
        ValueError: if the image has no pixels, or ``dpi`` is not positive.
    """
    settings = resolve_thresholds(thresholds)
    name = _describe_source(image, source)
    planes, border = _prepare(image, dpi)

    stats = _measures.analyse(planes, settings, border)

    # Orientation is settled before any line is measured. A page lying on its
    # side has its lines running down the image, and every line measure taken
    # that way round - text height, skew, the text-row test itself - would be
    # a measure of the wrong axis. So a sideways page is turned upright first
    # and reported as the page it will be once rotated. A page whose tone
    # already rules it out as a document skips the search entirely.
    turn, basis = 0, "none"
    found: Optional[_skew.Orientation] = None
    if check_orientation and not _measures.settled_without_lines(planes, stats, settings):
        found = _skew.orient(planes, stats.light)
        turn, basis = found.degrees, found.basis
        if turn in (90, 270):
            planes = planes.turned(turn // 90)
            stats = _measures.turn_stats(stats, turn // 90)
    if found is not None:
        # The orientation search already measured the skew and the lines of
        # the page the right way round; searching again would find the same.
        degrees, geometry = found.skew, found.geometry
    else:
        degrees, geometry, _ = _skew_and_lines(planes, stats.light)
    kind, evidence, reasons = _measures.classify(planes, stats, geometry, settings)
    if kind != "document":
        # A blank sheet or a photograph has no lines of text, whatever the
        # profile of its grain or its horizon happened to suggest. Reporting a
        # text height or a skew for it would be a number about nothing.
        degrees, geometry = 0.0, LineGeometry(line_contrast=geometry.line_contrast)
        turn = 0

    measures, issues, text_height_px = _measure_page(
        planes, stats, geometry, degrees, settings, kind
    )
    notes: List[str] = []
    if planes.exif_applied:
        notes.append(
            "The file carried an EXIF orientation tag, which was applied before "
            "measuring, so these numbers describe the upright page."
        )

    if border.found:
        notes.append(
            "A black scanner border was cropped off the {0} before measuring, so "
            "these numbers describe the page inside it.".format(border.describe())
        )
    if planes.outside_share > 0.0:
        notes.append(
            "The sheet lies crooked on a dark lid; the black corners outside it "
            "({0:.1%} of the image) were left out of every measure.".format(
                planes.outside_share
            )
        )

    if turn in (90, 270):
        notes.append(
            "The text runs down the image, so the page was measured as if "
            "already turned upright; the numbers describe it after rotation."
        )
    if kind == "document" and basis == "lines":
        if turn:
            notes.append(
                "Which way the lines run was clear, but upright versus upside "
                "down was not, so a 180 degree turn cannot be ruled out."
            )
        else:
            # Silence here would read as "checked, and upright". Say instead
            # that the letters gave no clear sign either way - a soft scan,
            # or text set in capitals or figures - so an upside-down page is
            # still possible.
            notes.append(
                "The letters give no clear sign of which way up the page "
                "reads, so it was taken to be upright; an upside-down page "
                "cannot be ruled out."
            )

    # A page that is not a document gets one thing said about it, not eight.
    # Telling the owner of a photograph to increase the lighting on its left
    # edge is advice about a page they do not have.
    if kind == "blank":
        issues = [_blank_issue(stats, reasons)]
    elif kind == "photograph":
        issues = [_photograph_issue(reasons)]
    else:
        turn_issue = _orientation_issue(turn)
        if turn_issue is not None:
            issues.insert(0, turn_issue)
        issues.sort(key=lambda item: item.rank)

    # The score answers "how good is this for OCR", so a sheet with nothing to
    # read scores nothing however cleanly it was scanned. The per-measure
    # scores are still in .measures for anyone who wants the scan quality.
    score = _overall_score(measures) if kind == "document" else 0.0
    ready = (
        kind == "document"
        and score >= settings.ready_score
        and not any(item.severity == "failure" for item in issues)
    )

    return PageReport(
        source=name,
        width=planes.width,
        height=planes.height,
        kind=kind,
        ocr_ready=bool(ready),
        score=score,
        skew_degrees=float(degrees),
        estimated_text_height_px=text_height_px,
        issues=issues,
        measures=measures,
        dpi=planes.dpi,
        dpi_source=planes.dpi_source,
        evidence=dict(evidence),
        notes=notes,
    )


def assess_batch(
    images: Iterable[Any],
    *,
    dpi: Optional[float] = None,
    thresholds: ThresholdLike = None,
    check_orientation: bool = True,
) -> BatchReport:
    """Assess many pages and sort the ones needing attention to the front.

    A page that cannot be read does not stop the batch: the error is recorded
    against that page in :attr:`~document_quality.BatchReport.failures` and the
    rest are assessed.

    Args:
        images: any iterable of the things :func:`assess` accepts.
        dpi: applied to every page that does not carry its own.
        thresholds: as for :func:`assess`.
        check_orientation: as for :func:`assess`.

    Returns:
        A :class:`~document_quality.BatchReport`. Start with ``.not_ready``.

    Raises:
        TypeError: if ``images`` is a single image or a bare string rather than
            an iterable of them.
    """
    if isinstance(images, (str, bytes, os.PathLike, Image.Image, np.ndarray)):
        raise TypeError(
            "assess_batch takes an iterable of images; pass [image] for one, or "
            "call assess(image)"
        )
    settings = resolve_thresholds(thresholds)
    batch = BatchReport()
    for item in images:
        name = _describe_source(item, None)
        try:
            batch.reports.append(
                assess(
                    item, dpi=dpi, thresholds=settings,
                    check_orientation=check_orientation,
                )
            )
        except (OSError, ValueError, TypeError) as error:
            logger.warning("could not assess %s: %s", name, error)
            batch.failures.append({"source": name, "error": str(error)})
    return batch


def estimate_skew(image: Any) -> float:
    """How far the text on ``image`` is turned from horizontal, in degrees.

    Positive is counter-clockwise, matching ``PIL.Image.rotate``, so
    ``image.rotate(-estimate_skew(image))`` puts the page straight. A page with
    no text lines on it returns ``0.0``.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``image`` is none of the accepted kinds.
        ValueError: if the image has no pixels.
    """
    planes, _ = _prepare(image, None)
    degrees, _, has_lines = _skew_and_lines(planes)
    return float(degrees) if has_lines else 0.0


def detect_orientation(image: Any) -> int:
    """Which quarter turn sets ``image`` upright: ``0``, ``90``, ``180`` or ``270``.

    The answer is counter-clockwise degrees, so
    ``image.rotate(detect_orientation(image), expand=True)`` is the correction.
    A page with no text lines on it returns ``0``, because there is nothing to
    be upright about.

    Raises:
        FileNotFoundError: if a path does not exist.
        TypeError: if ``image`` is none of the accepted kinds.
        ValueError: if the image has no pixels.
    """
    planes, _ = _prepare(image, None)
    turn, _ = _skew.detect_orientation_on_plane(planes)
    return int(turn)
