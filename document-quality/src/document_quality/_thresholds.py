"""Every boundary this package judges by, in one place with its reasoning.

Not one of these numbers is physics. They are the settings a 300 dpi office
scan of black text on white paper, headed for a general-purpose OCR engine,
usually meets. A microfilm frame, a carbon copy, a whiteboard photo and a
19th-century register all have a different notion of normal, so every boundary
is a field you can override rather than a literal buried in a function::

    report = document_quality.assess(page, thresholds={"target_dpi": 600})
    print(document_quality.describe_thresholds())

Each measure is given three points rather than one, and the score curve is bent
through all three: a ``target`` worth 100, a ``limit`` worth exactly the passing
mark, and a ``hopeless`` value worth 0. That is what keeps a measure's score and
its pass/fail from ever disagreeing - a measure scores at or above
:data:`PASS_SCORE` if and only if it is inside its limit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

#: Score a measure gets exactly at its limit. At or above this a measure passes.
PASS_SCORE = 60.0


@dataclass(frozen=True)
class Thresholds:
    """The boundaries every verdict in this package is made against.

    Construct one, or pass a dict of overrides anywhere ``thresholds=`` is
    accepted. Unknown names raise ``ValueError`` rather than being ignored.
    """

    #: Overall score at or above which a document page is called OCR-ready.
    ready_score: float = 60.0

    # -- resolution -------------------------------------------------------
    #: Dots per inch worth full marks. The usual advice for OCR is 300.
    target_dpi: float = 300.0
    #: Dots per inch below which a page is flagged. 200 is the practical floor.
    limit_dpi: float = 200.0
    #: Dots per inch at which resolution scores nothing at all.
    hopeless_dpi: float = 72.0
    #: A page whose claimed dpi implies a sheet wider than this is mis-tagged.
    max_credible_inches: float = 40.0
    #: A page whose claimed dpi implies a sheet narrower than this is too.
    min_credible_inches: float = 1.0

    # -- text size --------------------------------------------------------
    #: Inked line height, in pixels, worth full marks.
    target_text_height_px: float = 26.0
    #: Inked line height, in pixels, below which text is flagged as too small.
    limit_text_height_px: float = 16.0
    #: Inked line height at which text size scores nothing at all.
    hopeless_text_height_px: float = 6.0

    # -- geometry ---------------------------------------------------------
    #: Skew, in degrees either way, small enough to leave alone.
    target_skew_degrees: float = 0.1
    #: Skew, in degrees either way, beyond which the page wants deskewing.
    limit_skew_degrees: float = 0.5
    #: Skew at which the page scores nothing for geometry.
    hopeless_skew_degrees: float = 6.0

    # -- tone -------------------------------------------------------------
    #: Ink-to-paper separation, 0 to 1, worth full marks.
    target_contrast: float = 0.55
    #: Ink-to-paper separation below which text starts to dissolve into paper.
    limit_contrast: float = 0.25
    #: Ink-to-paper separation that scores nothing at all.
    hopeless_contrast: float = 0.05

    #: Edge acutance, 0 to 1, worth full marks. 1.0 would be a perfect
    #: one-pixel ink-to-paper step, which no real optical system reaches.
    target_sharpness: float = 0.45
    #: Edge acutance below which strokes are too soft to separate reliably.
    limit_sharpness: float = 0.18
    #: Edge acutance that scores nothing at all.
    hopeless_sharpness: float = 0.04

    # -- lighting and the reverse side ------------------------------------
    #: Fall in the paper level from the brightest part of the page to the
    #: darkest, as a share of the brightest, small enough to ignore.
    target_lighting: float = 0.04
    #: Fall beyond which one part of the page is badly lit. Past twice this
    #: the lighting issue is a failure rather than a warning.
    limit_lighting: float = 0.22
    #: Fall that scores nothing at all.
    hopeless_lighting: float = 0.60

    #: Share of the page showing soft grey marks from the reverse side that is
    #: small enough to ignore.
    target_show_through: float = 0.002
    #: Share of the page beyond which show-through will confuse an OCR engine.
    limit_show_through: float = 0.030
    #: Share of the page that scores nothing at all.
    hopeless_show_through: float = 0.150

    # -- clipping ---------------------------------------------------------
    #: Share of pixels at pure black worth full marks.
    target_black_clipping: float = 0.005
    #: Share of pixels at pure black beyond which strokes have lost their
    #: shape to a crushed black point.
    limit_black_clipping: float = 0.150
    #: Share of pixels at pure black that scores nothing at all.
    hopeless_black_clipping: float = 0.500
    #: Share of pixels at pure white above which faint content may have been
    #: erased by the scanner's white point. This only ever raises a warning:
    #: a bilevel scan is almost all pure white and OCRs perfectly well.
    warn_white_clipping: float = 0.990
    #: Text coverage below which high white clipping is worth warning about.
    faint_text_coverage: float = 0.010

    # -- what kind of page this is ----------------------------------------
    #: Ink-to-paper separation below which a sheet with no rows of text in
    #: its profile is blank. A faint sheet that does show rows of text is a
    #: document with a contrast problem, never a blank.
    blank_contrast: float = 0.120
    #: Share of the page that has to be inked before it is not blank (again
    #: only when the profile shows no rows of text).
    blank_ink_share: float = 0.0008
    #: Share of the page near paper white below which it is not a document.
    document_paper_share: float = 0.400
    #: Profile swing below which the page holds no rows of text.
    document_line_contrast: float = 1.000
    #: Colour spread above which the page is too colourful to be ink on paper.
    document_colour_spread: float = 0.120
    #: How many of the four document tests must fail before a page is called
    #: a photograph rather than a document.
    document_failed_tests: int = 2

    @classmethod
    def field_names(cls) -> List[str]:
        """Every overridable name, in declaration order."""
        return [item.name for item in fields(cls)]

    def replace(self, **changes: float) -> "Thresholds":
        """A copy with ``changes`` applied.

        Raises:
            ValueError: if a name is not a threshold, or a value is not a
                number.
        """
        known = set(self.field_names())
        for name, value in changes.items():
            if name not in known:
                raise ValueError(
                    "unknown threshold {0!r}; known names are {1}".format(
                        name, ", ".join(sorted(known))
                    )
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "threshold {0} must be a number, got {1!r}".format(name, value)
                )
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, float]:
        """Every threshold as a JSON-safe dict."""
        return {name: getattr(self, name) for name in self.field_names()}


#: The defaults, ready to use and safe to share: the dataclass is frozen.
DEFAULT_THRESHOLDS = Thresholds()

ThresholdLike = Union[None, Thresholds, Mapping[str, float]]


def resolve_thresholds(thresholds: ThresholdLike = None) -> Thresholds:
    """Turn ``None``, a ``Thresholds`` or a dict of overrides into a ``Thresholds``.

    Raises:
        TypeError: if ``thresholds`` is none of those.
        ValueError: if a dict names a threshold that does not exist.
    """
    if thresholds is None:
        return DEFAULT_THRESHOLDS
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        return DEFAULT_THRESHOLDS.replace(**dict(thresholds))
    raise TypeError(
        "thresholds must be None, a Thresholds or a dict, not {0}".format(
            type(thresholds).__name__
        )
    )


def _describe(name: str) -> str:
    """The doc comment written above ``name`` in the dataclass body."""
    return _DOC.get(name, "")


def describe_thresholds(thresholds: ThresholdLike = None) -> str:
    """A readable table of every threshold, its value and what it means.

    This is what ``document-quality --list-thresholds`` prints.
    """
    resolved = resolve_thresholds(thresholds)
    names = resolved.field_names()
    width = max(len(name) for name in names)
    lines = ["threshold{0}  value  meaning".format(" " * (width - 9))]
    lines.append("-" * (width + 9 + 60))
    for name in names:
        value = getattr(resolved, name)
        lines.append(
            "{0:<{1}}  {2:>5g}  {3}".format(name, width, value, _describe(name))
        )
    return "\n".join(lines)


#: One line of plain English per threshold, used by --list-thresholds.
_DOC: Dict[str, str] = {
    "ready_score": "overall score at or above which a page is OCR-ready",
    "target_dpi": "dots per inch worth full marks",
    "limit_dpi": "dots per inch below which resolution is flagged",
    "hopeless_dpi": "dots per inch scoring nothing at all",
    "max_credible_inches": "a wider implied sheet means the dpi tag is wrong",
    "min_credible_inches": "a narrower implied sheet means the same",
    "target_text_height_px": "inked line height in pixels worth full marks",
    "limit_text_height_px": "inked line height below which text is too small",
    "hopeless_text_height_px": "inked line height scoring nothing at all",
    "target_skew_degrees": "skew small enough to leave alone",
    "limit_skew_degrees": "skew beyond which the page wants deskewing",
    "hopeless_skew_degrees": "skew scoring nothing at all",
    "target_contrast": "ink-to-paper separation worth full marks",
    "limit_contrast": "separation below which text dissolves into paper",
    "hopeless_contrast": "separation scoring nothing at all",
    "target_sharpness": "edge acutance worth full marks",
    "limit_sharpness": "acutance below which strokes are too soft",
    "hopeless_sharpness": "acutance scoring nothing at all",
    "target_lighting": "fall in paper level across the page, ignorable",
    "limit_lighting": "fall beyond which part of the page is badly lit",
    "hopeless_lighting": "fall scoring nothing at all",
    "target_show_through": "share of page showing the reverse side, ignorable",
    "limit_show_through": "share beyond which show-through confuses OCR",
    "hopeless_show_through": "share scoring nothing at all",
    "target_black_clipping": "share of pixels at pure black worth full marks",
    "limit_black_clipping": "share at pure black that has crushed the strokes",
    "hopeless_black_clipping": "share at pure black scoring nothing at all",
    "warn_white_clipping": "share at pure white worth a warning",
    "faint_text_coverage": "text coverage below which that warning is raised",
    "blank_contrast": "separation below which a sheet with no text rows is blank",
    "blank_ink_share": "share of page inked before it is not blank",
    "document_paper_share": "share near paper white a document needs",
    "document_line_contrast": "profile swing a page of text rows needs",
    "document_colour_spread": "colour above which it is not ink on paper",
    "document_failed_tests": "failed document tests before it is a photograph",
}


def bend(
    value: Optional[float],
    target: float,
    limit: float,
    hopeless: float,
) -> Optional[float]:
    """Score ``value`` 0-100 on a curve bent through three named points.

    ``target`` scores 100, ``limit`` scores exactly :data:`PASS_SCORE`, and
    ``hopeless`` scores 0, with straight lines between and flat ends outside.
    It works in either direction: pass ``target`` above ``limit`` when more is
    better (contrast), or below it when less is better (skew).

    ``None`` in, ``None`` out, which is how a measure that did not apply keeps
    itself out of the overall score.
    """
    if value is None:
        return None
    if target <= limit:                       # less is better
        if value <= target:
            return 100.0
        if value >= hopeless:
            return 0.0
        if value <= limit:
            span = max(limit - target, 1e-12)
            return 100.0 - (100.0 - PASS_SCORE) * (value - target) / span
        span = max(hopeless - limit, 1e-12)
        return PASS_SCORE * (1.0 - (value - limit) / span)
    if value >= target:                       # more is better
        return 100.0
    if value <= hopeless:
        return 0.0
    if value >= limit:
        span = max(target - limit, 1e-12)
        return 100.0 - (100.0 - PASS_SCORE) * (target - value) / span
    span = max(limit - hopeless, 1e-12)
    return PASS_SCORE * (value - hopeless) / span


def bend_pair(value: Optional[float], points: Tuple[float, float, float]) -> Any:
    """:func:`bend` with the three points given as one tuple."""
    return bend(value, points[0], points[1], points[2])
