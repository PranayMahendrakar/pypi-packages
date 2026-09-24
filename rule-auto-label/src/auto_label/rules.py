"""Rules: how a label is asserted for an item, and how a rule is matched against data."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from auto_label._data import TABULAR, Prepared

logger = logging.getLogger(__name__)

Pattern = Union[str, "re.Pattern[str]"]


@dataclass
class Rule:
    """One labelling rule. A rule fires when *every* condition it carries matches.

    - ``keywords``: case-insensitive whole words / phrases matched against text
    - ``regex``: a pattern searched in the text (compile with ``(?i)`` for case-insensitive)
    - ``query``: a pandas query string, evaluated on the DataFrame (tabular data only)
    - ``func``: ``callable(item) -> bool``; the item is the text, or the row as a Series
      whose ``.name`` is that row's label in the caller's index
    - ``weight``: votes this rule adds to its label when it fires (negative vetoes)
    """

    label: str
    keywords: Tuple[str, ...] = ()
    regex: Optional["re.Pattern[str]"] = None
    query: Optional[str] = None
    func: Optional[Callable[[Any], bool]] = None
    weight: float = 1.0
    keyword_pattern: Optional["re.Pattern[str]"] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.keywords and self.keyword_pattern is None:
            try:
                self.keyword_pattern = compile_keywords(self.keywords)
            except ValueError as exc:
                raise ValueError(f"rule {self.label!r}: {exc}") from None

    @property
    def conditions(self) -> List[str]:
        """Names of the conditions this rule carries, in evaluation order."""
        names = []
        if self.keywords:
            names.append("keywords")
        if self.regex is not None:
            names.append("regex")
        if self.query is not None:
            names.append("query")
        if self.func is not None:
            names.append("func")
        return names

    def describe(self) -> str:
        """Short human-readable form, e.g. ``spam: keywords=('free', 'prize') weight=1.0``."""
        bits = []
        if self.keywords:
            bits.append(f"keywords={self.keywords!r}")
        if self.regex is not None:
            bits.append(f"regex={self.regex.pattern!r}")
        if self.query is not None:
            bits.append(f"query={self.query!r}")
        if self.func is not None:
            bits.append(f"func={getattr(self.func, '__name__', repr(self.func))}")
        return f"{self.label}: {' '.join(bits)} weight={self.weight:g}"


_UNSPACED_RANGES = (
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
)


def _unspaced(ch: str) -> bool:
    """True for scripts that do not separate words with spaces (CJK, Thai, ...)."""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _UNSPACED_RANGES)


def compile_keywords(keywords: Iterable[str]) -> "re.Pattern[str]":
    """Build one case-insensitive pattern matching any keyword as a whole word / phrase.

    Whitespace inside a phrase matches any run of whitespace. Word boundaries are
    taken as "not adjacent to another word character", which also works for keywords
    that start or end with punctuation (``c++``, ``#promo``). Keywords in scripts
    written without spaces (Chinese, Japanese, Korean, Thai) match anywhere.
    """
    cleaned = [str(kw) for kw in keywords]
    if any(not kw.strip() for kw in cleaned):
        raise ValueError("keywords must all be non-empty strings")
    if not cleaned:
        raise ValueError("keywords is empty")
    parts = []
    for kw in sorted(cleaned, key=len, reverse=True):
        kw = kw.strip()
        body = r"\s+".join(re.escape(t) for t in kw.split())
        lead = "" if _unspaced(kw[0]) else r"(?<!\w)"
        trail = "" if _unspaced(kw[-1]) else r"(?!\w)"
        parts.append(f"(?:{lead}{body}{trail})")
    return re.compile("|".join(parts), re.IGNORECASE)


def make_rule(
    label: str,
    *,
    keywords: Union[None, str, Iterable[str]] = None,
    regex: Optional[Pattern] = None,
    query: Optional[str] = None,
    func: Optional[Callable[[Any], bool]] = None,
    weight: float = 1.0,
) -> Rule:
    """Validate arguments and build a :class:`Rule`; raises ``ValueError`` on bad input."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("rule label must be a non-empty string")

    kw_tuple: Tuple[str, ...] = ()
    if keywords is not None:
        if isinstance(keywords, str):
            keywords = [keywords]
        try:
            kw_list = [str(k).strip() for k in keywords]
        except TypeError:
            raise ValueError(f"rule {label!r}: keywords must be a list of strings") from None
        kw_tuple = tuple(k for k in kw_list if k)
        if not kw_tuple:
            raise ValueError(f"rule {label!r}: keywords is empty")

    compiled: Optional["re.Pattern[str]"] = None
    if regex is not None:
        if isinstance(regex, re.Pattern):
            compiled = regex
        elif isinstance(regex, str):
            if not regex:
                raise ValueError(f"rule {label!r}: regex is empty")
            try:
                compiled = re.compile(regex)
            except re.error as exc:
                raise ValueError(f"rule {label!r}: invalid regex {regex!r}: {exc}") from None
        else:
            raise ValueError(f"rule {label!r}: regex must be a string or compiled pattern")

    if query is not None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"rule {label!r}: query must be a non-empty pandas query string")
        query = query.strip()

    if func is not None and not callable(func):
        raise ValueError(f"rule {label!r}: func must be callable")

    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
        raise ValueError(f"rule {label!r}: weight must be a finite number, got {weight!r}")

    rule = Rule(label=label, keywords=kw_tuple, regex=compiled, query=query, func=func, weight=float(weight))
    if not rule.conditions:
        raise ValueError(f"rule {label!r}: give at least one of keywords, regex, query or func")
    return rule


def match_rule(rule: Rule, prepared: Prepared) -> np.ndarray:
    """Boolean mask over the items: where every condition of ``rule`` holds."""
    n = len(prepared)
    mask = np.ones(n, dtype=bool)
    if n == 0:
        return mask

    if rule.keyword_pattern is not None:
        pat = rule.keyword_pattern
        mask &= np.fromiter((pat.search(t) is not None for t in prepared.texts), dtype=bool, count=n)
        if not mask.any():
            return mask

    if rule.regex is not None:
        pat = rule.regex
        mask &= np.fromiter((pat.search(t) is not None for t in prepared.texts), dtype=bool, count=n)
        if not mask.any():
            return mask

    if rule.query is not None:
        if prepared.mode != TABULAR or prepared.frame is None:
            raise ValueError(
                f"rule {rule.label!r} uses query={rule.query!r}, which needs a DataFrame; "
                "text input only supports keywords, regex and func"
            )
        try:
            hits = prepared.frame.query(rule.query, engine="python")
        except Exception as exc:
            raise ValueError(f"rule {rule.label!r}: query {rule.query!r} failed: {exc}") from None
        qmask = np.zeros(n, dtype=bool)
        qmask[np.asarray(hits.index, dtype=int)] = True
        mask &= qmask
        if not mask.any():
            return mask

    if rule.func is not None:
        func = rule.func
        tabular = prepared.mode == TABULAR and prepared.frame is not None
        frame = prepared.frame
        texts = prepared.texts
        # The frame is held re-indexed 0..n-1 so masks and query hits stay positional,
        # but that is an internal detail: a func rule keyed on the row's identity
        # (row.name in {"T-100", ...}) must see the label the caller passed in, not the
        # renumbered one. Otherwise such a rule matches nothing, silently, and disagrees
        # with result.index - which does keep the caller's labels.
        index = prepared.index
        rename = tabular and len(index) == n
        for i in np.flatnonzero(mask):
            pos = int(i)
            if not tabular:
                item: Any = texts[pos]
            else:
                item = frame.iloc[pos]
                if rename:
                    item = item.rename(index[pos])
            try:
                mask[pos] = bool(func(item))
            except Exception as exc:  # noqa: BLE001 - name the rule that failed
                raise ValueError(
                    f"rule {rule.label!r}: func raised on item {pos} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc

    return mask
