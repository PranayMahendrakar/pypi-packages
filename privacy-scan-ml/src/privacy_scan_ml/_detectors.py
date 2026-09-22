"""Detectors: one compiled pattern plus a validator per PII type, and overlap resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import _validators as v
from ._hints import SENSITIVE_TYPES

# Types found by pattern, in both free text and cell values; order is the overlap priority.
PATTERN_TYPES = (
    "email",
    "url",
    "ipv6",
    "credit_card",
    "aadhaar",
    "pan",
    "phone",
    "ipv4",
    "date_of_birth",
    "address",
    "postal_code",
)
# Types found only from the column name plus the shape of the values.
COLUMN_ONLY_TYPES = ("person_name",) + SENSITIVE_TYPES
ALL_TYPES = PATTERN_TYPES + COLUMN_ONLY_TYPES

# Severity used for the report's risk level.
CRITICAL_TYPES = frozenset(
    {"aadhaar", "pan", "credit_card", "religion", "caste", "ethnicity", "sexual_orientation",
     "disability", "political_opinion", "health_condition"}
)
HIGH_TYPES = frozenset({"email", "phone", "date_of_birth", "person_name", "address"})
LOW_TYPES = frozenset({"ipv4", "ipv6", "url", "postal_code", "gender", "nationality"})


@dataclass(frozen=True)
class Match:
    """A validated hit inside a blob of text: half-open span and the exact text matched."""

    type: str
    start: int
    end: int
    text: str


ContextCheck = Callable[[str, int, int, str], bool]


@dataclass(frozen=True)
class Detector:
    name: str
    pattern: "re.Pattern[str]"
    validate: Callable[[str], Optional[str]]
    priority: int
    gate: Optional["re.Pattern[str]"] = None
    context: Optional[ContextCheck] = None  # extra evidence needed around the match (dob, postal)


# --------------------------------------------------------------------------- patterns

# An ASCII-only class here missed every RFC 6531 address (a non-ASCII local part, an IDN
# domain) outright, and an address the detector misses is one mask() leaves in the data.
# "\w" is not enough either: a Devanagari vowel sign is a combining mark, so आशा@... would
# still be lost. Anything non-ASCII is admitted here and email_valid() has the final say.
_ALNUM = r"(?:[^\W_]|[^\x00-\x7f])"
_LOCAL_CH = r"(?:[\w.%+-]|[^\x00-\x7f])"
_LABEL_CH = r"(?:[\w-]|[^\x00-\x7f])"
_EMAIL = re.compile(
    r"(?<![\w.+-])" + _ALNUM + r"(?:" + _LOCAL_CH + r"{0,62}" + _ALNUM + r")?"
    r"@(?:" + _ALNUM + r"(?:" + _LABEL_CH + r"{0,61}" + _ALNUM + r")?\.)+"
    r"(?:xn--[A-Za-z0-9-]{2,59}|" + _ALNUM + r"{2,24})(?![\w-])",
    re.IGNORECASE,
)
_URL = re.compile(
    r"(?<![\w@.])(?:(?:https?|ftp)://[^\s<>\"'`]+"
    r"|www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[^\s<>\"'`]*)?)",
    re.IGNORECASE,
)
_IPV6 = re.compile(
    r"(?<![\w:.])(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:.])"
)
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_AADHAAR = re.compile(r"(?<![\d-])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![\d-])")
# pan_valid() accepts an all-upper or an all-lower PAN, so the pattern has to offer both
# or the two disagree on what a PAN is; mixed case stays rejected, by the validator.
_PAN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]"
    r"|[a-z]{3}[abcfghljpt][a-z][0-9]{4}[a-z])(?![A-Za-z0-9])"
)
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,5}\)[ .-]?|\d{2,5}[ .-]?)?\d{3,5}[ .-]?\d{3,5}(?!\w)"
)
_MONTH = (
    r"(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DAY = r"0?[1-9]|[12]\d|3[01]"
_DATE = re.compile(
    r"(?<![\w/.-])(?:"
    r"(?P<y1>(?:19|20)\d{2})[-/.](?P<m1>0?[1-9]|1[0-2])[-/.](?P<d1>" + _DAY + r")"
    r"|(?P<a2>" + _DAY + r")[-/.](?P<b2>" + _DAY + r")[-/.](?P<y2>(?:19|20)\d{2})"
    r"|(?P<d3>" + _DAY + r")(?:st|nd|rd|th)?[ \t]+(?P<m3>" + _MONTH + r")\.?,?[ \t]+(?P<y3>(?:19|20)\d{2})"
    r"|(?P<m4>" + _MONTH + r")\.?[ \t]+(?P<d4>" + _DAY + r")(?:st|nd|rd|th)?,?[ \t]+(?P<y4>(?:19|20)\d{2})"
    r")(?![\w/.-])"
)
_POSTAL = re.compile(r"(?<![\w-])(?:[1-9]\d{2}[ ]?\d{3}|\d{5}(?:-\d{4})?)(?![\w-])")

_STREET_FULL = (
    "street road avenue lane boulevard drive nagar colony marg sector block phase layout enclave "
    "society apartment apartments residency tower towers heights villa villas plaza complex chowk "
    "gali mohalla extension vihar puram peth wadi cross circle court place highway square terrace "
    "garden gardens estate market bazaar bazar chawl path parkway trail crescent close mews grove "
    "quay wharf"
).split()
_STREET_ABBR = "st rd ave ln blvd dr ct pl sq hwy apt ste ext extn".split()
STREET_WORDS = frozenset(_STREET_FULL + _STREET_ABBR)
_STREET_ALT = "|".join(
    [w.capitalize() + "|" + w.upper() for w in _STREET_FULL]
    + [w + "|" + w.capitalize() + "|" + w.upper() for w in _STREET_ABBR]
)
_UNIT = r"(?:Flat|FLAT|Apt|APT|Apartment|Unit|UNIT|Suite|SUITE|Plot|PLOT|House|HOUSE|Shop|SHOP|Office|Room|Door|Floor|H\.?No\.?|No\.|#)"
_NUMBER = r"\d{1,5}(?:st|nd|rd|th)?[A-Za-z]?(?:[/-][A-Za-z0-9]{1,5})?"
_TOKEN = r"[A-Za-z][\w'.&-]{0,24}"
_ADDRESS = re.compile(
    r"(?<![\w/#-])"
    r"(?:" + _UNIT + r"[ \t]*[A-Za-z0-9/-]{1,8},?[ \t]+(?:" + _NUMBER + r",?[ \t]+)?"
    r"|" + _NUMBER + r",?[ \t]+)"
    r"(?:" + _TOKEN + r",?[ \t]+){0,5}?"
    r"(?:" + _STREET_ALT + r")\.?(?!\w)"
    r"(?:,?[ \t]+(?:" + _TOKEN + r",?[ \t]+){0,5}?(?:-[ \t]*)?\d{6}(?!\d))?"
)

# cheap gates: skip a detector when the blob cannot contain a hit
_GATE_AT = re.compile(r"@")
_GATE_URL = re.compile(r"://|www\.", re.IGNORECASE)
_GATE_COLON2 = re.compile(r":[0-9A-Fa-f:]*:")
_GATE_DOT_DIGIT = re.compile(r"\d\.\d")
_GATE_DIGITS3 = re.compile(r"\d{3}")
_GATE_DIGITS4 = re.compile(r"\d{4}")
_GATE_PAN = re.compile(r"[A-Za-z]{5}\d{4}[A-Za-z]")
_GATE_YEAR = re.compile(r"(?:19|20)\d{2}")
# "221B Baker Street" and "12A Nehru Road" carry no digit-then-space, so the old gate
# vetoed them before the address pattern ever ran. A digit beside a letter counts too.
_GATE_ADDRESS = re.compile(r"\d[ \t,A-Za-z]|[ \t#]\d|^\d", re.MULTILINE)

# context cues (looked for on the same line, just before the match)
_DOB_CUE = re.compile(r"(?i)(?:\bborn\b|\bbirth|\bdob\b|d\.o\.b|\bb'?day\b|\bbirthday\b)[^\n]{0,30}$")
_POSTAL_CUE = re.compile(
    r"(?i)(?:\bpin\b|\bpincode\b|pin[ \t]?code|\bzip\b|zip[ \t]?code|\bpostal\b|post[ \t]?code)[^\n]{0,20}$"
)
_STATE_CUE = re.compile(r"\b[A-Z]{2},?[ \t]+$")
_ZIP4 = re.compile(r"\d{5}-\d{4}")


def _before(blob: str, start: int, width: int = 40) -> str:
    """Up to ``width`` characters before ``start`` on the same line."""
    return blob[max(0, start - width) : start].rsplit("\n", 1)[-1]


def _dob_context(blob: str, start: int, end: int, text: str) -> bool:
    return bool(_DOB_CUE.search(_before(blob, start)))


def _postal_context(blob: str, start: int, end: int, text: str) -> bool:
    if _ZIP4.fullmatch(text):
        return True
    before = _before(blob, start, 30)
    return bool(_POSTAL_CUE.search(before) or _STATE_CUE.search(before))


# --------------------------------------------------------------------------- validators


def _keep_if(check: Callable[[str], bool]) -> Callable[[str], Optional[str]]:
    def validate(text: str) -> Optional[str]:
        return text if check(text) else None

    return validate


def _validate_url(text: str) -> Optional[str]:
    ok, cleaned = v.url_valid(text)
    return cleaned if ok else None


def _validate_date(text: str) -> Optional[str]:
    m = _DATE.fullmatch(text)
    if m is None:
        return None
    g = m.groupdict()
    if g["y1"]:
        parsed = v.parse_date_parts(g["y1"], g["m1"], g["d1"])
    elif g["y2"]:
        parsed = v.parse_ambiguous(g["a2"], g["b2"], g["y2"])
    elif g["y3"]:
        parsed = v.parse_date_parts(g["y3"], g["m3"], g["d3"])
    else:
        parsed = v.parse_date_parts(g["y4"], g["m4"], g["d4"])
    if parsed is None or not v.plausible_birth_date(parsed):
        return None
    return text


def _validate_postal(text: str) -> Optional[str]:
    digits = v.only_digits(text)
    if len(digits) == 6:
        return text if digits[0] != "0" and len(set(digits)) > 1 else None
    if len(digits) in (5, 9):
        return text if len(set(digits[:5])) > 1 else None
    return None


def _validate_address(text: str) -> Optional[str]:
    if not 6 <= len(text) <= 120:
        return None
    if not any(ch.isdigit() for ch in text) or not any(ch.isalpha() for ch in text):
        return None
    return text.rstrip(",")


DETECTORS: Dict[str, Detector] = {
    d.name: d
    for d in (
        Detector("email", _EMAIL, _keep_if(v.email_valid), 0, _GATE_AT),
        Detector("url", _URL, _validate_url, 1, _GATE_URL),
        Detector("ipv6", _IPV6, _keep_if(v.ipv6_valid), 2, _GATE_COLON2),
        Detector("credit_card", _CARD, _keep_if(v.credit_card_valid), 3, _GATE_DIGITS4),
        Detector("aadhaar", _AADHAAR, _keep_if(v.aadhaar_valid), 4, _GATE_DIGITS4),
        Detector("pan", _PAN, _keep_if(v.pan_valid), 5, _GATE_PAN),
        Detector("phone", _PHONE, _keep_if(v.phone_valid), 6, _GATE_DIGITS3),
        Detector("ipv4", _IPV4, _keep_if(v.ipv4_valid), 7, _GATE_DOT_DIGIT),
        Detector("date_of_birth", _DATE, _validate_date, 8, _GATE_YEAR, _dob_context),
        Detector("address", _ADDRESS, _validate_address, 9, _GATE_ADDRESS),
        Detector("postal_code", _POSTAL, _validate_postal, 10, _GATE_DIGITS3, _postal_context),
    )
}


# --------------------------------------------------------------------------- finding


def find_matches(
    blob: str,
    types: Optional[Iterable[str]] = None,
    covers_cell: Optional[Callable[[int, int], bool]] = None,
) -> List[Match]:
    """Every validated hit of the requested pattern types in ``blob`` (may overlap).

    ``covers_cell(start, end)`` tells whether a span is a whole cell value; a date or postal
    code that fills its cell needs no cue word, one buried in text does.
    """
    names = [t for t in PATTERN_TYPES if types is None or t in types]
    out: List[Match] = []
    for name in names:
        det = DETECTORS[name]
        if det.gate is not None and det.gate.search(blob) is None:
            continue
        for m in det.pattern.finditer(blob):
            cleaned = det.validate(m.group(0))
            if cleaned is None:
                continue
            start = m.start()
            end = start + len(cleaned)
            if det.context is not None and not det.context(blob, start, end, cleaned):
                if covers_cell is None or not covers_cell(start, end):
                    continue
            out.append(Match(name, start, end, cleaned))
    return out


def resolve(matches: Sequence[Match]) -> List[Match]:
    """Drop overlaps: the earliest span wins, then the longest, then the higher-priority type."""
    ordered = sorted(matches, key=lambda m: (m.start, -(m.end - m.start), DETECTORS[m.type].priority))
    kept: List[Match] = []
    last_end = -1
    for m in ordered:
        if m.start >= last_end:
            kept.append(m)
            last_end = m.end
    return kept


def covers_all_of(text: str) -> Callable[[int, int], bool]:
    """A ``covers_cell`` callback for ``text``: true for a span that fills it exactly."""
    length = len(text)

    def covers_cell(start: int, end: int) -> bool:
        return start == 0 and end == length

    return covers_cell


def find_in_text(
    text: str,
    types: Optional[Iterable[str]] = None,
    whole_value: bool = True,
) -> List[Match]:
    """Non-overlapping validated hits in a piece of free text, sorted by position.

    ``whole_value`` (the default) lets a match filling the whole string satisfy a context
    gate, exactly as a whole cell value does. Scanning and masking must agree here:
    anything the scan can flag, the mask has to be able to replace.
    """
    covers = covers_all_of(text) if whole_value else None
    return resolve(find_matches(text, types, covers))
