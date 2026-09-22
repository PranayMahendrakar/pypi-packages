"""Validation behind every detector: a regex proposes a value, these functions confirm it."""
from __future__ import annotations

import calendar
import ipaddress
import re
import unicodedata
from datetime import date
from typing import Optional, Tuple
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- checksums

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(digits: str) -> bool:
    """True when the digit string (check digit last) passes the Verhoeff checksum."""
    if not digits or not digits.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][ord(ch) - 48]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    """The Verhoeff check digit to append to ``digits``."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][ord(ch) - 48]]
    return str(_VERHOEFF_INV[c])


def luhn_valid(digits: str) -> bool:
    """True when the digit string passes the Luhn (mod 10) checksum."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_check_digit(digits: str) -> str:
    """The Luhn check digit to append to ``digits``."""
    for d in "0123456789":
        if luhn_valid(digits + d):
            return d
    return "0"  # pragma: no cover - one digit always works


# --------------------------------------------------------------------------- helpers

_DIGITS = re.compile(r"\d")


def only_digits(text: str) -> str:
    """The digits of ``text`` in order, separators dropped."""
    return "".join(_DIGITS.findall(text))


def _monotone_digits(digits: str) -> bool:
    """All the same digit (0000000000) - never a real identifier."""
    return len(set(digits)) == 1


# --------------------------------------------------------------------------- identifiers


def aadhaar_valid(text: str) -> bool:
    """12 digits, first digit 2-9, Verhoeff checksum, separators allowed in groups of four."""
    digits = only_digits(text)
    if len(digits) != 12 or digits[0] in "01" or _monotone_digits(digits):
        return False
    return verhoeff_valid(digits)


_PAN_RE = re.compile(r"[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]")


def pan_valid(text: str) -> bool:
    """Indian PAN: AAAAA9999A with the fourth letter one of P/C/H/F/A/T/B/L/J/G."""
    return bool(_PAN_RE.fullmatch(text.upper())) and (text.isupper() or text.islower())


_CARD_FIRST = "23456"


def credit_card_valid(text: str) -> bool:
    """13-19 digits, a plausible issuer prefix (2-6) and a Luhn checksum."""
    digits = only_digits(text)
    if not 13 <= len(digits) <= 19 or digits[0] not in _CARD_FIRST or _monotone_digits(digits):
        return False
    return luhn_valid(digits)


_PHONE_SEP_RE = re.compile(r"[ .\-()]")


def phone_valid(text: str) -> bool:
    """International (+CC, 8-15 digits), Indian mobiles (10 digits from 6-9, optionally 0/91
    prefixed) and separator-formatted NANP numbers; separators may be space, dot or hyphen."""
    stripped = text.strip()
    digits = only_digits(stripped)
    if not digits or _monotone_digits(digits):
        return False
    if stripped.startswith("+"):
        return 8 <= len(digits) <= 15 and digits[0] != "0"
    n = len(digits)
    formatted = bool(_PHONE_SEP_RE.search(stripped))
    if n == 10:
        if digits[0] in "6789":
            return True
        # NANP needs separators to be trusted: 415-555-2671, (415) 555 2671
        return formatted and digits[0] in "23456789" and digits[3] in "23456789"
    if n == 11:
        if digits[0] == "0" and digits[1] in "6789":
            return True
        return formatted and digits[0] == "1" and digits[1] in "23456789" and digits[4] in "23456789"
    if n == 12:
        return digits[:2] == "91" and digits[2] in "6789"
    if n == 13:
        return digits[:3] == "091" and digits[3] in "6789"
    return False


# --------------------------------------------------------------------------- internet

# Unicode-aware on purpose: RFC 6531 allows a non-ASCII local part and IDN allows
# non-ASCII domain labels, so andre@example.fr and user@muenchen.de spelled with their
# real accents are real addresses, as is a Devanagari mailbox. Character categories, not
# a regex class: a Devanagari vowel sign is a combining mark, which ``\w`` does not match,
# so a class built on ``\w`` would silently drop half the scripts in the world.
_ASCII_TLD_RE = re.compile(r"[A-Za-z]{2,24}|xn--[A-Za-z0-9-]{2,59}", re.IGNORECASE)
_LOCAL_PUNCT = frozenset("._%+-")
_MAX_LABEL = 63
_MAX_TLD = 24


def _letter_or_digit(ch: str) -> bool:
    """A letter, a combining mark or a digit, in any script."""
    return unicodedata.category(ch)[0] in ("L", "M", "N")


def _letter(ch: str) -> bool:
    """A letter or a combining mark, in any script."""
    return unicodedata.category(ch)[0] in ("L", "M")


def _local_valid(local: str) -> bool:
    """The part before the ``@``: letters, digits and ``. _ % + -`` in any script."""
    return all(ch in _LOCAL_PUNCT or _letter_or_digit(ch) for ch in local)


def _label_valid(label: str) -> bool:
    """One dotted piece of a host name: letters, digits and inner hyphens, 1-63 chars."""
    if not 1 <= len(label) <= _MAX_LABEL or label[0] == "-" or label[-1] == "-":
        return False
    return all(ch == "-" or _letter_or_digit(ch) for ch in label)


def _tld_valid(tld: str) -> bool:
    """The last label: ASCII letters, a punycode ``xn--`` label, or letters in any script."""
    if _ASCII_TLD_RE.fullmatch(tld):
        return True
    return 2 <= len(tld) <= _MAX_TLD and all(_letter(ch) for ch in tld)


def _host_valid(host: str, need_dot: bool = True) -> bool:
    if not host or len(host) > 253:
        return False
    if host.lower() == "localhost":
        return True
    labels = host.split(".")
    if need_dot and len(labels) < 2:
        return False
    if not all(_label_valid(label) for label in labels):
        return False
    return _tld_valid(labels[-1]) or labels[-1].isdigit()


def email_valid(text: str) -> bool:
    """RFC-shaped address: local part up to 64 chars, dotted domain, alphabetic TLD.

    Non-ASCII local parts and IDN domains count (RFC 6531); an underscore is fine in the
    local part but never in a host label.
    """
    if text.count("@") != 1 or len(text) > 254:
        return False
    local, domain = text.split("@")
    if not 1 <= len(local) <= 64 or ".." in local or local[0] == "." or local[-1] == ".":
        return False
    if not _local_valid(local):
        return False
    return _host_valid(domain) and not domain.split(".")[-1].isdigit()


def ipv4_valid(text: str) -> bool:
    """Four dotted decimal octets, each 0-255, no leading zeros."""
    parts = text.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or len(part) > 3 or (len(part) > 1 and part[0] == "0"):
            return False
        if int(part) > 255:
            return False
    return True


def ipv6_valid(text: str) -> bool:
    """Anything the standard library accepts as an IPv6 address (needs two colons at least)."""
    if text.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return True


_URL_TRAIL = ".,;:!?)]}" + "'" + '"'


def url_valid(text: str) -> Tuple[bool, str]:
    """Return ``(ok, cleaned)``: trailing sentence punctuation is dropped before checking."""
    cleaned = text.rstrip(_URL_TRAIL)
    probe = "http://" + cleaned if cleaned.lower().startswith("www.") else cleaned
    try:
        parts = urlsplit(probe)
    except ValueError:
        return False, cleaned
    if parts.scheme not in ("http", "https", "ftp"):
        return False, cleaned
    host = parts.hostname or ""
    if not _host_valid(host, need_dot=True) and not ipv4_valid(host):
        return False, cleaned
    return True, cleaned


# --------------------------------------------------------------------------- dates

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9


def _make_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_parts(y: str, m: str, d: str) -> Optional[date]:
    """Build a date from string parts; ``m`` may be a month name or abbreviation."""
    month = int(m) if m.isdigit() else MONTHS.get(m.lower().rstrip("."))
    if month is None:
        return None
    return _make_date(int(y), month, int(d))


def parse_ambiguous(a: str, b: str, y: str) -> Optional[date]:
    """dd/mm/yyyy or mm/dd/yyyy - whichever is a real date (day-first preferred)."""
    return parse_date_parts(y, b, a) or parse_date_parts(y, a, b)


def plausible_birth_date(value: date, today: Optional[date] = None) -> bool:
    """A real calendar date not in the future and not more than 120 years back."""
    today = today or date.today()
    return (today.year - 120) <= value.year and value <= today
