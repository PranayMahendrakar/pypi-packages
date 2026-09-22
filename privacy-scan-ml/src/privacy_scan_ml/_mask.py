"""Masking: safe previews for reports, and the redact / hash / partial strategies."""
from __future__ import annotations

import hashlib
import secrets
from typing import Any, List, Optional

from ._detectors import STREET_WORDS, find_in_text

STRATEGIES = ("redact", "hash", "partial")
_KEEP = frozenset(" -.@+()/:,_")

#: Where a generated salt is recorded on a masked DataFrame or Series.
SALT_KEY = "privacy_scan_ml_salt"
#: Where anything the mask could not do is recorded on a masked DataFrame.
WARNINGS_KEY = "privacy_scan_ml_warnings"


def new_salt() -> str:
    """A fresh random salt, used when ``strategy="hash"`` is called without one."""
    return secrets.token_hex(16)


class MaskedText(str):
    """A masked string that also carries the ``salt`` used to hash it."""

    salt: Optional[str] = None


class MaskedList(list):
    """A masked list of strings that also carries the ``salt`` used to hash them."""

    salt: Optional[str] = None


class MaskedTuple(tuple):
    """A masked tuple of strings that also carries the ``salt`` used to hash them."""

    salt: Optional[str] = None


def attach_salt(result: Any, salt: Optional[str]) -> Any:
    """Record ``salt`` on a masked result so the caller can reproduce the mapping.

    A DataFrame or Series gets it in ``.attrs[SALT_KEY]``; a string, list or tuple comes
    back as a thin subclass carrying ``.salt``. Without a salt the result is untouched.
    """
    if salt is None:
        return result
    attrs = getattr(result, "attrs", None)
    if isinstance(attrs, dict):
        attrs[SALT_KEY] = salt
        return result
    if isinstance(result, str):
        result = MaskedText(result)
    elif isinstance(result, tuple):
        result = MaskedTuple(result)
    elif isinstance(result, list):
        result = MaskedList(result)
    else:
        return result
    result.salt = salt
    return result


def _stars(text: str, keep_tail: int = 0) -> str:
    """Replace letters and digits with ``*``; separators stay, the last ``keep_tail`` chars stay."""
    keep_tail = min(keep_tail, max(len(text) - 4, 0)) if keep_tail else 0
    cut = len(text) - keep_tail
    head = "".join("*" if ch.isalnum() else ch for ch in text[:cut])
    return head + text[cut:]


def _tail(text: str) -> int:
    """How many trailing characters a partial mask may reveal."""
    return 4 if len(text) >= 8 else len(text) // 2


def preview(value: str, pii_type: str) -> str:
    """A masked example for reports: shows the shape of the value, never the value."""
    text = str(value)
    if pii_type == "email" and "@" in text:
        local, domain = text.rsplit("@", 1)
        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        return local[:1] + "***@***" + ("." + tld if tld else "")
    if pii_type == "url":
        lowered = text.lower()
        if "://" in lowered:
            return text.split("://", 1)[0] + "://***"
        return "www.***" if lowered.startswith("www.") else "***"
    if pii_type == "ipv4":
        return text.split(".", 1)[0] + ".***.***.***"
    if pii_type == "ipv6":
        return text.split(":", 1)[0] + ":***"
    if pii_type in ("phone", "credit_card", "aadhaar", "pan"):
        return _stars(text, 4)
    if pii_type == "postal_code":
        return _stars(text, 2)
    if pii_type == "date_of_birth":
        return _stars(text)
    if pii_type == "address":
        parts = []
        for token in text.split(" "):
            core = token.strip(",.")
            parts.append(token if core.lower() in STREET_WORDS else _stars(token))
        return " ".join(parts)
    if pii_type == "person_name":
        return " ".join(tok[:1] + "*" * (len(tok) - 1) if len(tok) > 1 else "*" for tok in text.split())
    # sensitive categories and anything else: first letter only
    return text[:1] + "*" * max(0, len(text) - 1) if len(text) > 1 else "*"


def check_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    return strategy


def check_salt(salt: Optional[str]) -> Optional[str]:
    """Reject a non-string salt by name, rather than at the ``+`` inside the hash."""
    if salt is not None and not isinstance(salt, str):
        raise TypeError(f"salt must be a str (or None), got {type(salt).__name__}")
    return salt


def mask_value(value: str, pii_type: str, strategy: str = "redact", salt: Optional[str] = None) -> str:
    """Apply one strategy to one matched value."""
    text = str(value)
    if strategy == "redact":
        return "[" + pii_type.upper() + "]"
    if strategy == "hash":
        return hashlib.sha256(((salt or "") + text).encode("utf-8")).hexdigest()[:12]
    if strategy == "partial":
        return _stars(text, _tail(text))
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")


def mask_text(
    text: str,
    strategy: str = "redact",
    salt: Optional[str] = None,
    types: Optional[List[str]] = None,
    whole_value: bool = True,
) -> str:
    """Replace every validated hit inside a piece of free text.

    ``whole_value`` matches the scanner: a date or a PIN code that fills the whole value
    needs no cue word, so masking replaces exactly what the scan would flag.
    """
    matches = find_in_text(text, types, whole_value)
    if not matches:
        return text
    out: List[str] = []
    last = 0
    for m in matches:
        out.append(text[last : m.start])
        out.append(mask_value(m.text, m.type, strategy, salt))
        last = m.end
    out.append(text[last:])
    return "".join(out)
