"""Prompt normalisation and cache-key hashing.

The database never stores a prompt as an index: a key is always the hex
SHA-256 digest of a canonical byte string, so a one-megabyte prompt and a
three-word prompt cost exactly the same 64 characters to look up.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

#: Bumped if the key recipe ever changes, so old entries simply stop matching
#: instead of being read with the wrong meaning.
KEY_VERSION = "1"

#: How much of the prompt is kept next to the entry for humans to read.
PREVIEW_CHARS = 200


def normalize_prompt(prompt: str) -> str:
    """Return ``prompt`` in the canonical form used for hashing.

    Exactly four steps, in this order:

    1. line endings ``\r\n`` and ``\r`` become ``\n``;
    2. trailing spaces and tabs are stripped from the end of every line;
    3. a run of two or more blank lines collapses to one blank line;
    4. leading and trailing blank lines are dropped.

    Nothing else is touched: case, inner spacing, punctuation, and every
    non-ASCII character are preserved exactly as written.
    """
    text = prompt.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    kept: list = []
    blanks = 0
    for line in lines:
        if line == "":
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        kept.append(line)
    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()
    return "\n".join(kept)


def canonical(value: Any, _depth: int = 0) -> Any:
    """Turn a parameter value into something JSON can render deterministically.

    Types JSON knows are kept as they are; everything else becomes a string
    that carries the type name, so ``1``, ``"1"`` and ``Decimal("1")`` stay
    three different cache keys.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if _depth > 20:
        return "<deep:%s>" % type(value).__name__
    if isinstance(value, (list, tuple)):
        return [canonical(item, _depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return ["<set>"] + sorted(repr(canonical(item, _depth + 1)) for item in value)
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k], _depth + 1) for k in sorted(value, key=repr)}
    return "<%s.%s:%r>" % (type(value).__module__, type(value).__name__, value)


def make_key(
    prompt: str,
    *,
    namespace: str,
    params: Optional[Mapping[str, Any]] = None,
    args: Sequence[Any] = (),
    function: Optional[str] = None,
    normalize: bool = True,
) -> str:
    """Hash everything that identifies one cached answer into a hex digest."""
    text = normalize_prompt(prompt) if normalize else prompt
    material: Dict[str, Any] = {
        "v": KEY_VERSION,
        "ns": namespace,
        "prompt": text,
        "args": [canonical(a) for a in args],
        "params": {str(k): canonical(v) for k, v in sorted((params or {}).items())},
    }
    if function:
        material["fn"] = function
    payload = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preview(prompt: str, limit: int = PREVIEW_CHARS) -> str:
    """A short, single-line excerpt of a prompt, for humans reading the CLI."""
    flat = " ".join(prompt.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


__all__ = ["KEY_VERSION", "PREVIEW_CHARS", "canonical", "make_key", "normalize_prompt", "preview"]
