"""Turn a cached value into bytes and back, recording which format was used.

JSON is tried first and is used only when the round trip is type-faithful, so
what comes out of the cache is what went in. Anything else falls back to
pickle. The format is stored next to the blob, so a read never has to guess.
"""

from __future__ import annotations

import json
import math
import pickle
from typing import Any, Tuple

JSON = "json"
PICKLE = "pickle"

#: Protocol 5 is understood by every Python this package supports (3.9+).
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _json_faithful(value: Any, depth: int = 0) -> bool:
    """True when ``json.loads(json.dumps(value))`` gives back the same types.

    Tuples become lists and non-string dict keys become strings under JSON, so
    those go to pickle instead. ``NaN`` and the infinities are excluded too:
    they survive Python's JSON but are not valid JSON for anyone else.
    """
    if depth > 50:
        return False
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_faithful(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_faithful(item, depth + 1)
            for key, item in value.items()
        )
    return False


def encode(value: Any) -> Tuple[str, bytes]:
    """Return ``(format, blob)`` for ``value``.

    Raises:
        TypeError: if the value can be stored by neither JSON nor pickle. The
            message names the offending type.
    """
    if _json_faithful(value):
        try:
            return JSON, json.dumps(value, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            pass
    try:
        return PICKLE, pickle.dumps(value, protocol=PICKLE_PROTOCOL)
    except Exception as exc:  # pickle raises a wide family of errors
        raise TypeError(
            "prompt-cache cannot store a value of type {name!r}: it is not "
            "JSON-serialisable and pickle refused it ({exc}). Return a str, "
            "dict, list or another picklable object instead.".format(
                name=type(value).__name__, exc=exc
            )
        ) from None


def decode(fmt: str, blob: bytes) -> Any:
    """Rebuild a value stored by :func:`encode`.

    Raises:
        ValueError: if the recorded format is not one this version writes.
    """
    if fmt == JSON:
        return json.loads(blob.decode("utf-8"))
    if fmt == PICKLE:
        return pickle.loads(blob)
    raise ValueError("unknown cache entry format {0!r}".format(fmt))


__all__ = ["JSON", "PICKLE", "PICKLE_PROTOCOL", "decode", "encode"]
