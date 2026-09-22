"""The result object returned by :meth:`prompt_cache.Cache.stats`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _count(number: int, singular: str, plural: Optional[str] = None) -> str:
    """``1 hit`` / ``2 hits``, with an explicit plural where English needs one."""
    if number == 1:
        return "1 " + singular
    return "{0} {1}".format(number, plural if plural is not None else singular + "s")


@dataclass(frozen=True)
class Stats:
    """What the cache has done so far, and what it is holding.

    ``hits``, ``misses``, ``evictions`` and ``saved_calls`` count what *this*
    :class:`~prompt_cache.Cache` object has seen since it was created; another
    process sharing the same file keeps its own counts. ``entries`` and
    ``size_mb`` are read from the database, so they describe everything stored
    in this cache's namespace no matter who put it there.
    """

    hits: int
    misses: int
    hit_rate: float
    entries: int
    size_mb: float
    evictions: int
    saved_calls: int

    @property
    def lookups(self) -> int:
        """How many times the cache was asked for something."""
        return self.hits + self.misses

    def summary(self) -> str:
        """Two plain-ASCII lines a human can read straight from a terminal."""
        if self.lookups == 0:
            head = "prompt-cache: no lookups yet"
        else:
            head = "prompt-cache: {hits}, {misses} out of {total} ({rate:.1f}% hit rate)".format(
                hits=_count(self.hits, "hit"),
                misses=_count(self.misses, "miss", "misses"),
                total=_count(self.lookups, "lookup"),
                rate=self.hit_rate * 100.0,
            )
        body = "  {entries} stored, {size:.2f} MB, {evictions}, {saved} avoided".format(
            entries=_count(self.entries, "entry", "entries"),
            size=self.size_mb,
            evictions=_count(self.evictions, "eviction"),
            saved=_count(self.saved_calls, "call"),
        )
        return head + "\n" + body

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of every field, plus ``lookups``."""
        data = asdict(self)
        data["lookups"] = self.lookups
        return data

    def __str__(self) -> str:
        return self.summary()


__all__ = ["Stats"]
