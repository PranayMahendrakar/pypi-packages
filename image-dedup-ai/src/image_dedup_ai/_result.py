"""Result objects: plain dataclasses with ``.summary()`` (text) and ``.to_dict()`` (JSON-safe)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Tuple

_HASH_LIMIT_NOTE = (
    "Perceptual hashes do not match rotated, flipped or heavily cropped copies; "
    "pass embed= to Index and use method='embed' to catch those."
)


def format_bytes(n: int) -> str:
    """Human-readable byte count in ASCII, e.g. ``'3.4 MB'``."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"  # pragma: no cover - loop always returns


def _plural(n: int, word: str, plural: str = "") -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def _describe(m: "Member") -> str:
    size = f"{m.width}x{m.height}"
    return f"{size}, {format_bytes(m.bytes)}" if m.bytes else size


class Match(NamedTuple):
    """One neighbour returned by ``Index.near``: an indexed path and its similarity (0-1)."""

    path: str
    similarity: float


@dataclass
class Member:
    """One image in a duplicate group, described relative to the copy that is kept."""

    path: str
    width: int
    height: int
    bytes: int
    similarity: float
    identical: bool
    kept: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return {
            "path": self.path,
            "width": int(self.width),
            "height": int(self.height),
            "bytes": int(self.bytes),
            "similarity_to_keep": round(float(self.similarity), 6),
            "identical_to_keep": bool(self.identical),
            "kept": bool(self.kept),
        }


@dataclass
class AddReport:
    """What one ``Index.add`` call did."""

    hashed: int = 0
    unchanged: int = 0
    embedded: int = 0
    removed: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)

    @property
    def indexed(self) -> int:
        """Images from this call that are now in the index (what ``add`` returns)."""
        return self.hashed + self.unchanged

    def summary(self) -> str:
        """One line of plain text."""
        parts = [f"indexed {_plural(self.indexed, 'image')}: {self.hashed:,} hashed, {self.unchanged:,} unchanged"]
        if self.embedded:
            parts.append(f"{self.embedded:,} embedded")
        if self.removed:
            parts.append(f"{self.removed:,} gone from disk and removed")
        if self.skipped:
            parts.append(f"{len(self.skipped):,} skipped (unreadable or not images)")
        if self.ignored:
            parts.append(f"{_plural(len(self.ignored), 'non-image file')} ignored")
        return "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return {
            "indexed": self.indexed,
            "hashed": self.hashed,
            "unchanged": self.unchanged,
            "embedded": self.embedded,
            "removed": self.removed,
            "skipped": [{"path": p, "reason": r} for p, r in self.skipped],
            "ignored": list(self.ignored),
        }


@dataclass
class DedupeResult:
    """Duplicate groups found in an index, and what keeping one image per group would free.

    ``groups`` lists each group's paths with the kept image first; ``keep`` is that
    first path of every group; ``drop`` is every other member (the copies).
    """

    groups: List[List[str]]
    pairs: List[Tuple[str, str, float]]
    keep: List[str]
    wasted_bytes: int
    method: str
    threshold: float
    n_images: int
    bits: int = 0
    details: List[List[Member]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def drop(self) -> List[str]:
        """Every non-kept member of every group: the copies you could delete."""
        return [p for g in self.groups for p in g[1:]]

    @property
    def n_groups(self) -> int:
        """Number of duplicate groups."""
        return len(self.groups)

    @property
    def n_duplicates(self) -> int:
        """Number of images that are a copy of a kept image."""
        return sum(len(g) - 1 for g in self.groups)

    def __repr__(self) -> str:
        return (
            f"DedupeResult(method={self.method!r}, threshold={self.threshold}, n_images={self.n_images}, "
            f"groups={self.n_groups}, duplicates={self.n_duplicates}, wasted_bytes={self.wasted_bytes})"
        )

    def summary(self, max_groups: int = 10) -> str:
        """Human-readable report: headline counts, the largest groups, skipped files, caveats."""
        how = f"{self.method} ({self.bits}-bit)" if self.bits else self.method
        lines = [f"Compared {_plural(self.n_images, 'image')} with {how}, similarity >= {self.threshold:g}."]
        if self.n_images == 0:
            lines.append("No images indexed, so there is nothing to compare.")
        elif not self.groups:
            lines.append("No duplicates found.")
        else:
            identical = sum(1 for g in self.details for m in g[1:] if m.identical)
            near = self.n_duplicates - identical
            members = sum(len(g) for g in self.groups)
            freeing = f", freeing {format_bytes(self.wasted_bytes)}" if self.wasted_bytes else ""
            lines.append(
                f"{_plural(self.n_groups, 'duplicate group')} covering {members:,} images: "
                f"keep {self.n_groups:,}, {_plural(self.n_duplicates, 'copy', 'copies')} could go{freeing}."
            )
            lines.append(
                f"Of the copies, {identical:,} {'is' if identical == 1 else 'are'} identical in content and "
                f"{near:,} {'is a near-duplicate' if near == 1 else 'are near-duplicates'} "
                "(resized, re-compressed or lightly edited)."
            )
            for number, group in enumerate(self.details[:max_groups], start=1):
                kept = group[0]
                lines.append("")
                lines.append(f"Group {number}: keep {kept.path} ({_describe(kept)})")
                for m in group[1:]:
                    tag = "identical content" if m.identical else f"similarity {m.similarity:.3f}"
                    lines.append(f"  copy {m.path} ({_describe(m)}), {tag}")
            if self.n_groups > max_groups:
                lines.append("")
                lines.append(f"... and {self.n_groups - max_groups:,} more groups (see .groups).")
        if self.skipped:
            lines.append("")
            lines.append(f"{_plural(len(self.skipped), 'file')} skipped (unreadable or not images):")
            for path, reason in self.skipped[:5]:
                lines.append(f"  {path}: {reason}")
            if len(self.skipped) > 5:
                lines.append(f"  ... and {len(self.skipped) - 5:,} more (see .skipped).")
        if self.notes:
            lines.append("")
            lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything in the result."""
        return {
            "method": self.method,
            "threshold": float(self.threshold),
            "bits": int(self.bits),
            "n_images": int(self.n_images),
            "n_groups": self.n_groups,
            "n_duplicates": self.n_duplicates,
            "wasted_bytes": int(self.wasted_bytes),
            "groups": [list(g) for g in self.groups],
            "keep": list(self.keep),
            "drop": self.drop,
            "pairs": [[a, b, round(float(s), 6)] for a, b, s in self.pairs],
            "details": [[m.to_dict() for m in g] for g in self.details],
            "skipped": [{"path": p, "reason": r} for p, r in self.skipped],
            "notes": list(self.notes),
        }


__all__ = ["AddReport", "DedupeResult", "Match", "Member", "format_bytes"]
