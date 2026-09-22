"""The fault vocabulary: what can be wrong with a camera, and how badly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

OBSTRUCTION = "obstruction"
DEFOCUS = "defocus"
DARKNESS = "darkness"
OVEREXPOSURE = "overexposure"
FROZEN = "frozen"
TAMPERING = "tampering"
COLOUR_CAST = "colour_cast"
NOISE = "noise"
DRIFT = "drift"

FAULT_KINDS = (
    OBSTRUCTION,
    DEFOCUS,
    DARKNESS,
    OVEREXPOSURE,
    FROZEN,
    TAMPERING,
    COLOUR_CAST,
    NOISE,
    DRIFT,
)
"""Every kind a Fault can have, in the order summaries list them."""

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

SEVERITIES = (CRITICAL, WARNING, INFO)
"""Severities, worst first."""

_SEVERITY_RANK = {CRITICAL: 0, WARNING: 1, INFO: 2}

# Kind spellings people reach for, mapped to the canonical one.
_ALIASES = {
    "color_cast": COLOUR_CAST,
    "colour-cast": COLOUR_CAST,
    "color-cast": COLOUR_CAST,
    "cast": COLOUR_CAST,
    "blur": DEFOCUS,
    "blurry": DEFOCUS,
    "out_of_focus": DEFOCUS,
    "dark": DARKNESS,
    "underexposure": DARKNESS,
    "bright": OVEREXPOSURE,
    "blocked": OBSTRUCTION,
    "obstructed": OBSTRUCTION,
    "freeze": FROZEN,
    "frozen_feed": FROZEN,
    "tamper": TAMPERING,
}


def canonical_kind(kind: str) -> str:
    """Map a spelling of a fault kind onto the canonical one.

    Raises:
        ValueError: the name is not a fault kind at all.
    """
    name = str(kind).strip().lower().replace(" ", "_")
    name = _ALIASES.get(name, name)
    if name not in FAULT_KINDS:
        raise ValueError(
            "unknown fault kind {!r}; known kinds: {}".format(kind, ", ".join(FAULT_KINDS))
        )
    return name


@dataclass(frozen=True)
class Fault:
    """One thing that is wrong with a frame.

    Attributes:
        kind: one of FAULT_KINDS.
        severity: "critical", "warning" or "info".
        confidence: 0-1, how sure the check is that this is real.
        message: one plain sentence a person can act on.
    """

    kind: str
    severity: str
    confidence: float
    message: str

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_RANK:
            raise ValueError(
                "severity must be one of {}; got {!r}".format(", ".join(SEVERITIES), self.severity)
            )

    @property
    def is_critical(self) -> bool:
        """True when this fault alone makes the frame unusable."""
        return self.severity == CRITICAL

    @property
    def rank(self) -> int:
        """Sort key: 0 for critical, 1 for warning, 2 for info."""
        return _SEVERITY_RANK[self.severity]

    def line(self) -> str:
        """The one-line form used inside every summary."""
        return "  [{:<8}] {:<12} confidence {:.2f}  {}".format(
            self.severity, self.kind, self.confidence, self.message
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping of the fault."""
        return {
            "kind": self.kind,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 3),
            "message": self.message,
        }


def sort_faults(faults: Any) -> list:
    """Faults worst first, then most confident, then by kind - a stable order."""
    return sorted(faults, key=lambda f: (f.rank, -float(f.confidence), f.kind))
