"""Operating limits you set yourself, and the violations they produce.

A :class:`Rule` is one limit on one channel. Hard limits (``min`` / ``max``) are
breaches; soft limits (``warn_min`` / ``warn_max``) are early warnings. Rules are
the only part of the score that encodes what *you* consider acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._util import describe_names, fmt, jsonable

#: How much of a rule's compliance is lost per fraction of rows breaching a hard limit.
HARD_WEIGHT = 1.0
#: The same for rows that are only inside the warning band.
WARN_WEIGHT = 0.4
#: A single breached hard limit costs at least this many of the rule's 100 points.
MIN_HARD_DEFICIT = 25.0
#: A single warning costs at least this many.
MIN_WARN_DEFICIT = 10.0

_LIMIT_FIELDS = ("min", "max", "warn_min", "warn_max")


@dataclass(frozen=True)
class Violation:
    """One limit, broken by some rows of the scored window."""

    channel: str
    limit: str  # "min", "max", "warn_min" or "warn_max"
    value: float  # the limit that was set
    severity: str  # "critical" for min/max, "warning" for warn_min/warn_max
    count: int  # rows that broke it
    n_rows: int  # rows with a usable reading on that channel
    fraction: float  # count / n_rows
    worst: float  # the most extreme reading seen
    first_time: Any = None  # when it first happened, if a time column was given

    @property
    def message(self) -> str:
        """One plain-ASCII line describing the violation."""
        direction = "below" if self.limit in ("min", "warn_min") else "above"
        when = f", first at {self.first_time}" if self.first_time is not None else ""
        return (
            f"{self.channel} {direction} {self.limit}={fmt(self.value)} "
            f"on {self.count} of {self.n_rows} rows ({self.fraction * 100:.1f}%), "
            f"worst {fmt(self.worst)}{when}"
        )

    def summary(self) -> str:
        """Human-readable text, severity first."""
        return f"[{self.severity}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        return jsonable(
            {
                "channel": self.channel,
                "limit": self.limit,
                "value": self.value,
                "severity": self.severity,
                "count": self.count,
                "n_rows": self.n_rows,
                "fraction": self.fraction,
                "worst": self.worst,
                "first_time": self.first_time,
                "message": self.message,
            }
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


@dataclass(frozen=True)
class Rule:
    """A limit on one channel.

    ``min`` / ``max`` are hard limits; breaking one is critical. ``warn_min`` /
    ``warn_max`` are soft limits that fire before the hard one. ``weight`` decides
    how much this rule counts relative to the others inside the compliance
    component (it does not change the weight of compliance itself).
    """

    channel: str
    min: Optional[float] = None
    max: Optional[float] = None
    warn_min: Optional[float] = None
    warn_max: Optional[float] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError(f"Rule.channel must be a non-empty column name, got {self.channel!r}")
        for name in _LIMIT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Rule({self.channel!r}): {name} must be a number or None, got {value!r}"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(f"Rule({self.channel!r}): {name} must be finite, got {value!r}")
            object.__setattr__(self, name, number)
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Rule({self.channel!r}): weight must be a number, got {self.weight!r}"
            ) from exc
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"Rule({self.channel!r}): weight must be a positive number, got {self.weight!r}"
            )
        object.__setattr__(self, "weight", weight)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"Rule({self.channel!r}): min={fmt(self.min)} is above max={fmt(self.max)}"
            )
        if all(getattr(self, name) is None for name in _LIMIT_FIELDS):
            raise ValueError(
                f"Rule({self.channel!r}) sets no limit; give at least one of "
                "min, max, warn_min or warn_max"
            )

    @property
    def limits(self) -> Dict[str, float]:
        """The limits that were actually set, as ``{'max': 80.0}``."""
        return {name: getattr(self, name) for name in _LIMIT_FIELDS if getattr(self, name) is not None}

    def describe(self) -> str:
        """'temp: max=80, warn_max=75' - plain ASCII."""
        parts = [f"{name}={fmt(value)}" for name, value in self.limits.items()]
        tail = f", weight={fmt(self.weight)}" if self.weight != 1.0 else ""
        return f"{self.channel}: {', '.join(parts)}{tail}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        data: Dict[str, Any] = {"channel": self.channel, "weight": self.weight}
        data.update(self.limits)
        return jsonable(data)


def normalize_rules(rules: Any) -> List[Rule]:
    """Accept the several shapes a user may pass and return a list of Rule.

    Supported: ``None``, a single Rule, a list of Rule or of dicts, and a mapping
    ``{"temp": {"max": 80}}``. A mapping value may also be a ``(min, max)`` pair.
    """
    if rules is None:
        return []
    if isinstance(rules, Rule):
        return [rules]
    out: List[Rule] = []
    if isinstance(rules, Mapping):
        for channel, spec in rules.items():
            out.append(_rule_from_spec(str(channel), spec))
        return out
    if isinstance(rules, (str, bytes)):
        raise TypeError(
            "rules must be a list of Rule, a list of dicts, or a mapping "
            "{channel: {'min': .., 'max': ..}}, not a string"
        )
    if isinstance(rules, Iterable):
        for item in rules:
            if isinstance(item, Rule):
                out.append(item)
            elif isinstance(item, Mapping):
                spec = dict(item)
                channel = spec.pop("channel", None)
                if channel is None:
                    raise ValueError(
                        f"rule {item!r} has no 'channel' key; every rule names one channel"
                    )
                out.append(_rule_from_spec(str(channel), spec))
            else:
                raise TypeError(
                    f"rules may contain Rule objects or dicts, not {type(item).__name__}"
                )
        return out
    raise TypeError(
        f"rules must be a Rule, a list of rules or a mapping, not {type(rules).__name__}"
    )


def _rule_from_spec(channel: str, spec: Any) -> Rule:
    if isinstance(spec, Rule):
        if spec.channel != channel:
            raise ValueError(
                f"rule for {channel!r} carries a different channel name {spec.channel!r}"
            )
        return spec
    if isinstance(spec, Mapping):
        unknown = [k for k in spec if str(k) not in _LIMIT_FIELDS + ("weight",)]
        if unknown:
            raise ValueError(
                f"rule for {channel!r} has unknown key(s) {describe_names(unknown)}; "
                f"valid keys: {', '.join(_LIMIT_FIELDS)}, weight"
            )
        return Rule(channel=channel, **{str(k): v for k, v in spec.items()})
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return Rule(channel=channel, min=spec[0], max=spec[1])
    raise TypeError(
        f"rule for {channel!r} must be a dict like {{'min': .., 'max': ..}} or a (min, max) "
        f"pair, not {type(spec).__name__}"
    )


def check_rule_channels(rules: Sequence[Rule], available: Sequence[str]) -> None:
    """Raise a clear ValueError when a rule names a channel that is not present."""
    known = set(available)
    missing = []
    for rule in rules:
        if rule.channel not in known and rule.channel not in missing:
            missing.append(rule.channel)
    if missing:
        raise ValueError(
            f"rule(s) name channel(s) not present in the data: {describe_names(missing)}; "
            f"available channels: {describe_names(list(available))}"
        )


def _rule_masks(
    rule: Rule, usable: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, float, np.ndarray]]]:
    """(hard breaches, warning-only breaches, one mask per limit that was set)."""
    n_rows = int(usable.size)
    hard_mask = np.zeros(n_rows, dtype=bool)
    warn_mask = np.zeros(n_rows, dtype=bool)
    breaks: List[Tuple[str, float, np.ndarray]] = []
    if rule.min is not None:
        mask = usable < rule.min
        hard_mask |= mask
        breaks.append(("min", rule.min, mask))
    if rule.max is not None:
        mask = usable > rule.max
        hard_mask |= mask
        breaks.append(("max", rule.max, mask))
    if rule.warn_min is not None:
        mask = (usable < rule.warn_min) & ~hard_mask
        warn_mask |= mask
        breaks.append(("warn_min", rule.warn_min, mask))
    if rule.warn_max is not None:
        mask = (usable > rule.warn_max) & ~hard_mask
        warn_mask |= mask
        breaks.append(("warn_max", rule.warn_max, mask))
    return hard_mask, warn_mask, breaks


def breach_fraction(rule: Rule, values: np.ndarray) -> Optional[float]:
    """Share of usable readings breaking this rule, warnings counted at ``WARN_WEIGHT``.

    Continuous and never floored, unlike :func:`evaluate_rule`. The trend compares
    two halves of one window, and a minimum deficit would let a single stray reading
    in one half look like a change of direction.
    """
    finite = np.isfinite(values)
    n_rows = int(finite.sum())
    if n_rows == 0:
        return None
    hard_mask, warn_mask, _ = _rule_masks(rule, values[finite])
    breached = HARD_WEIGHT * float(hard_mask.sum()) + WARN_WEIGHT * float(warn_mask.sum())
    return breached / n_rows


def evaluate_rule(
    rule: Rule,
    values: np.ndarray,
    times: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[float], List[Violation]]:
    """Score one rule against one channel's values.

    Returns ``(deficit, violations)`` where the deficit is 0-100 (0 means fully
    compliant) or ``None`` when the channel has no usable reading in the window.
    """
    finite = np.isfinite(values)
    n_rows = int(finite.sum())
    if n_rows == 0:
        return None, []
    usable = values[finite]
    positions = np.flatnonzero(finite)
    hard_mask, warn_mask, breaks = _rule_masks(rule, usable)

    violations: List[Violation] = []
    for limit, value, mask in breaks:
        count = int(mask.sum())
        if not count:
            continue
        hit = usable[mask]
        worst = float(hit.min()) if limit in ("min", "warn_min") else float(hit.max())
        first_time = None
        if times is not None:
            first_position = int(positions[np.flatnonzero(mask)[0]])
            if 0 <= first_position < len(times):
                first_time = times[first_position]
        violations.append(
            Violation(
                channel=rule.channel,
                limit=limit,
                value=float(value),
                severity="critical" if limit in ("min", "max") else "warning",
                count=count,
                n_rows=n_rows,
                fraction=count / n_rows,
                worst=worst,
                first_time=first_time,
            )
        )

    hard_fraction = float(hard_mask.sum()) / n_rows
    warn_fraction = float(warn_mask.sum()) / n_rows
    deficit = 100.0 * (HARD_WEIGHT * hard_fraction + WARN_WEIGHT * warn_fraction)
    if hard_mask.any():
        deficit = max(deficit, MIN_HARD_DEFICIT)
    elif warn_mask.any():
        deficit = max(deficit, MIN_WARN_DEFICIT)
    return min(100.0, deficit), violations
