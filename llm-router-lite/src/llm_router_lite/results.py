"""The objects the router hands back. Every one of them explains itself.

Each result is a frozen dataclass with a ``summary()`` (one or two lines of
plain ASCII) and a ``to_dict()`` (JSON-safe, nothing but built-in types).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .complexity import Complexity
from .models import round_cost


@dataclass(frozen=True)
class Candidate:
    """One model considered for a prompt, with the numbers it was judged on."""

    name: str
    score: float
    estimated_cost: float
    quality: float
    latency_ms: Optional[float]
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict."""
        return {
            "name": self.name,
            "score": self.score,
            "estimated_cost": self.estimated_cost,
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class Route:
    """Which model should answer this prompt, and why.

    Nothing has been called at this point: a route is a decision, not a call.

    ``skipped`` and ``demoted`` are different verdicts. A skipped model is out
    of the running altogether - wrong tags, too small for the prompt, over
    ``max_cost`` - and cannot be a fallback. A demoted one is only ranked
    behind: it is rated under the quality floor for this band, but a weak
    answer beats no answer, so it still stands in the fallback chain.
    """

    model: str
    reason: str
    estimated_cost: float
    alternatives: Tuple[str, ...]
    complexity: Complexity
    prefer: str
    estimated_tokens: int
    considered: Tuple[Candidate, ...]
    skipped: Dict[str, str]
    demoted: Dict[str, str] = field(default_factory=dict)

    @property
    def fallbacks(self) -> Tuple[str, ...]:
        """Another name for ``alternatives``, in the order they will be tried."""
        return self.alternatives

    def cost_of(self, model: str) -> float:
        """The estimated cost of one call to ``model`` for this prompt."""
        for candidate in self.considered:
            if candidate.name == model:
                return candidate.estimated_cost
        raise KeyError(
            "{0!r} was not considered for this prompt; considered: {1}".format(
                model, ", ".join(candidate.name for candidate in self.considered)
            )
        )

    def summary(self) -> str:
        """One human line: the pick, the cost and the fallbacks."""
        line = "route -> {0} (about {1:g} per call, {2} prompt): {3}".format(
            self.model,
            self.estimated_cost,
            self.complexity.band,
            self.reason,
        )
        return line

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the whole decision."""
        return {
            "model": self.model,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
            "alternatives": list(self.alternatives),
            "prefer": self.prefer,
            "estimated_tokens": self.estimated_tokens,
            "complexity": self.complexity.to_dict(),
            "considered": [candidate.to_dict() for candidate in self.considered],
            "skipped": dict(self.skipped),
            "demoted": dict(self.demoted),
        }

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class Attempt:
    """One call that was actually made: which model, did it work, how long."""

    model: str
    ok: bool
    error: Optional[str]
    ms: float

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict."""
        return {
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "ms": self.ms,
        }

    def summary(self) -> str:
        """One human line."""
        if self.ok:
            return "{0}: ok in {1:.1f} ms".format(self.model, self.ms)
        return "{0}: failed in {1:.1f} ms - {2}".format(self.model, self.ms, self.error)

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class Completion:
    """An answer, the model that gave it, and everything that was tried."""

    text: str
    model: str
    attempts: Tuple[Attempt, ...]
    cost: float
    route: Route

    @property
    def ok(self) -> bool:
        """True whenever there is an answer, which for a Completion is always."""
        return True

    @property
    def fallbacks_used(self) -> int:
        """How many models failed before this answer arrived."""
        return max(0, len(self.attempts) - 1)

    @property
    def ms(self) -> float:
        """Wall-clock time of every attempt, including the failed ones."""
        return round(sum(attempt.ms for attempt in self.attempts), 3)

    def summary(self) -> str:
        """Two plain-ASCII lines: what answered, and what it took to get there."""
        head = "{0} answered in {1:.1f} ms for about {2:g}".format(
            self.model, self.ms, self.cost
        )
        if self.fallbacks_used:
            failed = ", ".join(
                attempt.model for attempt in self.attempts if not attempt.ok
            )
            head += " after {0} failed".format(failed)
        return head + "\n  " + "\n  ".join(attempt.summary() for attempt in self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the answer, the attempts and the route."""
        return {
            "text": self.text,
            "model": self.model,
            "cost": self.cost,
            "ms": self.ms,
            "fallbacks_used": self.fallbacks_used,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "route": self.route.to_dict(),
        }

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class ModelStats:
    """What one model has done since the router was built."""

    name: str
    calls: int
    failures: int
    total_cost: float
    mean_latency_ms: float

    @property
    def successes(self) -> int:
        """Calls that returned an answer."""
        return self.calls - self.failures

    @property
    def failure_rate(self) -> float:
        """Share of calls that raised, between 0 and 1."""
        return round(self.failures / float(self.calls), 4) if self.calls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict."""
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "successes": self.successes,
            "failure_rate": self.failure_rate,
            "total_cost": self.total_cost,
            "mean_latency_ms": self.mean_latency_ms,
        }

    def summary(self) -> str:
        """One human line."""
        return (
            "{0}: {1} call(s), {2} failed, cost {3:g}, mean {4:.1f} ms".format(
                self.name, self.calls, self.failures, self.total_cost, self.mean_latency_ms
            )
        )

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class Stats:
    """Per-model counters for the whole router."""

    per_model: Dict[str, ModelStats]

    @property
    def calls(self) -> int:
        """Every handler call made, successful or not."""
        return sum(item.calls for item in self.per_model.values())

    @property
    def failures(self) -> int:
        """Every handler call that raised or returned something unusable."""
        return sum(item.failures for item in self.per_model.values())

    @property
    def total_cost(self) -> float:
        """Estimated cost of every call that succeeded."""
        return round_cost(sum(item.total_cost for item in self.per_model.values()))

    def __getitem__(self, name: str) -> ModelStats:
        return self.per_model[name]

    def __iter__(self):
        return iter(self.per_model.values())

    def __len__(self) -> int:
        return len(self.per_model)

    def summary(self) -> str:
        """A short table, one line per model."""
        head = "{0} call(s) across {1} model(s), {2} failed, cost {3:g}".format(
            self.calls, len(self.per_model), self.failures, self.total_cost
        )
        lines: List[str] = [head]
        for item in self.per_model.values():
            lines.append("  " + item.summary())
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict."""
        return {
            "calls": self.calls,
            "failures": self.failures,
            "total_cost": self.total_cost,
            "models": {
                name: item.to_dict() for name, item in self.per_model.items()
            },
        }

    def __str__(self) -> str:
        return self.summary()


class AllModelsFailed(RuntimeError):
    """Every model in the fallback chain failed.

    The message names the last error. ``attempts`` holds one
    :class:`Attempt` per model tried, in order, and ``route`` is the decision
    that produced the chain. The original exception is chained as ``__cause__``.
    """

    def __init__(self, attempts: Tuple[Attempt, ...], route: Route) -> None:
        self.attempts = tuple(attempts)
        self.route = route
        self.last_error = self.attempts[-1].error if self.attempts else None
        self.last_model = self.attempts[-1].model if self.attempts else None
        tried = ", ".join(attempt.model for attempt in self.attempts)
        super().__init__(
            "all {0} model(s) failed ({1}); last error from {2}: {3}".format(
                len(self.attempts), tried, self.last_model, self.last_error
            )
        )

    def summary(self) -> str:
        """The message plus one line per attempt."""
        return str(self) + "\n  " + "\n  ".join(
            attempt.summary() for attempt in self.attempts
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the failure."""
        return {
            "error": str(self),
            "last_model": self.last_model,
            "last_error": self.last_error,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "route": self.route.to_dict(),
        }
