"""The router: pick a model for a prompt, then call it with fallbacks.

Routing is a pure comparison of the numbers you registered. It never touches
the network, never imports a provider SDK and never calls a handler - only
:meth:`Router.complete` calls anything, and only the handler you supplied.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .complexity import Complexity, complexity, estimate_tokens
from .models import DEFAULT_LATENCY_MS, Model, make_model, normalize_models, round_cost
from .results import (
    AllModelsFailed,
    Attempt,
    Candidate,
    Completion,
    ModelStats,
    Route,
    Stats,
)

logger = logging.getLogger(__name__)

#: The four things you can ask the router to optimise for.
PREFERENCES = ("cheap", "fast", "quality", "balanced")

#: How much each preference weighs cost, speed and quality.
PREFERENCE_WEIGHTS = {
    "cheap": {"cost": 0.75, "speed": 0.10, "quality": 0.15},
    "fast": {"cost": 0.10, "speed": 0.75, "quality": 0.15},
    "quality": {"cost": 0.05, "speed": 0.05, "quality": 0.90},
}

#: "balanced" reads the prompt first: an easy prompt leans on cost, a hard one
#: leans on quality.
BALANCED_WEIGHTS = {
    "simple": {"cost": 0.60, "speed": 0.20, "quality": 0.20},
    "moderate": {"cost": 0.35, "speed": 0.15, "quality": 0.50},
    "hard": {"cost": 0.15, "speed": 0.10, "quality": 0.75},
}

#: Quality a model should have before it is handed a prompt of each band. If
#: nothing clears the floor the best available model is used anyway, and the
#: route says so.
QUALITY_FLOOR = {"simple": 0.0, "moderate": 0.35, "hard": 0.60}

_RESERVED_KEYWORDS = ("prefer", "require", "max_cost", "route")


def _normalize_require(require: Any) -> Tuple[str, ...]:
    if require is None:
        return ()
    if isinstance(require, str):
        require = (require,)
    if isinstance(require, Mapping) or not isinstance(require, Iterable):
        raise TypeError(
            "require must be a string or a sequence of strings, got {0}".format(
                type(require).__name__
            )
        )
    tags: List[str] = []
    for tag in require:
        if not isinstance(tag, str):
            raise TypeError(
                "every required tag must be a string, got {0}".format(
                    type(tag).__name__
                )
            )
        stripped = tag.strip()
        if stripped and stripped not in tags:
            tags.append(stripped)
    return tuple(tags)


def _ratio(value: float, best: float, floor: float = 0.0) -> float:
    """How ``value`` compares with the best (lowest) in the field, as a proportion.

    Cost and latency are ratio quantities: a model that costs twice as much is
    twice as expensive no matter who else is registered. Scoring them as
    ``best / value`` keeps that true - the cheapest scores 1.0, one that costs
    ten times more scores 0.1 - so registering a third, pricier model cannot
    change which of the other two wins.

    Stretching the field between its own cheapest and dearest instead (a
    min-max normalisation) does not have that property: one expensive outlier
    squashes every cheaper model into a narrow band near 1.0 where the cost
    term can no longer tell them apart, and ``prefer="cheap"`` stops picking
    the cheapest. Free is treated as infinitely cheap: when the best of the
    field costs nothing, the ranking falls back to ``floor``: see below.
    """
    if value <= best:
        return 1.0
    if best <= 0.0:
        # A free model makes ``best`` zero. Returning 0.0 here scored EVERY priced
        # model identically, so the cost term stopped ordering them at all and the
        # tie was broken by quality instead - registering one free model flipped
        # prefer="cheap" fallbacks to most-expensive-first, and put a 100x pricier
        # model ahead of a cheap one. Ranking against the cheapest model that does
        # have a price keeps relative cost meaningful: free still scores 1.0, and
        # everything else stays in price order.
        if floor <= 0.0:
            return 0.0
        return floor / (floor + value)
    return best / value


class _Counter:
    """Mutable per-model tally; turned into a frozen ModelStats on request."""

    __slots__ = ("calls", "failures", "total_cost", "total_ms")

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self.total_cost = 0.0
        self.total_ms = 0.0


class Router:
    """Hold a line-up of models and send each prompt to the right one.

    ``Router()`` starts empty; ``Router(models)`` accepts a mapping of
    ``name -> handler``, a mapping of ``name -> options``, or a sequence of
    :class:`~llm_router_lite.models.Model` objects.
    """

    def __init__(self, models: Any = None) -> None:
        self._models: Dict[str, Model] = {}
        self._counters: Dict[str, _Counter] = {}
        self._lock = threading.Lock()
        for model in normalize_models(models):
            self._install(model)

    # ------------------------------------------------------------------ setup

    def _install(self, model: Model) -> None:
        if model.name in self._models:
            raise ValueError(
                "a model named {0!r} is already registered; pick another name or "
                "call remove({0!r}) first".format(model.name)
            )
        self._models[model.name] = model
        self._counters[model.name] = _Counter()

    def add(
        self,
        name: str,
        handler: Callable[..., str],
        *,
        cost: float = 0.0,
        latency_ms: Optional[float] = None,
        quality: float = 0.5,
        max_tokens: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> "Router":
        """Register one model. Returns the router, so calls chain.

        ``handler`` is ``callable(prompt, **kw) -> str``. Nothing is called
        until :meth:`complete` runs; :meth:`route` only compares numbers.
        """
        self._install(
            make_model(
                name,
                handler,
                cost=cost,
                latency_ms=latency_ms,
                quality=quality,
                max_tokens=max_tokens,
                tags=tags,
            )
        )
        return self

    def remove(self, name: str) -> "Router":
        """Drop a model and its counters. Unknown names raise ``KeyError``."""
        if name not in self._models:
            raise KeyError(
                "no model named {0!r}; registered: {1}".format(
                    name, ", ".join(self._models) or "none"
                )
            )
        del self._models[name]
        del self._counters[name]
        return self

    @property
    def models(self) -> Tuple[Model, ...]:
        """Every registered model, in registration order."""
        return tuple(self._models.values())

    @property
    def names(self) -> Tuple[str, ...]:
        """Every registered model name, in registration order."""
        return tuple(self._models)

    def get(self, name: str) -> Model:
        """One registered model by name."""
        if name not in self._models:
            raise KeyError(
                "no model named {0!r}; registered: {1}".format(
                    name, ", ".join(self._models) or "none"
                )
            )
        return self._models[name]

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def __repr__(self) -> str:
        return "Router({0} model(s): {1})".format(
            len(self._models), ", ".join(self._models) or "none"
        )

    # ---------------------------------------------------------------- routing

    def _require_models(self) -> None:
        if not self._models:
            raise ValueError(
                "no models registered; call router.add(name, handler) or pass "
                "models= to Router() before routing"
            )

    def _weights(self, prefer: str, band: str) -> Dict[str, float]:
        if prefer == "balanced":
            return BALANCED_WEIGHTS[band]
        return PREFERENCE_WEIGHTS[prefer]

    def route(
        self,
        prompt: str,
        *,
        prefer: str = "balanced",
        require: Any = None,
        max_cost: Optional[float] = None,
    ) -> Route:
        """Decide which model should answer ``prompt``. Calls nothing.

        ``prefer`` is one of ``"cheap"``, ``"fast"``, ``"quality"`` or
        ``"balanced"``. ``require`` is a tag, or tags, a model must carry.
        ``max_cost`` is a ceiling on the estimated cost of one call.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string, got {0}".format(type(prompt).__name__)
            )
        self._require_models()
        if prefer not in PREFERENCES:
            raise ValueError(
                "prefer must be one of {0}, got {1!r}".format(
                    ", ".join(PREFERENCES), prefer
                )
            )
        if max_cost is not None:
            if isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)):
                raise TypeError(
                    "max_cost must be a number or None, got {0}".format(
                        type(max_cost).__name__
                    )
                )
            if max_cost < 0:
                raise ValueError("max_cost must be >= 0, got {0}".format(max_cost))
        required = _normalize_require(require)

        comp = complexity(prompt)
        prompt_tokens = estimate_tokens(prompt)
        skipped: Dict[str, str] = {}

        # 1. tags -------------------------------------------------------------
        pool = [model for model in self._models.values() if model.has_tags(required)]
        kept = {model.name for model in pool}
        for model in self._models.values():
            if model.name not in kept:
                skipped[model.name] = "does not carry the required tag(s) {0}".format(
                    ", ".join(required)
                )
        if not pool:
            known = sorted({tag for model in self._models.values() for tag in model.tags})
            raise ValueError(
                "no model carries the required tag(s) {0}; tags in this router: "
                "{1}".format(", ".join(required), ", ".join(known) or "none")
            )

        # 2. context window ---------------------------------------------------
        fitting = [model for model in pool if model.fits(prompt_tokens)]
        kept = {model.name for model in fitting}
        for model in pool:
            if model.name not in kept:
                skipped[model.name] = (
                    "holds {0} tokens, this prompt is about {1}".format(
                        model.max_tokens, prompt_tokens
                    )
                )
        if not fitting:
            largest = max(pool, key=lambda m: (m.max_tokens or 0, m.name))
            raise ValueError(
                "this prompt is about {0} tokens and no model can hold it; the "
                "largest is {1!r} at {2} tokens. Shorten the prompt or register a "
                "bigger model.".format(prompt_tokens, largest.name, largest.max_tokens)
            )

        # 3. cost ceiling -----------------------------------------------------
        costs = {model.name: model.estimated_cost(prompt_tokens) for model in fitting}
        if max_cost is None:
            affordable = list(fitting)
        else:
            affordable = [model for model in fitting if costs[model.name] <= max_cost]
            kept = {model.name for model in affordable}
            for model in fitting:
                if model.name not in kept:
                    skipped[model.name] = "costs about {0:g}, over max_cost {1:g}".format(
                        costs[model.name], max_cost
                    )
        if not affordable:
            cheapest = min(fitting, key=lambda m: (costs[m.name], m.name))
            raise ValueError(
                "no model fits max_cost={0:g} for this prompt (about {1} tokens); "
                "the cheapest is {2!r} at about {3:g} per call ({4:g} per 1000 "
                "tokens). Raise max_cost or register a cheaper model.".format(
                    max_cost,
                    prompt_tokens,
                    cheapest.name,
                    costs[cheapest.name],
                    cheapest.cost,
                )
            )

        # 4. quality floor for this band --------------------------------------
        # The floor demotes, it never discards: a model rated under the floor
        # still stands behind the pick as a fallback, because a weak answer
        # beats no answer when the good model is down. An explicit ``prefer``
        # is your stated intent, so the floor only applies to "balanced",
        # which has to guess.
        floor = QUALITY_FLOOR[comp.band] if prefer == "balanced" else 0.0
        qualified = list(affordable)
        above_floor = [model for model in qualified if model.quality >= floor]
        floor_relaxed = bool(floor) and not above_floor
        demoted: Dict[str, str] = {}
        if floor_relaxed:
            logger.warning(
                "no model reaches quality %.2f for a %s prompt; using the best of "
                "what is registered",
                floor,
                comp.band,
            )
        elif floor:
            for model in qualified:
                if model.quality < floor:
                    demoted[model.name] = (
                        "quality {0:.2f} is under the {1:.2f} wanted for a {2} "
                        "prompt, so it ranks behind as a fallback".format(
                            model.quality, floor, comp.band
                        )
                    )

        # 5. score -------------------------------------------------------------
        weights = self._weights(prefer, comp.band)
        known_latencies = [
            model.latency_ms for model in qualified if model.latency_ms is not None
        ]
        assumed_latency = (
            float(statistics.median(known_latencies))
            if known_latencies
            else DEFAULT_LATENCY_MS
        )
        latency = {
            model.name: (
                float(model.latency_ms)
                if model.latency_ms is not None
                else assumed_latency
            )
            for model in qualified
        }
        low_cost = min(costs[model.name] for model in qualified)
        low_ms = min(latency[model.name] for model in qualified)
        # the cheapest model that actually has a price, so a free model in the field
        # cannot collapse the cost ordering of everything else
        priced = [costs[model.name] for model in qualified if costs[model.name] > 0.0]
        cost_floor = min(priced) if priced else 0.0
        priced_ms = [latency[model.name] for model in qualified if latency[model.name] > 0.0]
        ms_floor = min(priced_ms) if priced_ms else 0.0

        scored: List[Tuple[Tuple[float, float, float, float, str], Candidate]] = []
        for model in qualified:
            cheapness = _ratio(costs[model.name], low_cost, cost_floor)
            speed = _ratio(latency[model.name], low_ms, ms_floor)
            score = round(
                weights["cost"] * cheapness
                + weights["speed"] * speed
                + weights["quality"] * model.quality,
                6,
            )
            candidate = Candidate(
                name=model.name,
                score=score,
                estimated_cost=costs[model.name],
                quality=model.quality,
                latency_ms=model.latency_ms,
                tags=model.tags,
            )
            sort_key = (
                0 if (floor_relaxed or model.quality >= floor) else 1,
                -score,
                costs[model.name],
                latency[model.name],
                -model.quality,
                model.name,
            )
            scored.append((sort_key, candidate))
        scored.sort(key=lambda pair: pair[0])
        ranked = tuple(candidate for _, candidate in scored)

        winner = ranked[0]
        alternatives = tuple(candidate.name for candidate in ranked[1:])
        reason = self._explain(
            winner=winner,
            ranked=ranked,
            comp=comp,
            prefer=prefer,
            weights=weights,
            required=required,
            max_cost=max_cost,
            floor=floor,
            floor_relaxed=floor_relaxed,
            skipped=skipped,
            demoted=demoted,
        )
        return Route(
            model=winner.name,
            reason=reason,
            estimated_cost=winner.estimated_cost,
            alternatives=alternatives,
            complexity=comp,
            prefer=prefer,
            estimated_tokens=prompt_tokens,
            considered=ranked,
            skipped=dict(skipped),
            demoted=dict(demoted),
        )

    def _explain(
        self,
        *,
        winner: Candidate,
        ranked: Tuple[Candidate, ...],
        comp: Complexity,
        prefer: str,
        weights: Dict[str, float],
        required: Tuple[str, ...],
        max_cost: Optional[float],
        floor: float,
        floor_relaxed: bool,
        skipped: Dict[str, str],
        demoted: Dict[str, str],
    ) -> str:
        """Build the plain-language ``Route.reason``."""
        leading = max(weights, key=lambda key: (weights[key], key))
        parts = [
            "{0!r} for a {1} prompt (complexity {2:.2f})".format(
                winner.name, comp.band, comp.score
            )
        ]
        if prefer == "balanced":
            parts.append(
                "balanced routing leans on {0} for a {1} prompt".format(
                    leading, comp.band
                )
            )
        else:
            parts.append("you asked for {0}".format(prefer))

        count = len(ranked)
        if count == 1:
            parts.append("it is the only model that qualifies")
        elif leading == "cost":
            cheapest = min(candidate.estimated_cost for candidate in ranked)
            if winner.estimated_cost <= cheapest:
                parts.append("it is the cheapest of the {0} that qualify".format(count))
            else:
                parts.append(
                    "it is not the cheapest, but it wins once speed and quality "
                    "are weighed in"
                )
        elif leading == "speed":
            fastest = min(
                candidate.latency_ms
                for candidate in ranked
                if candidate.latency_ms is not None
            ) if any(c.latency_ms is not None for c in ranked) else None
            if fastest is not None and winner.latency_ms == fastest:
                parts.append("it is the fastest of the {0} that qualify".format(count))
            else:
                parts.append(
                    "it wins the speed-weighted score across {0} models".format(count)
                )
        else:
            best_quality = max(candidate.quality for candidate in ranked)
            if winner.quality >= best_quality:
                parts.append(
                    "it has the best quality rating of the {0} that qualify".format(count)
                )
            else:
                parts.append(
                    "it wins the quality-weighted score across {0} models".format(count)
                )

        if required:
            parts.append("required tag(s): {0}".format(", ".join(required)))
        if max_cost is not None:
            parts.append("kept under max_cost {0:g}".format(max_cost))
        if floor_relaxed:
            parts.append(
                "no model reaches the quality {0:.2f} a {1} prompt usually wants, "
                "so the best available was used".format(floor, comp.band)
            )
        # A prompt the scorer could not read is a caveat on the whole decision,
        # so it belongs in the sentence the caller actually reads, not only in
        # complexity.warnings.
        parts.extend(comp.warnings)
        if demoted:
            parts.append(
                "{0} model(s) kept only as a fallback: {1}".format(
                    len(demoted),
                    "; ".join(
                        "{0} {1}".format(name, why)
                        for name, why in sorted(demoted.items())
                    ),
                )
            )
        if skipped:
            parts.append(
                "{0} model(s) set aside: {1}".format(
                    len(skipped),
                    "; ".join(
                        "{0} {1}".format(name, why) for name, why in sorted(skipped.items())
                    ),
                )
            )
        if len(ranked) > 1:
            parts.append(
                "fallbacks, in order: {0}".format(
                    ", ".join(candidate.name for candidate in ranked[1:])
                )
            )
        else:
            parts.append("no fallback behind it")
        return "; ".join(parts) + "."

    # --------------------------------------------------------------- calling

    def complete(
        self,
        prompt: str,
        *,
        prefer: str = "balanced",
        require: Any = None,
        max_cost: Optional[float] = None,
        route: Optional[Route] = None,
        **kw: Any,
    ) -> Completion:
        """Route ``prompt``, call the handler, fall back when one fails.

        Every keyword other than ``prefer``, ``require``, ``max_cost`` and
        ``route`` is passed straight to the handler. A handler that raises is
        caught, recorded as a failed :class:`~llm_router_lite.results.Attempt`,
        and the next model is tried. If every model fails,
        :class:`~llm_router_lite.results.AllModelsFailed` is raised carrying
        every attempt and the last error.
        """
        if route is None:
            route = self.route(
                prompt, prefer=prefer, require=require, max_cost=max_cost
            )
        elif not isinstance(route, Route):
            raise TypeError(
                "route= must be a Route from router.route(), got {0}".format(
                    type(route).__name__
                )
            )

        attempts: List[Attempt] = []
        last_exception: Optional[BaseException] = None
        chain = (route.model,) + tuple(route.alternatives)
        for name in chain:
            model = self._models.get(name)
            if model is None:  # removed between routing and calling
                attempts.append(
                    Attempt(
                        model=name,
                        ok=False,
                        error="model {0!r} is no longer registered".format(name),
                        ms=0.0,
                    )
                )
                last_exception = KeyError(name)
                continue
            started = time.perf_counter()
            try:
                text = model.handler(prompt, **kw)
            except Exception as exc:  # noqa: BLE001 - the whole point is to catch it
                elapsed = round((time.perf_counter() - started) * 1000.0, 3)
                message = "{0}: {1}".format(type(exc).__name__, exc)
                attempts.append(Attempt(model=name, ok=False, error=message, ms=elapsed))
                self._record(name, ok=False, ms=elapsed, cost=0.0)
                last_exception = exc
                logger.warning("model %s failed: %s", name, message)
                continue
            elapsed = round((time.perf_counter() - started) * 1000.0, 3)
            if not isinstance(text, str):
                message = "TypeError: handler returned {0}, expected str".format(
                    type(text).__name__
                )
                attempts.append(Attempt(model=name, ok=False, error=message, ms=elapsed))
                self._record(name, ok=False, ms=elapsed, cost=0.0)
                last_exception = TypeError(message)
                logger.warning("model %s failed: %s", name, message)
                continue
            cost = route.cost_of(name)
            attempts.append(Attempt(model=name, ok=True, error=None, ms=elapsed))
            self._record(name, ok=True, ms=elapsed, cost=cost)
            return Completion(
                text=text,
                model=name,
                attempts=tuple(attempts),
                cost=cost,
                route=route,
            )

        failure = AllModelsFailed(tuple(attempts), route)
        raise failure from last_exception

    def _record(self, name: str, *, ok: bool, ms: float, cost: float) -> None:
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                return
            counter.calls += 1
            counter.total_ms += ms
            if ok:
                counter.total_cost += cost
            else:
                counter.failures += 1

    # ----------------------------------------------------------------- stats

    def stats(self) -> Stats:
        """Calls, failures, cost and mean latency for every registered model."""
        with self._lock:
            per_model = {}
            for name, counter in self._counters.items():
                mean_ms = (
                    round(counter.total_ms / counter.calls, 3) if counter.calls else 0.0
                )
                per_model[name] = ModelStats(
                    name=name,
                    calls=counter.calls,
                    failures=counter.failures,
                    total_cost=round_cost(counter.total_cost),
                    mean_latency_ms=mean_ms,
                )
        return Stats(per_model=per_model)

    def reset_stats(self) -> "Router":
        """Zero every counter. The line-up is untouched."""
        with self._lock:
            for name in self._counters:
                self._counters[name] = _Counter()
        return self


def route(prompt: str, models: Any, **kw: Any) -> Route:
    """Route one prompt against a line-up, in a single line.

    ``models`` takes the same shapes as ``Router(models)``. Keyword arguments
    go to :meth:`Router.route`.
    """
    return Router(models).route(prompt, **kw)
