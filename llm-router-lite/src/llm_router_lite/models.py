"""One registered model: what it costs, how fast it is, how good it is.

A model here is not an API client. It is a name, a handler you supply, and a
few numbers the router compares. Nothing in this module calls anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Completion tokens assumed when estimating what a call will cost.
DEFAULT_COMPLETION_TOKENS = 256
#: Latency assumed for a model that was registered without ``latency_ms``,
#: when no other model has a known latency either.
DEFAULT_LATENCY_MS = 1000.0
#: Significant digits kept on an estimated cost. Enough to tidy the float
#: noise a multiplication leaves behind, never enough to round two prices
#: together: a cost of 0.000000001 stays distinct from one of 0.00000001.
COST_SIGNIFICANT_DIGITS = 12


def round_cost(value: float) -> float:
    """Tidy float noise off a cost without flattening the order of two prices.

    Rounding to a fixed number of decimal places turns every price below that
    place into 0.0, which loses the ordering the caller supplied. This rounds
    to significant digits instead, so the result stays faithful at any scale.
    """
    if not value or not math.isfinite(value):
        return float(value)
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, COST_SIGNIFICANT_DIGITS - 1 - magnitude)


def _clean_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError("model name must be a string, got {0}".format(type(name).__name__))
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("model name must not be empty")
    return cleaned


def _clean_tags(tags: Any) -> Tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = (tags,)
    if isinstance(tags, Mapping) or not isinstance(tags, Iterable):
        raise TypeError(
            "tags must be a string or a sequence of strings, got {0}".format(
                type(tags).__name__
            )
        )
    cleaned: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(
                "every tag must be a string, got {0}".format(type(tag).__name__)
            )
        stripped = tag.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _clean_number(
    value: Any, label: str, *, minimum: float = 0.0, allow_none: bool = False
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError("{0} must be a number, got None".format(label))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "{0} must be a number, got {1}".format(label, type(value).__name__)
        )
    number = float(value)
    if number != number:  # NaN
        raise ValueError("{0} must be a real number, got NaN".format(label))
    if number < minimum:
        raise ValueError("{0} must be >= {1}, got {2}".format(label, minimum, value))
    return number


@dataclass(frozen=True)
class Model:
    """A model the router may pick.

    ``cost`` is the price of 1000 tokens (prompt plus answer) in whatever unit
    you care about - dollars, credits, cents. ``quality`` is your own 0-1
    opinion of how good the answers are. ``latency_ms`` is the typical round
    trip. ``max_tokens`` is the largest prompt-plus-answer it can hold, and
    ``tags`` are free labels such as ``"local"`` you can require at routing
    time.
    """

    name: str
    handler: Callable[..., str]
    cost: float = 0.0
    latency_ms: Optional[float] = None
    quality: float = 0.5
    max_tokens: Optional[int] = None
    tags: Tuple[str, ...] = ()

    def has_tags(self, required: Sequence[str]) -> bool:
        """True when this model carries every tag in ``required`` (case-insensitive)."""
        mine = {tag.lower() for tag in self.tags}
        return all(tag.lower() in mine for tag in required)

    def fits(self, prompt_tokens: int) -> bool:
        """True when a prompt of this size leaves room for an answer."""
        if self.max_tokens is None:
            return True
        return prompt_tokens < self.max_tokens

    def tokens_for(self, prompt_tokens: int) -> int:
        """Prompt tokens plus the answer allowance, capped by ``max_tokens``."""
        allowance = DEFAULT_COMPLETION_TOKENS
        if self.max_tokens is not None:
            allowance = max(0, min(allowance, self.max_tokens - prompt_tokens))
        return prompt_tokens + allowance

    def estimated_cost(self, prompt_tokens: int) -> float:
        """What one call with a prompt this size is expected to cost."""
        return round_cost(self.cost * self.tokens_for(prompt_tokens) / 1000.0)

    def describe(self) -> str:
        """One plain-ASCII line about this model."""
        parts = ["cost {0:g} per 1k tokens".format(self.cost), "quality {0:.2f}".format(self.quality)]
        if self.latency_ms is not None:
            parts.append("{0:g} ms".format(self.latency_ms))
        if self.max_tokens is not None:
            parts.append("up to {0} tokens".format(self.max_tokens))
        if self.tags:
            parts.append("tags: " + ", ".join(self.tags))
        return "{0} ({1})".format(self.name, ", ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict. The handler is described, not serialised."""
        return {
            "name": self.name,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "quality": self.quality,
            "max_tokens": self.max_tokens,
            "tags": list(self.tags),
            "handler": getattr(self.handler, "__name__", type(self.handler).__name__),
        }


def make_model(
    name: str,
    handler: Callable[..., str],
    *,
    cost: float = 0.0,
    latency_ms: Optional[float] = None,
    quality: float = 0.5,
    max_tokens: Optional[int] = None,
    tags: Sequence[str] = (),
) -> Model:
    """Validate the numbers and build a :class:`Model`."""
    clean_name = _clean_name(name)
    if not callable(handler):
        raise TypeError(
            "handler for {0!r} must be callable, got {1}".format(
                clean_name, type(handler).__name__
            )
        )
    clean_cost = _clean_number(cost, "cost for {0!r}".format(clean_name))
    clean_latency = _clean_number(
        latency_ms, "latency_ms for {0!r}".format(clean_name), allow_none=True
    )
    clean_quality = _clean_number(quality, "quality for {0!r}".format(clean_name))
    if clean_quality > 1.0:
        raise ValueError(
            "quality for {0!r} must be between 0 and 1, got {1}".format(
                clean_name, quality
            )
        )
    clean_max_tokens: Optional[int] = None
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError(
                "max_tokens for {0!r} must be an int, got {1}".format(
                    clean_name, type(max_tokens).__name__
                )
            )
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens for {0!r} must be positive, got {1}".format(
                    clean_name, max_tokens
                )
            )
        clean_max_tokens = int(max_tokens)
    return Model(
        name=clean_name,
        handler=handler,
        cost=float(clean_cost),
        latency_ms=clean_latency,
        quality=float(clean_quality),
        max_tokens=clean_max_tokens,
        tags=_clean_tags(tags),
    )


_SPEC_KEYS = ("handler", "cost", "latency_ms", "quality", "max_tokens", "tags")


def _model_from_spec(name: Optional[str], spec: Mapping[str, Any]) -> Model:
    unknown = [key for key in spec if key not in _SPEC_KEYS and key != "name"]
    if unknown:
        raise ValueError(
            "unknown model option(s) {0}; allowed: {1}".format(
                ", ".join(repr(key) for key in sorted(unknown)),
                ", ".join(_SPEC_KEYS),
            )
        )
    spec_name = name if name is not None else spec.get("name")
    if spec_name is None:
        raise ValueError("a model given as a dict needs a 'name'")
    handler = spec.get("handler")
    if handler is None:
        raise ValueError(
            "model {0!r} has no handler; give it a callable(prompt, **kw) -> str".format(
                spec_name
            )
        )
    return make_model(
        spec_name,
        handler,
        cost=spec.get("cost", 0.0),
        latency_ms=spec.get("latency_ms"),
        quality=spec.get("quality", 0.5),
        max_tokens=spec.get("max_tokens"),
        tags=spec.get("tags", ()),
    )


def normalize_models(models: Any) -> List[Model]:
    """Turn whatever the caller passed as ``models=`` into a list of models.

    Accepted shapes:

    * ``None`` - no models yet.
    * a mapping of ``name -> handler``, ``name -> dict of options``, or
      ``name -> Model``;
    * a sequence of :class:`Model`, of option dicts carrying a ``"name"``, or
      of ``(name, handler)`` pairs.
    """
    if models is None:
        return []
    if isinstance(models, Model):
        return [models]
    result: List[Model] = []
    if isinstance(models, Mapping):
        for name, value in models.items():
            if isinstance(value, Model):
                result.append(value)
            elif isinstance(value, Mapping):
                result.append(_model_from_spec(name, value))
            elif callable(value):
                result.append(make_model(name, value))
            else:
                raise TypeError(
                    "model {0!r} must map to a handler, a dict of options or a Model, "
                    "got {1}".format(name, type(value).__name__)
                )
        return result
    if isinstance(models, (str, bytes)) or not isinstance(models, Iterable):
        raise TypeError(
            "models must be a mapping or a sequence, got {0}".format(
                type(models).__name__
            )
        )
    for item in models:
        if isinstance(item, Model):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(_model_from_spec(None, item))
        elif isinstance(item, tuple) and len(item) == 2:
            result.append(make_model(item[0], item[1]))
        else:
            raise TypeError(
                "each model must be a Model, a dict of options or a (name, handler) "
                "pair, got {0}".format(type(item).__name__)
            )
    return result
