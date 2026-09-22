"""llm-router-lite: send each prompt to the cheapest model that can handle it.

Three lines get you the whole thing::

    from llm_router_lite import Router

    router = Router()
    router.add("small", small_model, cost=0.0002, quality=0.4)
    router.add("large", large_model, cost=0.01, quality=0.95)

    answer = router.complete("What is 2 + 2?")

Easy prompts go to the cheap model, hard prompts to the good one, and if a
model raises, the next one in line is tried. ``answer.route.reason`` says in
plain words why that model was picked.

This package never talks to a provider. It has no API keys, no HTTP client and
no network code at all: *you* supply the handler for each model, and the
router decides which of your handlers to call.
"""

from .complexity import (
    CHARS_PER_TOKEN,
    MODERATE_MAX,
    SIGNAL_WEIGHTS,
    SIMPLE_MAX,
    Complexity,
    band_for,
    complexity,
    estimate_tokens,
)
from .models import (
    DEFAULT_COMPLETION_TOKENS,
    DEFAULT_LATENCY_MS,
    Model,
    make_model,
    normalize_models,
)
from .results import (
    AllModelsFailed,
    Attempt,
    Candidate,
    Completion,
    ModelStats,
    Route,
    Stats,
)
from .router import (
    BALANCED_WEIGHTS,
    PREFERENCE_WEIGHTS,
    PREFERENCES,
    QUALITY_FLOOR,
    Router,
    route,
)

__version__ = "0.1.0"

__all__ = [
    "AllModelsFailed",
    "Attempt",
    "BALANCED_WEIGHTS",
    "CHARS_PER_TOKEN",
    "Candidate",
    "Complexity",
    "Completion",
    "DEFAULT_COMPLETION_TOKENS",
    "DEFAULT_LATENCY_MS",
    "MODERATE_MAX",
    "Model",
    "ModelStats",
    "PREFERENCES",
    "PREFERENCE_WEIGHTS",
    "QUALITY_FLOOR",
    "Route",
    "Router",
    "SIGNAL_WEIGHTS",
    "SIMPLE_MAX",
    "Stats",
    "__version__",
    "band_for",
    "complexity",
    "estimate_tokens",
    "make_model",
    "normalize_models",
    "route",
]
