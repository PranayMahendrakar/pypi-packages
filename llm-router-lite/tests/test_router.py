"""Routing: which model wins, what is set aside, and why."""

import pytest

from llm_router_lite import Route, Router, route

HARD_PROMPT = (
    "Explain step by step why this design scales, compare it with a "
    "queue-based approach, and then recommend one."
)


def small(prompt, **kw):
    return "small: " + prompt


def large(prompt, **kw):
    return "large: " + prompt


def two_model_router():
    router = Router()
    router.add("small", small, cost=0.0002, latency_ms=100, quality=0.4)
    router.add("large", large, cost=0.01, latency_ms=900, quality=0.95)
    return router


def test_quickstart_path():
    router = Router()
    router.add("small", lambda p, **kw: "small says: " + p, cost=0.0002, quality=0.4)
    router.add("large", lambda p, **kw: "large says: " + p, cost=0.01, quality=0.95)

    easy = router.route("What is 2 + 2?")
    assert easy.model == "small"
    assert easy.reason

    answer = router.complete(HARD_PROMPT)
    assert answer.model == "large"
    assert answer.text.startswith("large says: ")


def test_easy_prompt_goes_cheap_hard_prompt_goes_good():
    router = two_model_router()
    assert router.route("What is 2 + 2?").model == "small"
    assert router.route(HARD_PROMPT).model == "large"


def test_prefer_cheap_and_quality_pull_opposite_ways():
    router = two_model_router()
    assert router.route(HARD_PROMPT, prefer="cheap").model == "small"
    assert router.route("What is 2 + 2?", prefer="quality").model == "large"


def test_prefer_fast_picks_the_fast_model():
    router = Router()
    router.add("slow-good", large, cost=0.001, latency_ms=2000, quality=0.9)
    router.add("quick", small, cost=0.001, latency_ms=50, quality=0.5)
    assert router.route("Summarise this.", prefer="fast").model == "quick"


def test_unknown_prefer_raises_valueerror():
    with pytest.raises(ValueError) as excinfo:
        two_model_router().route("hello", prefer="cheapest")
    assert "cheapest" in str(excinfo.value)


def test_require_keeps_a_prompt_local():
    router = Router()
    router.add("local", small, cost=0.0, quality=0.5, tags=("local",))
    router.add("cloud", large, cost=0.01, quality=0.99, tags=("cloud",))
    decision = router.route(HARD_PROMPT, require=("local",))
    assert decision.model == "local"
    assert decision.alternatives == ()
    assert "cloud" in decision.skipped


def test_require_is_case_insensitive_and_accepts_a_bare_string():
    router = Router()
    router.add("local", small, tags=("Local",))
    assert router.route("hi", require="local").model == "local"


def test_require_matching_nothing_raises_valueerror():
    router = two_model_router()
    with pytest.raises(ValueError) as excinfo:
        router.route("hello", require=("vision",))
    assert "vision" in str(excinfo.value)


def test_max_cost_filters_the_field():
    router = two_model_router()
    decision = router.route(HARD_PROMPT, max_cost=0.001)
    assert decision.model == "small"
    assert "large" in decision.skipped


def test_max_tokens_skips_a_model_that_cannot_hold_the_prompt():
    router = Router()
    router.add("tiny", small, cost=0.0, quality=0.9, max_tokens=8)
    router.add("roomy", large, cost=0.01, quality=0.5, max_tokens=100000)
    decision = router.route("word " * 200)
    assert decision.model == "roomy"
    assert "tiny" in decision.skipped


def test_a_prompt_too_big_for_every_model_raises_valueerror():
    router = Router()
    router.add("tiny", small, max_tokens=8)
    with pytest.raises(ValueError) as excinfo:
        router.route("word " * 500)
    message = str(excinfo.value)
    assert "tiny" in message
    assert "8" in message


def test_quality_floor_keeps_a_hard_prompt_off_a_weak_model():
    router = two_model_router()
    decision = router.route(HARD_PROMPT)
    assert decision.model == "large"
    # The floor demotes, it never discards: the weak model loses the pick but
    # stays in the chain, because a weak answer beats no answer if "large" is
    # down. So it is reported as demoted, not skipped, and is still a fallback.
    assert "small" in decision.demoted
    assert "small" not in decision.skipped
    assert decision.alternatives == ("small",)
    assert "quality 0.40" in decision.demoted["small"]
    assert "small" in decision.reason
    assert "small" in decision.to_dict()["demoted"]


def test_quality_floor_relaxes_rather_than_returning_nothing():
    router = Router()
    router.add("weak", small, cost=0.0, quality=0.1)
    decision = router.route(HARD_PROMPT)
    assert decision.model == "weak"
    assert "best available" in decision.reason


def test_single_model_route_says_there_is_no_fallback():
    router = Router()
    router.add("only", small)
    decision = router.route("hello")
    assert decision.alternatives == ()
    assert "no fallback" in decision.reason


def test_route_is_deterministic():
    decisions = [two_model_router().route(HARD_PROMPT) for _ in range(5)]
    first = decisions[0]
    for decision in decisions[1:]:
        assert decision.model == first.model
        assert decision.alternatives == first.alternatives
        assert decision.estimated_cost == first.estimated_cost
        assert decision.reason == first.reason


def test_registration_order_does_not_change_the_pick():
    forward = Router()
    forward.add("small", small, cost=0.0002, latency_ms=100, quality=0.4)
    forward.add("large", large, cost=0.01, latency_ms=900, quality=0.95)
    backward = Router()
    backward.add("large", large, cost=0.01, latency_ms=900, quality=0.95)
    backward.add("small", small, cost=0.0002, latency_ms=100, quality=0.4)
    assert forward.route(HARD_PROMPT).model == backward.route(HARD_PROMPT).model


def test_ties_break_by_name_so_the_result_is_stable():
    router = Router()
    router.add("bbb", small, cost=0.001, latency_ms=100, quality=0.5)
    router.add("aaa", small, cost=0.001, latency_ms=100, quality=0.5)
    assert router.route("hello").model == "aaa"


def test_empty_prompt_routes_as_simple():
    router = two_model_router()
    decision = router.route("")
    assert decision.complexity.band == "simple"
    assert decision.model == "small"
    assert decision.estimated_tokens == 0


def test_routing_calls_no_handler():
    calls = []

    def spy(prompt, **kw):
        calls.append(prompt)
        return "never"

    router = Router()
    router.add("spy", spy)
    router.route("anything at all")
    assert calls == []


def test_no_models_registered_raises_a_clear_valueerror():
    with pytest.raises(ValueError) as excinfo:
        Router().route("hello")
    message = str(excinfo.value)
    assert "no models registered" in message
    assert "add" in message


def test_max_cost_nothing_satisfies_names_the_cheapest():
    router = two_model_router()
    with pytest.raises(ValueError) as excinfo:
        router.route("hello", max_cost=0.0)
    message = str(excinfo.value)
    assert "small" in message
    assert "cheapest" in message


def test_negative_max_cost_raises_valueerror():
    with pytest.raises(ValueError):
        two_model_router().route("hello", max_cost=-1)


def test_non_string_prompt_raises_typeerror():
    with pytest.raises(TypeError):
        two_model_router().route(None)


def test_route_object_shape():
    decision = two_model_router().route(HARD_PROMPT)
    assert isinstance(decision, Route)
    assert decision.model == "large"
    assert decision.alternatives == ("small",)
    assert decision.fallbacks == decision.alternatives
    assert decision.prefer == "balanced"
    assert decision.complexity.band == "hard"
    assert decision.cost_of("small") < decision.cost_of("large")
    payload = decision.to_dict()
    assert payload["model"] == "large"
    assert payload["complexity"]["band"] == "hard"
    assert [c["name"] for c in payload["considered"]]
    assert str(decision) == decision.summary()


def test_cost_of_an_unconsidered_model_raises_keyerror():
    decision = two_model_router().route(HARD_PROMPT)
    with pytest.raises(KeyError):
        decision.cost_of("nope")


def test_estimated_cost_grows_with_the_prompt():
    router = two_model_router()
    short = router.route("hi", prefer="cheap").estimated_cost
    long_one = router.route("word " * 500, prefer="cheap").estimated_cost
    assert long_one > short


def test_convenience_route_function():
    decision = route("What is 2 + 2?", {"small": small, "large": large})
    assert decision.model in ("small", "large")
    assert isinstance(decision, Route)


def test_convenience_route_function_passes_keywords():
    models = {
        "local": {"handler": small, "cost": 0.0, "tags": ["local"]},
        "cloud": {"handler": large, "cost": 0.01, "quality": 0.99},
    }
    assert route(HARD_PROMPT, models, require=("local",)).model == "local"


def test_unicode_prompt_routes_and_explains():
    router = two_model_router()
    decision = router.route("Explique pourquoi: 説明してください 🚀")
    assert decision.model in ("small", "large")
    decision.reason.encode("ascii")
    decision.summary().encode("ascii")


# ------------------------------------------------- three-model cost ranking
#
# Regression tests for the reviewer's blocker. With exactly two models any
# normalisation of cost separates them perfectly, so every preference test
# above passes for a reason that vanishes the moment a third model joins.
# Real line-ups are local + mid + frontier, so the cost preference has to be
# held to that shape.


def three_model_lineup(with_expensive):
    """cheap / mid, and optionally a price outlier nobody should pick."""
    router = Router()
    router.add("cheap", lambda p, **kw: "c", cost=0.0001, quality=0.35, latency_ms=80)
    router.add("mid", lambda p, **kw: "m", cost=0.002, quality=0.70, latency_ms=600)
    if with_expensive:
        router.add(
            "expensive", lambda p, **kw: "e", cost=0.05, quality=0.97, latency_ms=3000
        )
    return router


def test_prefer_cheap_picks_the_cheapest_of_three_models():
    assert three_model_lineup(True).route("hi", prefer="cheap").model == "cheap"


def test_an_expensive_model_nobody_picks_does_not_flip_the_other_two():
    # Independence of irrelevant alternatives: adding an option that does not
    # win must not change which of the remaining options does.
    for prefer in ("cheap", "balanced"):
        without = three_model_lineup(False).route("hi", prefer=prefer).model
        with_outlier = three_model_lineup(True).route("hi", prefer=prefer).model
        assert without == with_outlier == "cheap", prefer


def test_the_outliers_price_does_not_move_the_tipping_point():
    for price in (0.05, 0.5, 5.0, 500.0):
        router = Router()
        router.add("cheap", lambda p, **kw: "c", cost=0.0001, quality=0.35)
        router.add("mid", lambda p, **kw: "m", cost=0.002, quality=0.70)
        router.add("frontier", lambda p, **kw: "f", cost=price, quality=0.97)
        assert router.route("hi", prefer="cheap").model == "cheap", price


def test_a_free_local_model_wins_prefer_cheap_across_realistic_lineups():
    # The flagship shape: a free local model, a cloud mid-tier and a frontier
    # model. "hi" bands simple, so the quality floor is 0.0 and everything
    # qualifies; the free model must win every time.
    for local_quality in (0.20, 0.35):
        for mid_cost in (0.0006, 0.002):
            for mid_quality in (0.60, 0.70):
                for top_cost in (0.03, 0.05):
                    for top_quality in (0.90, 0.97):
                        router = Router()
                        router.add(
                            "local", small, cost=0.0, quality=local_quality, latency_ms=120
                        )
                        router.add(
                            "cloud-mid",
                            large,
                            cost=mid_cost,
                            quality=mid_quality,
                            latency_ms=700,
                        )
                        router.add(
                            "frontier",
                            large,
                            cost=top_cost,
                            quality=top_quality,
                            latency_ms=2200,
                        )
                        decision = router.route("hi", prefer="cheap")
                        assert decision.model == "local", (
                            local_quality,
                            mid_cost,
                            mid_quality,
                            top_cost,
                            top_quality,
                        )


def test_cheapness_is_a_ratio_not_a_stretch_of_the_current_field():
    # A model that costs 20x the cheapest must score far below it on cost, not
    # be squashed next to it by a pricier third model.
    decision = three_model_lineup(True).route("hi", prefer="cheap")
    by_name = {candidate.name: candidate for candidate in decision.considered}
    assert by_name["cheap"].score > by_name["mid"].score
    assert by_name["mid"].score > by_name["expensive"].score


def test_prefer_cheap_still_beats_quality_when_a_better_model_is_registered():
    router = three_model_lineup(True)
    assert router.route(HARD_PROMPT, prefer="quality").model == "expensive"
    assert router.route(HARD_PROMPT, prefer="cheap").model == "cheap"


def test_a_tenfold_price_gap_survives_at_tiny_prices():
    # round_cost keeps significant digits, so sub-1e-8 prices still order.
    router = Router()
    router.add("nano", small, cost=1e-9, quality=0.5)
    router.add("micro", large, cost=1e-8, quality=0.5)
    decision = router.route("hi", prefer="cheap")
    costs = {c.name: c.estimated_cost for c in decision.considered}
    assert costs["nano"] > 0.0
    assert costs["micro"] > costs["nano"]
    assert decision.model == "nano"


# --------------------------------------------------------- non-Latin scripts
#
# Regression tests for the reviewer's major finding: every non-Latin prompt
# used to score exactly 0.0 and land on the weakest model.

HARD_IN_SCRIPT = {
    "japanese": (
        "この分散アーキテクチャがなぜスケールするのかを段階的に説明し、"
        "キューベースの設計と比較し、トレードオフを分析し、限界を証明し、"
        "そして一つを推奨してください。"
    ),
    "chinese": (
        "请逐步解释这个分布式架构为什么能扩展，"
        "与基于队列的设计进行比较，分析权衡，证明极限，然后推荐一个。"
    ),
    "russian": (
        "Объясни пошагово, почему эта распределённая архитектура масштабируется, "
        "сравни её с очередной схемой, проанализируй компромиссы, докажи пределы, "
        "затем порекомендуй один."
    ),
    "arabic": (
        "اشرح خطوة بخطوة لماذا تتوسع هذه البنية الموزعة، "
        "وقارنها بتصميم يعتمد على الطوابير، وحلل المقارنة، ثم أوص بواحد."
    ),
}


def weak_and_strong():
    router = Router()
    router.add("weak", small, cost=0.0001, quality=0.30)
    router.add("strong", large, cost=0.02, quality=0.97)
    return router


def test_a_hard_prompt_reaches_the_strong_model_in_every_script():
    assert weak_and_strong().route(HARD_PROMPT).model == "strong"
    for language, prompt in HARD_IN_SCRIPT.items():
        assert weak_and_strong().route(prompt).model == "strong", language


def test_an_unreadable_script_warns_in_the_reason_instead_of_pretending():
    thai = "อธิบายทีละขั้นตอนว่าทำไมสถาปัตยกรรมนี้จึงขยายตัวได้และเปรียบเทียบกับคิว"
    decision = weak_and_strong().route(thai)
    assert decision.complexity.warnings
    assert "script" in decision.reason
    decision.reason.encode("ascii")


def test_an_english_prompt_routes_without_any_warning():
    decision = weak_and_strong().route(HARD_PROMPT)
    assert decision.complexity.warnings == ()
    assert "script" not in decision.reason


def test_a_free_model_does_not_collapse_the_cost_ordering():
    """Regression: a zero-cost model made the cheapest cost zero, so every priced model
    scored 0.0 on cost and tied. The tie fell through to quality, and prefer="cheap"
    started ordering its fallbacks most-expensive-first."""
    import llm_router_lite as lr

    def build(with_free):
        router = lr.Router()
        router.add("tiny", lambda p, **k: "t", cost=0.10, quality=0.3, latency_ms=50)
        router.add("mid", lambda p, **k: "m", cost=1.00, quality=0.6, latency_ms=200)
        router.add("big", lambda p, **k: "b", cost=10.0, quality=0.95, latency_ms=900)
        if with_free:
            router.add("free", lambda p, **k: "f", cost=0.0, quality=0.2, latency_ms=30)
        return router

    def names(route):
        return [a if isinstance(a, str) else getattr(a, "model", a) for a in route.alternatives]

    without = build(False).route("Summarise this paragraph.", prefer="cheap")
    assert without.model == "tiny"
    assert names(without) == ["mid", "big"]

    with_free = build(True).route("Summarise this paragraph.", prefer="cheap")
    assert with_free.model == "free"
    # the priced models must still fall back cheapest-first
    assert names(with_free) == ["tiny", "mid", "big"]
