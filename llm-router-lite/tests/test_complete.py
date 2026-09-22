"""Calling: fallbacks, failures, attempts and stats."""

import pytest

from llm_router_lite import AllModelsFailed, Attempt, Completion, Router

HARD_PROMPT = (
    "Explain step by step why this design scales, compare it with a "
    "queue-based approach, and then recommend one."
)


def ok_handler(prompt, **kw):
    return "answered: " + prompt


def boom(prompt, **kw):
    raise RuntimeError("provider is down")


def test_complete_returns_the_handler_text():
    router = Router()
    router.add("only", ok_handler, cost=0.001)
    answer = router.complete("hello")
    assert isinstance(answer, Completion)
    assert answer.text == "answered: hello"
    assert answer.model == "only"
    assert str(answer) == answer.text
    assert answer.ok is True
    assert answer.fallbacks_used == 0
    assert len(answer.attempts) == 1
    assert answer.attempts[0].ok is True


def test_a_raising_handler_is_caught_and_the_next_model_answers():
    router = Router()
    router.add("broken", boom, cost=0.0, quality=0.9)
    router.add("backup", ok_handler, cost=0.01, quality=0.9)
    answer = router.complete("hello")
    assert answer.model == "backup"
    assert answer.fallbacks_used == 1
    assert [a.model for a in answer.attempts] == ["broken", "backup"]
    assert answer.attempts[0].ok is False
    assert "provider is down" in answer.attempts[0].error
    assert answer.attempts[1].ok is True


def test_a_raising_handler_never_propagates_raw():
    router = Router()
    router.add("broken", boom, cost=0.0, quality=0.9)
    router.add("backup", ok_handler, cost=0.01, quality=0.9)
    try:
        router.complete("hello")
    except RuntimeError:  # pragma: no cover - the point is that this never runs
        pytest.fail("the handler exception escaped complete()")


def test_every_model_failing_surfaces_the_last_error_with_all_attempts():
    def boom_one(prompt, **kw):
        raise ValueError("first failed")

    def boom_two(prompt, **kw):
        raise KeyError("second failed")

    router = Router()
    router.add("one", boom_one, cost=0.0, quality=0.9)
    router.add("two", boom_two, cost=0.01, quality=0.9)

    with pytest.raises(AllModelsFailed) as excinfo:
        router.complete("hello")

    failure = excinfo.value
    assert isinstance(failure, RuntimeError)
    assert [a.model for a in failure.attempts] == ["one", "two"]
    assert all(a.ok is False for a in failure.attempts)
    assert failure.last_model == "two"
    assert "second failed" in failure.last_error
    assert "second failed" in str(failure)
    assert "first failed" in failure.attempts[0].error
    assert isinstance(failure.__cause__, KeyError)
    assert failure.route.model == "one"
    payload = failure.to_dict()
    assert payload["last_model"] == "two"
    assert len(payload["attempts"]) == 2
    assert "one" in failure.summary()


def test_all_models_failed_message_is_not_a_bare_exception():
    router = Router()
    router.add("only", boom)
    with pytest.raises(AllModelsFailed) as excinfo:
        router.complete("hello")
    message = str(excinfo.value)
    assert "all 1 model(s) failed" in message
    assert "provider is down" in message


def test_a_handler_returning_a_non_string_is_recorded_as_a_failure():
    def wrong_type(prompt, **kw):
        return 42

    router = Router()
    router.add("wrong", wrong_type, cost=0.0, quality=0.9)
    router.add("right", ok_handler, cost=0.01, quality=0.9)
    answer = router.complete("hello")
    assert answer.model == "right"
    assert "expected str" in answer.attempts[0].error


def test_keywords_reach_the_handler():
    seen = {}

    def handler(prompt, **kw):
        seen.update(kw)
        return "ok"

    router = Router()
    router.add("only", handler)
    router.complete("hello", temperature=0.2, max_tokens=64)
    assert seen == {"temperature": 0.2, "max_tokens": 64}


def test_reserved_keywords_do_not_reach_the_handler():
    seen = {}

    def handler(prompt, **kw):
        seen.update(kw)
        return "ok"

    router = Router()
    router.add("local", handler, tags=("local",))
    router.complete("hello", prefer="cheap", require=("local",), max_cost=1.0)
    assert seen == {}


def test_complete_accepts_a_prepared_route():
    router = Router()
    router.add("only", ok_handler, cost=0.002)
    decision = router.route("hello")
    answer = router.complete("hello", route=decision)
    assert answer.route is decision
    assert answer.cost == decision.estimated_cost


def test_complete_rejects_a_route_of_the_wrong_type():
    router = Router()
    router.add("only", ok_handler)
    with pytest.raises(TypeError):
        router.complete("hello", route="not a route")


def test_complete_with_no_models_raises_valueerror():
    with pytest.raises(ValueError):
        Router().complete("hello")


def test_completion_shape_and_to_dict():
    router = Router()
    router.add("broken", boom, cost=0.0, quality=0.9)
    router.add("backup", ok_handler, cost=0.01, quality=0.9)
    answer = router.complete("hello")
    assert answer.ms >= 0.0
    assert answer.cost > 0.0
    payload = answer.to_dict()
    assert payload["model"] == "backup"
    assert payload["fallbacks_used"] == 1
    assert len(payload["attempts"]) == 2
    assert payload["route"]["model"] == "broken"
    assert answer.summary()
    answer.summary().encode("ascii")


def test_attempt_summary_reads_plainly():
    good = Attempt(model="m", ok=True, error=None, ms=1.5)
    bad = Attempt(model="m", ok=False, error="nope", ms=2.0)
    assert "ok" in good.summary()
    assert "nope" in bad.summary()
    assert str(bad) == bad.summary()
    assert good.to_dict()["ok"] is True


def test_stats_count_calls_failures_cost_and_latency():
    router = Router()
    router.add("broken", boom, cost=0.0, quality=0.9)
    router.add("backup", ok_handler, cost=0.01, quality=0.9)
    router.complete("hello")
    router.complete("hello again")

    stats = router.stats()
    assert stats["broken"].calls == 2
    assert stats["broken"].failures == 2
    assert stats["broken"].successes == 0
    assert stats["broken"].failure_rate == 1.0
    assert stats["broken"].total_cost == 0.0
    assert stats["backup"].calls == 2
    assert stats["backup"].failures == 0
    assert stats["backup"].total_cost > 0.0
    assert stats["backup"].mean_latency_ms >= 0.0
    assert stats.calls == 4
    assert stats.failures == 2
    assert stats.total_cost > 0.0
    assert len(stats) == 2
    assert {item.name for item in stats} == {"broken", "backup"}
    assert stats.summary()
    stats.summary().encode("ascii")
    assert stats.to_dict()["calls"] == 4


def test_stats_start_at_zero_and_reset():
    router = Router()
    router.add("only", ok_handler, cost=0.01)
    assert router.stats().calls == 0
    assert router.stats()["only"].failure_rate == 0.0
    router.complete("hello")
    assert router.stats().calls == 1
    router.reset_stats()
    assert router.stats().calls == 0
    assert router.stats()["only"].total_cost == 0.0


def test_stats_are_not_moved_by_routing_alone():
    router = Router()
    router.add("only", ok_handler)
    router.route("hello")
    assert router.stats().calls == 0
