"""Registering models: the shapes accepted and the numbers validated."""

import pytest

from llm_router_lite import Model, Router, make_model, normalize_models
from llm_router_lite.models import round_cost


def handler(prompt, **kw):
    return "ok"


def test_add_chains_and_registers():
    router = Router()
    returned = router.add("a", handler).add("b", handler)
    assert returned is router
    assert len(router) == 2
    assert router.names == ("a", "b")
    assert "a" in router
    assert router.get("a").name == "a"
    assert [m.name for m in router.models] == ["a", "b"]
    assert "2 model(s)" in repr(router)


def test_duplicate_name_raises_valueerror():
    router = Router()
    router.add("a", handler)
    with pytest.raises(ValueError) as excinfo:
        router.add("a", handler)
    assert "already registered" in str(excinfo.value)


def test_remove_drops_a_model_and_unknown_names_raise_keyerror():
    router = Router()
    router.add("a", handler).add("b", handler)
    router.remove("a")
    assert router.names == ("b",)
    with pytest.raises(KeyError):
        router.remove("a")
    with pytest.raises(KeyError):
        router.get("a")


def test_router_accepts_a_mapping_of_handlers():
    router = Router({"a": handler, "b": handler})
    assert set(router.names) == {"a", "b"}


def test_router_accepts_a_mapping_of_option_dicts():
    router = Router({"a": {"handler": handler, "cost": 0.5, "tags": ["local"]}})
    assert router.get("a").cost == 0.5
    assert router.get("a").tags == ("local",)


def test_router_accepts_a_sequence_of_pairs_dicts_and_models():
    model = make_model("c", handler, cost=1.0)
    router = Router([("a", handler), {"name": "b", "handler": handler}, model])
    assert set(router.names) == {"a", "b", "c"}


def test_router_accepts_a_single_model():
    router = Router(make_model("a", handler))
    assert router.names == ("a",)


def test_router_with_none_starts_empty():
    assert len(Router()) == 0
    assert len(Router(None)) == 0


def test_normalize_models_rejects_junk():
    with pytest.raises(TypeError):
        normalize_models(42)
    with pytest.raises(TypeError):
        normalize_models({"a": 42})
    with pytest.raises(TypeError):
        normalize_models([42])


def test_a_model_dict_without_a_handler_is_a_clear_valueerror():
    with pytest.raises(ValueError) as excinfo:
        normalize_models([{"name": "a"}])
    assert "handler" in str(excinfo.value)


def test_a_model_dict_without_a_name_is_a_clear_valueerror():
    with pytest.raises(ValueError) as excinfo:
        normalize_models([{"handler": handler}])
    assert "name" in str(excinfo.value)


def test_unknown_model_option_is_rejected_by_name():
    with pytest.raises(ValueError) as excinfo:
        normalize_models([{"name": "a", "handler": handler, "colour": "blue"}])
    assert "colour" in str(excinfo.value)


def test_handler_must_be_callable():
    with pytest.raises(TypeError) as excinfo:
        make_model("a", "not callable")
    assert "callable" in str(excinfo.value)


def test_name_must_be_a_non_empty_string():
    with pytest.raises(ValueError):
        make_model("   ", handler)
    with pytest.raises(TypeError):
        make_model(None, handler)


def test_numbers_are_validated():
    with pytest.raises(ValueError):
        make_model("a", handler, cost=-1)
    with pytest.raises(TypeError):
        make_model("a", handler, cost="free")
    with pytest.raises(ValueError):
        make_model("a", handler, quality=1.5)
    with pytest.raises(ValueError):
        make_model("a", handler, latency_ms=-5)
    with pytest.raises(ValueError):
        make_model("a", handler, max_tokens=0)
    with pytest.raises(TypeError):
        make_model("a", handler, max_tokens=1.5)


def test_nan_cost_is_rejected():
    with pytest.raises(ValueError):
        make_model("a", handler, cost=float("nan"))


def test_tags_are_cleaned_and_deduplicated():
    model = make_model("a", handler, tags=["local", " local ", "", "gpu"])
    assert model.tags == ("local", "gpu")
    assert make_model("b", handler, tags="local").tags == ("local",)
    with pytest.raises(TypeError):
        make_model("c", handler, tags=[1])


def test_model_helpers():
    model = make_model("a", handler, cost=1.0, quality=0.8, max_tokens=1000, tags=("local",))
    assert model.has_tags(("local",)) is True
    assert model.has_tags(("cloud",)) is False
    assert model.has_tags(()) is True
    assert model.fits(10) is True
    assert model.fits(5000) is False
    assert model.tokens_for(100) == 356
    assert model.tokens_for(900) == 1000
    assert model.estimated_cost(100) == 0.356
    assert "a" in model.describe()
    model.describe().encode("ascii")
    payload = model.to_dict()
    assert payload["name"] == "a"
    assert payload["tags"] == ["local"]
    assert payload["handler"] == "handler"


def test_a_model_without_max_tokens_always_fits():
    model = make_model("a", handler)
    assert model.fits(10 ** 9) is True
    assert isinstance(model, Model)


# ------------------------------------------------------------- tiny prices
#
# Regression tests for the reviewer's rounding finding: estimated_cost used to
# round to 8 decimal places, so every price below roughly 1e-8 collapsed to
# exactly 0.0 and the order the caller supplied was lost.


def test_round_cost_keeps_significant_digits_not_decimal_places():
    assert round_cost(0.0) == 0.0
    assert round_cost(1e-9) == 1e-9
    assert round_cost(1e-12) == 1e-12
    assert round_cost(-1e-9) == -1e-9
    # what it is actually for: tidying the float noise off a multiplication
    assert round_cost(0.1 * 3) == 0.3
    assert round_cost(float("inf")) == float("inf")


def test_a_tiny_price_does_not_round_away_to_zero():
    nano = make_model("nano", handler, cost=1e-9)
    micro = make_model("micro", handler, cost=1e-8)
    assert nano.estimated_cost(10) > 0.0
    assert micro.estimated_cost(10) > nano.estimated_cost(10)


def test_the_order_of_prices_survives_at_every_scale():
    previous = 0.0
    for exponent in range(-12, 3):
        model = make_model("m", handler, cost=10.0 ** exponent)
        current = model.estimated_cost(100)
        assert current > previous, exponent
        previous = current


def test_estimated_cost_is_still_the_documented_formula():
    model = make_model("a", handler, cost=0.01)
    # cost * (prompt tokens + 256 answer tokens) / 1000, with the float noise
    # of the multiplication tidied off and nothing else changed.
    assert model.estimated_cost(100) == pytest.approx(0.01 * 356 / 1000.0)
    assert model.estimated_cost(100) == 0.00356
