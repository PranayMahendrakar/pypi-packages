"""Which format a value is stored in, and what happens when neither works."""

from __future__ import annotations

import datetime as dt
import decimal

import pytest

from prompt_cache import Cache, decode, encode
from prompt_cache._serialize import JSON, PICKLE


@pytest.mark.parametrize(
    "value",
    [
        "a string",
        "unicode: 你好 مرحبا \U0001f680",
        123,
        -7,
        4.5,
        True,
        False,
        None,
        [],
        {},
        ["a", 1, None, {"nested": [1, 2]}],
        {"choices": [{"message": {"content": "hi"}}], "usage": {"tokens": 12}},
    ],
)
def test_json_values_round_trip_unchanged(cache_dir, value):
    cache = Cache(cache_dir)
    cache.set("p", value)
    found, restored = cache.lookup("p")
    assert found
    assert restored == value
    assert type(restored) is type(value)
    assert cache.entries()[0]["format"] == JSON


@pytest.mark.parametrize(
    "value",
    [
        ("a", "tuple"),
        {1: "int key"},
        {"when": dt.date(2026, 1, 31)},
        decimal.Decimal("1.25"),
        {"nan": float("nan")},
        b"raw bytes",
        {"a_set"},
    ],
)
def test_other_values_fall_back_to_pickle(cache_dir, value):
    cache = Cache(cache_dir)
    cache.set("p", value)
    found, restored = cache.lookup("p")
    assert found
    assert type(restored) is type(value)
    assert cache.entries()[0]["format"] == PICKLE


def test_nan_survives_through_pickle(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", float("nan"))
    restored = cache.get("p")
    assert restored != restored  # still NaN


def test_a_tuple_comes_back_as_a_tuple(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", ("a", "b"))
    assert cache.get("p") == ("a", "b")


def test_unserialisable_value_raises_a_typeerror_naming_the_type(cache_dir):
    cache = Cache(cache_dir)

    def gen():  # pragma: no cover - never iterated
        yield 1

    with pytest.raises(TypeError) as excinfo:
        cache.set("p", gen())
    message = str(excinfo.value)
    assert "generator" in message
    assert "prompt-cache cannot store" in message
    assert cache.stats().entries == 0


def test_lambda_also_raises_a_clear_typeerror(cache_dir):
    cache = Cache(cache_dir)
    with pytest.raises(TypeError) as excinfo:
        cache.set("p", lambda x: x)
    assert "function" in str(excinfo.value)


def test_encode_and_decode_are_public_and_symmetric():
    fmt, blob = encode({"a": 1})
    assert fmt == JSON
    assert decode(fmt, blob) == {"a": 1}
    fmt, blob = encode(("x",))
    assert fmt == PICKLE
    assert decode(fmt, blob) == ("x",)


def test_decode_refuses_an_unknown_format():
    with pytest.raises(ValueError):
        decode("protobuf", b"")


def test_an_unreadable_entry_is_dropped_and_reported_as_a_miss(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    key = cache.key("p")
    cache._store._run(
        lambda conn: conn.execute("UPDATE entries SET value = ? WHERE key = ?", (b"not json", key))
    )
    assert cache.get("p") is None
    assert cache.stats().entries == 0


def test_a_value_larger_than_the_size_limit_is_not_stored(cache_dir):
    cache = Cache(cache_dir, max_size_mb=0.001)
    cache.set("p", "x" * 5000)
    assert cache.get("p") is None
    assert cache.stats().entries == 0
