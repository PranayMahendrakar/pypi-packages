"""Core behaviour of Cache: get, set, parameters, namespaces, stats."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_cache import Cache, Stats, cached, resolve_path


def test_set_then_get_round_trips(cache_dir):
    cache = Cache(cache_dir)
    cache.set("hello", "world")
    assert cache.get("hello") == "world"
    cache.close()


def test_get_on_a_missing_key_returns_none_and_writes_nothing(cache_dir):
    cache = Cache(cache_dir)
    for index in range(200):
        assert cache.get("never stored {0}".format(index)) is None
    assert cache.stats().entries == 0
    assert cache.stats().misses == 200
    assert cache.stats().hits == 0


def test_parameters_are_part_of_the_key(cache_dir):
    cache = Cache(cache_dir)
    cache.set("summarise", "cheap answer", model="gpt-4o-mini")
    cache.set("summarise", "expensive answer", model="gpt-4o")
    assert cache.get("summarise", model="gpt-4o-mini") == "cheap answer"
    assert cache.get("summarise", model="gpt-4o") == "expensive answer"
    assert cache.get("summarise") is None
    assert cache.get("summarise", model="gpt-4o", temperature=0.7) is None


def test_parameter_order_does_not_matter(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v", model="a", temperature=0.2)
    assert cache.get("p", temperature=0.2, model="a") == "v"


def test_parameter_types_do_not_collide(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "as-int", n=1)
    cache.set("p", "as-str", n="1")
    cache.set("p", "as-bool", n=True)
    assert cache.get("p", n=1) == "as-int"
    assert cache.get("p", n="1") == "as-str"
    assert cache.get("p", n=True) == "as-bool"


def test_unhashable_parameter_values_still_work(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v", messages=[{"role": "user", "content": "hi"}])
    assert cache.get("p", messages=[{"role": "user", "content": "hi"}]) == "v"
    assert cache.get("p", messages=[{"role": "user", "content": "bye"}]) is None


def test_lookup_tells_a_stored_none_from_a_miss(cache_dir):
    cache = Cache(cache_dir)
    assert cache.lookup("p") == (False, None)
    cache.set("p", None)
    assert cache.lookup("p") == (True, None)
    assert cache.get("p") is None


def test_keys_are_hashes_so_long_prompts_cost_nothing_extra(cache_dir):
    cache = Cache(cache_dir)
    huge = "explain this document:\n" + ("word " * 40_000)
    assert len(huge) > 200_000
    cache.set(huge, "short answer")
    assert cache.get(huge) == "short answer"
    assert len(cache.key(huge)) == 64
    entry = cache.entries()[0]
    assert len(entry["preview"]) <= 200
    assert entry["key"] == cache.key(huge)


def test_namespaces_do_not_see_each_other(cache_dir):
    one = Cache(cache_dir, namespace="alpha")
    two = Cache(cache_dir, namespace="beta")
    one.set("p", "from alpha")
    two.set("p", "from beta")
    assert one.get("p") == "from alpha"
    assert two.get("p") == "from beta"
    assert one.stats().entries == 1
    assert two.stats().entries == 1


def test_clear_is_scoped_to_the_namespace(cache_dir):
    one = Cache(cache_dir, namespace="alpha")
    two = Cache(cache_dir, namespace="beta")
    one.set("p", 1)
    two.set("p", 2)
    assert one.clear() == 1
    assert one.get("p") is None
    assert two.get("p") == 2


def test_clear_star_empties_the_whole_file(cache_dir):
    one = Cache(cache_dir, namespace="alpha")
    two = Cache(cache_dir, namespace="beta")
    one.set("p", 1)
    two.set("p", 2)
    assert one.clear("*") == 2
    assert two.stats().entries == 0


def test_clear_can_name_another_namespace(cache_dir):
    one = Cache(cache_dir, namespace="alpha")
    two = Cache(cache_dir, namespace="beta")
    two.set("p", 2)
    assert one.clear("beta") == 1
    assert two.get("p") is None


def test_stats_reports_hits_misses_and_rate(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    cache.get("p")
    cache.get("p")
    cache.get("q")
    stats = cache.stats()
    assert isinstance(stats, Stats)
    assert (stats.hits, stats.misses, stats.lookups) == (2, 1, 3)
    assert stats.hit_rate == pytest.approx(2 / 3)
    assert stats.entries == 1
    assert stats.size_mb >= 0.0
    assert stats.to_dict()["hit_rate"] == pytest.approx(2 / 3)
    assert "hit rate" in stats.summary()
    assert str(stats) == stats.summary()


def test_stats_summary_is_plain_ascii(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    cache.get("p")
    text = cache.stats().summary()
    assert text.encode("ascii")  # raises if any character is not ASCII
    assert text.count("\n") == 1


def test_stats_on_an_empty_cache_says_so(cache_dir):
    stats = Cache(cache_dir).stats()
    assert stats.hit_rate == 0.0
    assert stats.summary().startswith("prompt-cache: no lookups yet")


def test_context_manager_closes_and_reopens(cache_dir):
    with Cache(cache_dir) as cache:
        cache.set("p", "v")
        assert cache.get("p") == "v"
    # closed, but a later call must still work rather than raise
    assert cache.get("p") == "v"
    cache.close()
    cache.close()  # idempotent


def test_repr_explains_the_configuration(cache_dir):
    text = repr(Cache(cache_dir, ttl=60, max_size_mb=5, namespace="ns", normalize=False))
    assert "ttl=60.0" in text
    assert "max_size_mb=5.0" in text
    assert "namespace='ns'" in text
    assert "normalize=False" in text


def test_properties_are_read_only(cache_dir):
    cache = Cache(cache_dir, ttl=30, namespace="ns")
    assert cache.namespace == "ns"
    assert cache.ttl == 30.0
    assert cache.max_size_mb is None
    assert cache.normalize is True
    assert cache.enabled is True
    with pytest.raises(AttributeError):
        cache.namespace = "other"


def test_memory_cache_needs_no_files():
    cache = Cache(":memory:")
    cache.set("p", "v")
    assert cache.get("p") == "v"
    assert cache.path == ":memory:"


def test_cached_decorator_exposes_its_cache(tmp_path):
    decorator = cached(tmp_path / "c", namespace="ns")

    @decorator
    def ask(prompt):
        return prompt.upper()

    assert ask("hi") == "HI"
    assert ask.cache is decorator.cache
    assert ask.cache.namespace == "ns"


def test_resolve_path_rules(tmp_path):
    assert resolve_path(tmp_path) == tmp_path / "cache.db"
    assert resolve_path(tmp_path / "my.db") == tmp_path / "my.db"
    assert resolve_path(str(tmp_path / "my.sqlite3")) == tmp_path / "my.sqlite3"
    assert resolve_path(":memory:") == ":memory:"
    assert Path(str(resolve_path(None))).name == "cache.db"
    with pytest.raises(TypeError):
        resolve_path(42)


def test_constructor_rejects_nonsense(tmp_path):
    with pytest.raises(ValueError):
        Cache(tmp_path, ttl=-1)
    with pytest.raises(ValueError):
        Cache(tmp_path, max_size_mb=0)
    with pytest.raises(TypeError):
        Cache(tmp_path, ttl="soon")
    with pytest.raises(TypeError):
        Cache(tmp_path, namespace=7)
    with pytest.raises(ValueError):
        Cache(tmp_path, namespace="")


def test_prompt_must_be_a_string(cache_dir):
    cache = Cache(cache_dir)
    with pytest.raises(TypeError) as excinfo:
        cache.get(["not", "a", "prompt"])
    assert "list" in str(excinfo.value)
    with pytest.raises(TypeError):
        cache.set(42, "v")
