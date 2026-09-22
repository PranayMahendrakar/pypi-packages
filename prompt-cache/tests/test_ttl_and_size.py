"""Time to live, pruning, and the least-recently-used size limit."""

from __future__ import annotations

from prompt_cache import Cache


def test_entries_expire_after_the_ttl(cache_dir, clock):
    cache = Cache(cache_dir, ttl=60)
    cache.set("p", "v")
    clock.advance(59)
    assert cache.get("p") == "v"
    clock.advance(2)
    assert cache.get("p") is None


def test_an_expired_entry_is_removed_when_it_is_found(cache_dir, clock):
    cache = Cache(cache_dir, ttl=10)
    cache.set("p", "v")
    clock.advance(11)
    assert cache.get("p") is None
    assert cache.stats().entries == 0


def test_ttl_none_never_expires(cache_dir, clock):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    clock.advance(10_000_000)
    assert cache.get("p") == "v"


def test_ttl_zero_never_caches(cache_dir):
    cache = Cache(cache_dir, ttl=0)
    assert cache.enabled is False
    cache.set("p", "v")
    assert cache.get("p") is None
    assert cache.stats().entries == 0
    assert cache.stats().misses == 1


def test_ttl_zero_still_lets_a_wrapped_function_work(cache_dir):
    cache = Cache(cache_dir, ttl=0)
    calls = []

    @cache.wrap
    def ask(prompt):
        calls.append(prompt)
        return prompt.upper()

    assert ask("hi") == "HI"
    assert ask("hi") == "HI"
    assert calls == ["hi", "hi"]  # the function ran both times
    assert cache.stats().saved_calls == 0


def test_ttl_zero_still_rejects_an_unserialisable_value(cache_dir):
    cache = Cache(cache_dir, ttl=0)
    try:
        cache.set("p", (x for x in ()))
    except TypeError as exc:
        assert "generator" in str(exc)
    else:  # pragma: no cover - the raise above is the point
        raise AssertionError("expected a TypeError")


def test_prune_removes_only_what_has_expired(cache_dir, clock):
    cache = Cache(cache_dir, ttl=100)
    cache.set("old", "v")
    clock.advance(150)
    cache.set("new", "v")
    assert cache.prune() == 1
    assert cache.stats().entries == 1
    assert cache.get("new") == "v"


def test_prune_is_scoped_to_the_namespace(cache_dir, clock):
    alpha = Cache(cache_dir, ttl=10, namespace="alpha")
    beta = Cache(cache_dir, ttl=10, namespace="beta")
    alpha.set("p", "v")
    beta.set("p", "v")
    clock.advance(20)
    assert alpha.prune() == 1
    assert beta.prune() == 1  # beta's expired row was still there to prune


def test_expired_entries_do_not_count_towards_stats(cache_dir, clock):
    cache = Cache(cache_dir, ttl=10)
    cache.set("p", "v")
    assert cache.stats().entries == 1
    clock.advance(20)
    assert cache.stats().entries == 0


def test_least_recently_used_entries_are_evicted_first(cache_dir, clock):
    cache = Cache(cache_dir, max_size_mb=0.01)  # about 10 kB of values
    payload = "x" * 1000
    for index in range(8):
        clock.advance(1)
        cache.set("prompt {0}".format(index), payload)
    assert cache.stats().evictions == 0

    clock.advance(1)
    assert cache.get("prompt 0") == payload  # touch the oldest so it is no longer the oldest

    for index in range(8, 14):
        clock.advance(1)
        cache.set("prompt {0}".format(index), payload)

    stats = cache.stats()
    assert stats.evictions > 0
    assert stats.size_mb <= 0.01
    assert cache.get("prompt 0") == payload  # survived because it was used recently
    assert cache.get("prompt 1") is None  # the genuinely least recently used went first
    assert cache.get("prompt 13") == payload


def test_eviction_is_scoped_to_the_namespace(cache_dir, clock):
    small = Cache(cache_dir, max_size_mb=0.002, namespace="small")
    other = Cache(cache_dir, namespace="other")
    other.set("keep me", "y" * 1000)
    for index in range(6):
        clock.advance(1)
        small.set("p{0}".format(index), "x" * 500)
    assert small.stats().evictions > 0
    assert other.get("keep me") == "y" * 1000


def test_no_limit_means_nothing_is_evicted(cache_dir, clock):
    cache = Cache(cache_dir)
    for index in range(50):
        clock.advance(1)
        cache.set("p{0}".format(index), "x" * 1000)
    assert cache.stats().entries == 50
    assert cache.stats().evictions == 0
