"""wrap(), memoize() and the module-level cached() decorator."""

from __future__ import annotations

import pytest

from prompt_cache import Cache, cached


def test_wrap_calls_the_function_once(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt):
        calls.append(prompt)
        return prompt.upper()

    assert ask("hi") == "HI"
    assert ask("hi") == "HI"
    assert calls == ["hi"]
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.saved_calls) == (1, 1, 1)


def test_wrap_keeps_the_function_metadata(cache_dir):
    cache = Cache(cache_dir)

    @cache.wrap
    def ask(prompt, model="m"):
        """Ask the model."""
        return prompt

    assert ask.__name__ == "ask"
    assert ask.__doc__ == "Ask the model."
    assert ask.cache is cache


def test_keyword_arguments_are_part_of_the_key(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt, model="gpt-4o", temperature=0.0):
        calls.append((prompt, model, temperature))
        return "{0}|{1}|{2}".format(prompt, model, temperature)

    ask("p")
    ask("p")
    ask("p", model="other")
    ask("p", temperature=0.9)
    assert len(calls) == 3


def test_extra_positional_arguments_are_part_of_the_key(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt, system):
        calls.append((prompt, system))
        return prompt + system

    ask("p", "be terse")
    ask("p", "be terse")
    ask("p", "be verbose")
    assert len(calls) == 2


def test_default_arguments_are_not_guessed(cache_dir):
    """Passing a default explicitly is a different call, and honestly a different key."""
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt, model="gpt-4o"):
        calls.append(model)
        return model

    ask("p")
    ask("p", model="gpt-4o")
    assert len(calls) == 2


def test_the_prompt_may_be_passed_by_keyword(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(question):
        calls.append(question)
        return question.upper()

    assert ask(question="hi") == "HI"
    assert ask("hi") == "HI"
    assert calls == ["hi"]


def test_calling_with_no_prompt_is_a_clear_error(cache_dir):
    cache = Cache(cache_dir)

    @cache.wrap
    def ask(prompt):
        return prompt

    with pytest.raises(TypeError) as excinfo:
        ask()
    assert "prompt" in str(excinfo.value)


def test_two_functions_never_read_each_other(cache_dir):
    cache = Cache(cache_dir)

    @cache.wrap
    def first(prompt):
        return "first"

    @cache.wrap
    def second(prompt):
        return "second"

    assert first("p") == "first"
    assert second("p") == "second"


def test_a_raised_exception_is_not_cached(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt):
        calls.append(prompt)
        raise RuntimeError("the api was down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            ask("p")
    assert len(calls) == 2
    assert cache.stats().entries == 0


def test_a_function_returning_none_is_still_cached(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.wrap
    def ask(prompt):
        calls.append(prompt)
        return None

    assert ask("p") is None
    assert ask("p") is None
    assert calls == ["p"]


def test_memoize_is_the_same_thing(cache_dir):
    cache = Cache(cache_dir)
    calls = []

    @cache.memoize
    def ask(prompt):
        calls.append(prompt)
        return prompt[::-1]

    assert ask("abc") == "cba"
    assert ask("abc") == "cba"
    assert calls == ["abc"]


def test_wrap_needs_a_callable(cache_dir):
    with pytest.raises(TypeError):
        Cache(cache_dir).wrap("not callable")


def test_cached_shares_one_cache_between_functions(tmp_path):
    decorator = cached(tmp_path / "c")

    @decorator
    def first(prompt):
        return "first"

    @decorator
    def second(prompt):
        return "second"

    first("p")
    second("p")
    assert first.cache is second.cache
    assert first.cache.stats().entries == 2


def test_cached_passes_keywords_through(tmp_path):
    @cached(tmp_path / "c", ttl=0, namespace="off")
    def ask(prompt):
        return prompt

    ask("p")
    assert ask.cache.namespace == "off"
    assert ask.cache.stats().entries == 0
