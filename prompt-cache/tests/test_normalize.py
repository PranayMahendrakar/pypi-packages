"""Prompt normalisation, and the keys built from it."""

from __future__ import annotations

from prompt_cache import Cache, make_key, normalize_prompt


def test_the_four_documented_steps():
    assert normalize_prompt("a\r\nb\rc") == "a\nb\nc"
    assert normalize_prompt("line   \nnext\t\t") == "line\nnext"
    assert normalize_prompt("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_prompt("\n\n  \nreal text\n\n\n") == "real text"


def test_normalisation_changes_nothing_else():
    text = "Summarise  THIS, please: 'quoted' éà你好\U0001f680"
    assert normalize_prompt(text) == text


def test_single_blank_line_is_preserved():
    assert normalize_prompt("a\n\nb") == "a\n\nb"


def test_trivially_different_prompts_hit_the_same_entry(cache_dir):
    cache = Cache(cache_dir)
    cache.set("Explain gravity.\n\n\nIn two lines.  ", "answer")
    assert cache.get("Explain gravity.\r\n\r\nIn two lines.") == "answer"
    assert cache.get("  Explain gravity.\n\nIn two lines.") is None  # leading spaces are content


def test_normalize_false_keeps_every_byte(cache_dir):
    cache = Cache(cache_dir, normalize=False)
    cache.set("prompt   ", "answer")
    assert cache.get("prompt") is None
    assert cache.get("prompt   ") == "answer"


def test_keys_are_sha256_hex_digests(cache_dir):
    cache = Cache(cache_dir)
    key = cache.key("anything", model="m")
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_key_depends_on_namespace():
    one = make_key("p", namespace="alpha")
    two = make_key("p", namespace="beta")
    assert one != two


def test_key_depends_on_the_function_identity():
    plain = make_key("p", namespace="default")
    named = make_key("p", namespace="default", function="mod.ask")
    other = make_key("p", namespace="default", function="mod.other")
    assert len({plain, named, other}) == 3


def test_key_is_stable_across_runs():
    assert make_key("p", namespace="n", params={"a": 1}) == make_key("p", namespace="n", params={"a": 1})
