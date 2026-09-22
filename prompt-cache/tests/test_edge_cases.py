"""The awkward cases: no folder yet, a corrupt file, unicode, empty prompts."""

from __future__ import annotations

import sqlite3
import time

import pytest

from prompt_cache import Cache
from prompt_cache._store import is_corruption

UNICODE_PROMPTS = [
    "Explique-moi la gravite, s'il te plait",
    "重力について教えて",
    "شرح الجاذبية",
    "Гравитация?",
    "why \U0001f680 falls \U0001f4c9 back \U0001f30d",
    "combining: éclair",
    "zero width: a​b",
]


def test_the_cache_directory_is_created_on_demand(tmp_path):
    target = tmp_path / "deep" / "nested" / "cache"
    assert not target.exists()
    cache = Cache(target)
    assert not target.exists()  # building a Cache touches nothing
    cache.set("p", "v")
    assert (target / "cache.db").is_file()


def test_a_db_file_path_is_used_as_given(tmp_path):
    target = tmp_path / "somewhere" / "answers.sqlite3"
    cache = Cache(target)
    cache.set("p", "v")
    assert target.is_file()
    assert cache.path == target


@pytest.mark.parametrize("prompt", UNICODE_PROMPTS)
def test_unicode_prompts_round_trip(cache_dir, prompt):
    cache = Cache(cache_dir)
    cache.set(prompt, "answer for " + prompt)
    assert cache.get(prompt) == "answer for " + prompt
    assert cache.entries()[0]["preview"]


def test_unicode_prompts_stay_distinct(cache_dir):
    cache = Cache(cache_dir)
    for index, prompt in enumerate(UNICODE_PROMPTS):
        cache.set(prompt, index)
    for index, prompt in enumerate(UNICODE_PROMPTS):
        assert cache.get(prompt) == index
    assert cache.stats().entries == len(UNICODE_PROMPTS)


def test_an_empty_prompt_is_a_valid_key(cache_dir):
    cache = Cache(cache_dir)
    cache.set("", "empty")
    assert cache.get("") == "empty"
    assert cache.get("   \n\n  ") == "empty"  # normalises to the same thing


def test_a_corrupt_database_is_rebuilt_not_raised(cache_dir):
    first = Cache(cache_dir)
    first.set("p", "v")
    first.close()
    db = cache_dir / "cache.db"
    db.write_bytes(b"this is definitely not a sqlite database" * 20)

    second = Cache(cache_dir)
    assert second.get("p") is None  # the old entry is gone, but nothing raised
    second.set("p", "v2")
    assert second.get("p") == "v2"
    assert (cache_dir / "cache.db.corrupt").is_file()


def test_a_database_corrupted_between_calls_is_rebuilt(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    cache.close()
    (cache_dir / "cache.db").write_bytes(b"garbage")
    assert cache.get("p") is None
    cache.set("p", "again")
    assert cache.get("p") == "again"


def test_a_locked_database_error_is_not_mistaken_for_corruption():
    assert is_corruption(sqlite3.OperationalError("database is locked")) is False
    assert is_corruption(sqlite3.DatabaseError("file is not a database")) is True
    assert is_corruption(ValueError("nothing to do with sqlite")) is False


def test_a_missing_key_is_fast(cache_dir):
    cache = Cache(cache_dir)
    cache.set("warm the file up", "v")
    started = time.time()
    for index in range(2000):
        assert cache.get("no such prompt {0}".format(index)) is None
    assert time.time() - started < 10.0
    assert cache.stats().entries == 1


def test_entries_lists_what_is_stored(cache_dir, clock):
    cache = Cache(cache_dir)
    for index in range(3):
        clock.advance(1)
        cache.set("prompt {0}".format(index), {"index": index})
    rows = cache.entries(limit=2)
    assert len(rows) == 2
    assert rows[0]["preview"] == "prompt 2"  # most recently used first
    assert set(rows[0]) == {
        "key",
        "preview",
        "size",
        "created",
        "accessed",
        "expires",
        "hits",
        "format",
    }


def test_entries_limit_zero_returns_nothing(cache_dir):
    cache = Cache(cache_dir)
    cache.set("p", "v")
    assert cache.entries(limit=0) == []


def test_two_caches_on_one_file_see_each_others_writes(cache_dir):
    writer = Cache(cache_dir)
    reader = Cache(cache_dir)
    writer.set("p", "v")
    assert reader.get("p") == "v"


def test_reopening_the_same_folder_keeps_the_entries(cache_dir):
    with Cache(cache_dir) as first:
        first.set("p", "v")
    with Cache(cache_dir) as second:
        assert second.get("p") == "v"
