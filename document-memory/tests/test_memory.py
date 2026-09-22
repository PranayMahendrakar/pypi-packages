"""The store itself: creating, reopening, updating, deleting, namespaces."""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from document_memory import Memory, Record, Records, open_memory, tokenize


def test_quickstart_from_the_readme(tmp_path):
    """The exact shape of the README quickstart, checked end to end."""
    store = str(tmp_path / "notes.db")
    with Memory(store) as memory:
        memory.add("Solar panels turn sunlight into electricity.", source="guide.md")
        memory.add("Wind turbines turn moving air into electricity.", source="guide.md")
        memory.remember("user", "which one works on a cloudy roof?")
        hits = memory.search("solar")
        assert hits[0].text.startswith("Solar panels")
        assert "solar" in hits[0].summary().lower()
        assert memory.context("electricity", budget=40)


def test_store_is_created_on_demand_including_parent_folders(tmp_path):
    store = str(tmp_path / "deep" / "nested" / "notes.db")
    assert not os.path.exists(store)
    memory = Memory(store)
    try:
        memory.add("created on demand")
        assert os.path.exists(store)
    finally:
        memory.close()


def test_reopening_finds_everything_added_before(tmp_path):
    store = str(tmp_path / "notes.db")
    with Memory(store) as first:
        ids = first.add_many(["alpha document", "beta document", "gamma document"])
        first.remember("user", "and a conversation turn")
    with Memory(store) as second:
        assert second.count() == 4
        assert second.get(ids[0]).text == "alpha document"
        assert second.search("beta")[0].id == ids[1]
        assert [turn.content for turn in second.history()] == ["and a conversation turn"]


def test_in_memory_store_needs_no_path():
    with Memory() as memory:
        memory.add("nothing touches the disk")
        assert memory.count() == 1
        assert memory.path is None
        assert ":memory:" in repr(memory)


def test_add_returns_the_id_and_get_round_trips_every_field():
    with Memory() as memory:
        when = dt.datetime(2026, 3, 4, 5, 6, 7, tzinfo=dt.timezone.utc)
        identifier = memory.add(
            "a document about roofs",
            id="doc-1",
            metadata={"project": "roof"},
            source="guide.md",
            timestamp=when,
        )
        assert identifier == "doc-1"
        record = memory.get("doc-1")
        assert isinstance(record, Record)
        assert record.text == "a document about roofs"
        assert record.metadata == {"project": "roof"}
        assert record.source == "guide.md"
        assert record.timestamp == pytest.approx(when.timestamp())
        assert record.when.startswith("2026-03-04T05:06:07")
        assert record.kind == "document"


def test_get_of_an_unknown_id_is_none():
    with Memory() as memory:
        assert memory.get("never-stored") is None


def test_delete_of_a_missing_id_returns_false_instead_of_raising():
    with Memory() as memory:
        memory.add("present", id="here")
        assert memory.delete("not-here") is False
        assert memory.delete("here") is True
        assert memory.delete("here") is False
        assert memory.count() == 0


def test_deleting_also_removes_the_document_from_the_index():
    with Memory() as memory:
        memory.add("quokka sighting", id="q")
        assert memory.search("quokka")
        memory.delete("q")
        assert memory.search("quokka") == []


def test_adding_the_same_id_twice_replaces_rather_than_duplicates():
    with Memory() as memory:
        memory.add("first version about otters", id="doc")
        memory.add("second version about badgers", id="doc")
        assert memory.count() == 1
        assert memory.get("doc").text == "second version about badgers"
        assert memory.search("otters") == []
        assert len(memory.search("badgers")) == 1


def test_add_many_accepts_strings_dicts_and_records():
    with Memory() as memory:
        carried = Record(id="from-record", text="a record carried straight back in")
        ids = memory.add_many(
            [
                "a bare string",
                {"text": "a dict", "id": "d1", "metadata": {"n": 1}, "source": "s"},
                carried,
            ]
        )
        assert len(ids) == 3
        assert ids[1] == "d1"
        assert memory.get("d1").metadata == {"n": 1}
        assert memory.get("from-record").text == "a record carried straight back in"


def test_add_many_of_nothing_is_an_empty_list():
    with Memory() as memory:
        assert memory.add_many([]) == []
        assert memory.count() == 0


def test_metadata_round_trips_exactly_including_nested_dicts_and_unicode():
    payload = {
        "user": {"name": "Ada Lovelace", "tags": ["math", "engine"]},
        "notes": "cafe - naive; the memo is written in Japanese; emoji ok",
        "unicode": "café — naïve; 日本語; \U0001f389",
        "depth": {"a": {"b": {"c": [1, 2.5, True, None, "ünïcode"]}}},
        "ключ": "значение",
        "count": 0,
        "flag": False,
    }
    with Memory() as memory:
        memory.add("a memory with rich metadata", id="m", metadata=payload)
        back = memory.get("m").metadata
        assert back == payload
        assert back["depth"]["a"]["b"]["c"][4] == "ünïcode"
        assert json.loads(json.dumps(back, ensure_ascii=False)) == payload


def test_unicode_text_is_stored_and_searchable():
    japanese = "日本語のメモ: 太陽光パネル"
    with Memory() as memory:
        memory.add(japanese, id="jp")
        memory.add("une note en français sur le café", id="fr")
        assert memory.get("jp").text == japanese
        assert memory.search("café")[0].id == "fr"
        assert memory.search("パネル")[0].id == "jp"


def test_metadata_that_json_cannot_carry_raises_a_clear_error():
    with Memory() as memory:
        with pytest.raises(ValueError, match="JSON-serialisable"):
            memory.add("bad", metadata={"when": object()})
        with pytest.raises(ValueError, match="must be a dict"):
            memory.add("bad", metadata=[1, 2, 3])


def test_update_changes_text_metadata_or_both():
    with Memory() as memory:
        memory.add("original text about otters", id="u", metadata={"a": 1, "b": 2})
        updated = memory.update("u", text="new text about badgers")
        assert updated.text == "new text about badgers"
        assert updated.metadata == {"a": 1, "b": 2}
        assert memory.search("otters") == []
        assert memory.search("badgers")[0].id == "u"

        merged = memory.update("u", metadata={"b": 3, "c": 4})
        assert merged.metadata == {"a": 1, "b": 3, "c": 4}
        assert merged.text == "new text about badgers"

        dropped = memory.update("u", metadata={"a": None})
        assert dropped.metadata == {"b": 3, "c": 4}


def test_update_of_an_unknown_id_is_none_and_no_arguments_is_a_read():
    with Memory() as memory:
        assert memory.update("nope", text="x") is None
        memory.add("unchanged", id="u")
        assert memory.update("u").text == "unchanged"


def test_recent_is_newest_first_and_filters_by_source():
    with Memory() as memory:
        memory.add("oldest", id="a", source="one", timestamp="2026-01-01")
        memory.add("middle", id="b", source="two", timestamp="2026-02-01")
        memory.add("newest", id="c", source="one", timestamp="2026-03-01")
        records = memory.recent()
        assert isinstance(records, Records)
        assert [r.id for r in records] == ["c", "b", "a"]
        assert [r.id for r in memory.recent(source="one")] == ["c", "a"]
        assert memory.recent(0) == []
        assert len(memory.recent(2)) == 2


def test_count_and_len_agree_and_clear_empties_the_namespace():
    with Memory() as memory:
        memory.add_many(["one", "two", "three"])
        assert memory.count() == len(memory) == 3
        assert memory.clear() == 3
        assert memory.count() == 0
        assert memory.search("one") == []


def test_namespaces_are_independent_and_star_clears_them_all(tmp_path):
    store = str(tmp_path / "shared.db")
    with Memory(store, namespace="ada") as ada, Memory(store, namespace="bob") as bob:
        ada.add("a private note about quokkas")
        bob.add("a private note about wombats")
        assert ada.count() == bob.count() == 1
        assert ada.search("wombats") == []
        assert bob.search("quokkas") == []
        assert ada.clear("*") == 2
        assert bob.count() == 0


def test_timestamps_accept_datetime_date_iso_string_and_epoch():
    with Memory() as memory:
        memory.add("a", id="a", timestamp=dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc))
        memory.add("b", id="b", timestamp=dt.date(2026, 1, 3))
        memory.add("c", id="c", timestamp="2026-01-04T05:06:07Z")
        memory.add("d", id="d", timestamp=1767225600.0)
        assert memory.get("a").when.startswith("2026-01-02T03:04")
        assert memory.get("b").when.startswith("2026-01-03T00:00")
        assert memory.get("c").when.startswith("2026-01-04T05:06:07")
        assert memory.get("d").timestamp == 1767225600.0
        with pytest.raises(ValueError, match="ISO 8601"):
            memory.add("e", timestamp="the day before yesterday")


def test_bad_arguments_raise_value_errors_that_say_what_is_wrong():
    with pytest.raises(ValueError, match="namespace"):
        Memory(namespace="  ")
    with pytest.raises(ValueError, match="embed must be a callable"):
        Memory(embed="not callable")
    with Memory() as memory:
        with pytest.raises(ValueError, match="text of type int"):
            memory.add_many([{"text": 7}])
        with pytest.raises(ValueError, match="no 'text' key"):
            memory.add_many([{"body": "wrong key"}])
        with pytest.raises(ValueError, match="item 0 must be"):
            memory.add_many([7])
        with pytest.raises(ValueError, match="empty id"):
            memory.add_many([{"text": "x", "id": "   "}])
        with pytest.raises(ValueError, match="role must be"):
            memory.remember("", "no role")


def test_empty_text_is_storable_and_simply_never_matches():
    with Memory() as memory:
        memory.add("", id="blank")
        assert memory.get("blank").text == ""
        assert memory.count() == 1
        assert memory.search("anything") == []


def test_a_single_memory_store_still_ranks():
    with Memory() as memory:
        memory.add("the only document in the store")
        hits = memory.search("only")
        assert len(hits) == 1
        assert hits[0].score == 1.0


def test_close_is_idempotent_and_using_a_closed_store_says_so():
    memory = Memory()
    memory.add("something")
    memory.close()
    memory.close()
    with pytest.raises(ValueError, match="closed"):
        memory.count()


def test_open_memory_is_the_same_thing_spelled_as_a_verb(tmp_path):
    store = str(tmp_path / "verb.db")
    with open_memory(store, namespace="team") as memory:
        assert isinstance(memory, Memory)
        assert memory.namespace == "team"
        memory.add("stored through open_memory")
    with Memory(store, namespace="team") as memory:
        assert memory.count() == 1


def test_summary_and_repr_describe_the_store(tmp_path):
    store = str(tmp_path / "notes.db")
    with Memory(store) as memory:
        memory.add_many(["one", "two"])
        memory.remember("user", "hello")
        text = memory.summary()
        assert "3 memories" in text
        assert "1 conversation turns" in text
        assert "BM25" in text
        assert text.isascii()
        assert "notes.db" in repr(memory)


def test_tokenize_is_exported_and_shows_what_gets_indexed():
    assert tokenize("Solar panels, 240 watts!") == ["solar", "panels", "240", "watts"]
    assert tokenize("") == []
    assert tokenize("日本語") == ["日", "本", "語"]
    assert tokenize("café") == ["café"]
