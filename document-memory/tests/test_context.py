"""Conversation turns, and the prompt context that is packed to a budget."""
from __future__ import annotations

import pytest

from document_memory import CONTEXT_HEADER, Memory, Turn, Turns


def _stock(memory: Memory, count: int = 40) -> None:
    for n in range(count):
        memory.add(
            f"Solar panel note {n}: the array on the south roof produces power "
            f"and the inverter logged reading number {n} this morning.",
            source=f"note-{n}.md",
        )


def _words(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------- the budget


@pytest.mark.parametrize(
    "budget", [1, 2, 3, 4, 5, 6, 7, 8, 11, 17, 29, 50, 120, 400, 2000]
)
def test_context_never_exceeds_the_budget(budget):
    """The whole returned string, header and blank lines included, fits."""
    with Memory() as memory:
        _stock(memory)
        text = memory.context("solar roof inverter", budget=budget)
        assert _words(text) <= budget, f"budget {budget} produced {_words(text)} words"


@pytest.mark.parametrize("budget", [4, 9, 25, 80, 300])
def test_context_never_exceeds_a_budget_measured_by_a_custom_counter(budget):
    """The same guarantee when the caller counts tokens their own way."""
    counter = len  # characters: a harsher unit than words

    with Memory() as memory:
        _stock(memory)
        text = memory.context("solar roof inverter", budget=budget, counter=counter)
        assert counter(text) <= budget


def test_context_actually_fills_the_budget_it_is_given():
    """Not exceeding the budget is easy; using it is the point."""
    with Memory() as memory:
        _stock(memory)
        small = memory.context("solar roof", budget=40)
        large = memory.context("solar roof", budget=400)
        assert 0 < _words(small) <= 40
        assert _words(small) > 20, "a 40-word budget should be mostly used"
        assert _words(large) > _words(small)
        assert _words(large) <= 400


def test_context_is_formatted_ready_to_paste():
    with Memory() as memory:
        memory.add("Solar panels turn sunlight into electricity.", source="guide.md")
        text = memory.context("solar", budget=200)
        assert text.startswith(CONTEXT_HEADER.format(query="solar"))
        assert "guide.md" in text
        assert "Solar panels turn sunlight into electricity." in text
        assert "\n\n" in text


def test_context_puts_the_best_match_first():
    with Memory() as memory:
        memory.add("a note about garden gnomes")
        memory.add("solar inverter maintenance schedule")
        text = memory.context("solar inverter", budget=100)
        assert text.index("inverter") < len(text)
        assert "solar inverter maintenance" in text


def test_a_budget_too_small_for_a_whole_memory_truncates_rather_than_giving_up():
    with Memory() as memory:
        memory.add("solar " + " ".join(f"word{n}" for n in range(300)))
        text = memory.context("solar", budget=25)
        assert 0 < _words(text) <= 25
        assert "..." in text, "the truncated memory should say it was cut"


def test_a_budget_too_small_for_even_the_header_returns_an_empty_string():
    with Memory() as memory:
        memory.add("solar panels")
        assert memory.context("solar", budget=1) == ""
        assert memory.context("solar", budget=0) == ""
        assert memory.context("solar", budget=-5) == ""


def test_context_of_an_empty_store_or_a_query_nothing_matches_is_empty():
    with Memory() as memory:
        assert memory.context("solar", budget=500) == ""
        memory.add("completely unrelated text")
        assert memory.context("solar", budget=500) == ""


def test_a_counter_that_is_not_callable_says_so():
    with Memory() as memory:
        memory.add("solar")
        with pytest.raises(ValueError, match="counter must be a callable"):
            memory.context("solar", budget=50, counter="len")


# ------------------------------------------------------------- conversations


def test_remember_stores_the_role_in_the_metadata():
    with Memory() as memory:
        identifier = memory.remember("user", "how do panels do in winter?")
        record = memory.get(identifier)
        assert record.metadata["role"] == "user"
        assert record.kind == "turn"
        assert record.text == "how do panels do in winter?"


def test_remember_carries_extra_metadata_source_and_timestamp():
    with Memory() as memory:
        identifier = memory.remember(
            "assistant",
            "less light, so less power",
            session="s1",
            source="chat.log",
            timestamp="2026-02-03T04:05:06",
        )
        record = memory.get(identifier)
        assert record.metadata == {"role": "assistant", "session": "s1"}
        assert record.source == "chat.log"
        assert record.when.startswith("2026-02-03T04:05:06")


def test_history_returns_the_most_recent_turns_oldest_first():
    with Memory() as memory:
        for n in range(6):
            memory.remember("user" if n % 2 == 0 else "assistant", f"turn {n}",
                            timestamp=1_700_000_000 + n)
        turns = memory.history(4)
        assert isinstance(turns, Turns)
        assert all(isinstance(turn, Turn) for turn in turns)
        assert [turn.content for turn in turns] == ["turn 2", "turn 3", "turn 4", "turn 5"]
        assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]


def test_history_ignores_documents_and_handles_an_empty_conversation():
    with Memory() as memory:
        memory.add("a document, not a turn")
        assert memory.history() == []
        memory.remember("user", "hello")
        assert [turn.content for turn in memory.history()] == ["hello"]
        assert memory.history(0) == []


def test_a_turn_explains_itself():
    with Memory() as memory:
        memory.remember("user", "which one works on a cloudy roof?")
        turn = memory.history()[0]
        assert "user:" in turn.summary()
        assert turn.text == turn.content
        payload = turn.to_dict()
        assert payload["role"] == "user"
        assert payload["content"] == turn.content
        assert payload["when"].startswith("20")
        assert "1 turn, oldest first" in memory.history().summary()


def test_turns_rank_alongside_documents_in_ordinary_search():
    with Memory() as memory:
        memory.add("solar panels on the south roof")
        memory.remember("user", "is the solar array working on a cloudy roof?")
        kinds = {hit.kind for hit in memory.search("solar roof", k=5)}
        assert kinds == {"document", "turn"}


def test_turns_appear_in_context_labelled_by_role():
    with Memory() as memory:
        memory.remember("user", "how does the solar inverter behave in winter?")
        text = memory.context("solar inverter", budget=200)
        assert "user" in text
        assert "winter" in text


def test_an_empty_role_is_refused():
    with Memory() as memory:
        with pytest.raises(ValueError, match="role must be a non-empty string"):
            memory.remember("", "no role given")
        with pytest.raises(ValueError, match="role must be a non-empty string"):
            memory.remember("   ", "no role given")
