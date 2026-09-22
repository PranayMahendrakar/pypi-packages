"""Ranking: BM25 first, the optional vector blend second, and the filters."""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pytest

from document_memory import (
    BM25_B,
    BM25_K1,
    VECTOR_FLOOR,
    VECTOR_WEIGHT,
    Hit,
    Hits,
    Memory,
    tokenize,
)
from document_memory._text import bm25_term_score, idf


# --------------------------------------------------------------- the empty cases


def test_search_on_an_empty_store_returns_an_empty_list():
    """Nothing stored is not an error; it is simply no hits."""
    with Memory() as memory:
        hits = memory.search("anything at all")
        assert hits == []
        assert isinstance(hits, Hits)
        assert len(hits) == 0
        assert "no hits" in hits.summary()
        assert hits.to_dict()["count"] == 0


def test_search_that_matches_nothing_returns_an_empty_list():
    with Memory() as memory:
        memory.add("solar panels on a roof")
        assert memory.search("submarine") == []


def test_an_empty_or_blank_query_returns_an_empty_list():
    with Memory() as memory:
        memory.add("solar panels on a roof")
        assert memory.search("") == []
        assert memory.search("   \n\t ") == []
        assert memory.search("!!! ??? ...") == []


def test_k_of_zero_or_less_returns_nothing():
    with Memory() as memory:
        memory.add("solar panels on a roof")
        assert memory.search("solar", k=0) == []
        assert memory.search("solar", k=-3) == []


# ------------------------------------------------------- the ranking is real BM25


def test_a_rare_term_outranks_a_common_one():
    """The check that proves the ranking is real and not a substring count.

    Two hundred memories carry the common word; exactly one also carries the
    rare one.  A query for both must put the rare-term memory first, and by a
    wide margin - that only happens if inverse document frequency is genuinely
    being applied.
    """
    with Memory() as memory:
        memory.add_many(
            [f"the electricity meter reading number {n}" for n in range(200)]
        )
        rare = memory.add("the electricity meter and one quokka")

        hits = memory.search("quokka electricity", k=5)

        assert hits[0].id == rare
        assert set(hits[0].terms) == {"quokka", "electricity"}
        assert hits[0].bm25 > 5 * hits[1].bm25
        assert hits[1].terms == ("electricity",)


def test_the_rarer_term_carries_the_larger_idf():
    """The same property one level down, on the function itself."""
    assert idf(n_docs=1000, doc_freq=1) > idf(n_docs=1000, doc_freq=500)
    assert idf(n_docs=1000, doc_freq=500) > 0.0, "the Lucene idf never goes negative"
    assert idf(n_docs=0, doc_freq=0) == 0.0


def test_the_bm25_constants_are_the_documented_ones():
    assert (BM25_K1, BM25_B) == (1.5, 0.75)


def test_term_frequency_saturates_and_length_is_normalised():
    """k1 caps repetition; b penalises padding.  Both are what BM25 is for."""
    once = bm25_term_score(1, 10, 10.0, 1.0)
    ten_times = bm25_term_score(10, 10, 10.0, 1.0)
    assert once < ten_times < 10 * once, "k1 must saturate, not multiply"

    short = bm25_term_score(1, 5, 10.0, 1.0)
    long = bm25_term_score(1, 40, 10.0, 1.0)
    assert short > long, "b must penalise the longer document"


def test_a_term_appearing_more_often_ranks_higher_all_else_equal():
    with Memory() as memory:
        once = memory.add("solar panel and some other words to pad the memory out")
        twice = memory.add("solar panel solar again with other words to pad it out")
        hits = memory.search("solar")
        assert hits[0].id == twice
        assert hits[1].id == once


def test_scores_are_normalised_but_bm25_is_the_comparable_number():
    with Memory() as memory:
        memory.add_many(["solar roof panel", "solar garden lamp", "unrelated note"])
        hits = memory.search("solar panel")
        assert hits[0].score == pytest.approx(1.0)
        assert all(0.0 < hit.score <= 1.0 for hit in hits)
        assert hits[0].bm25 >= hits[-1].bm25
        assert [hit.score for hit in hits] == sorted(
            (hit.score for hit in hits), reverse=True
        )


def test_k_limits_the_number_of_hits():
    with Memory() as memory:
        memory.add_many([f"solar note {n}" for n in range(20)])
        assert len(memory.search("solar", k=3)) == 3
        assert len(memory.search("solar", k=100)) == 20


# ------------------------------------------------------- every hit explains itself


def test_a_hit_explains_itself():
    with Memory() as memory:
        memory.add("Solar panels turn sunlight into electricity.", source="guide.md")
        hit = memory.search("solar electricity")[0]

        assert isinstance(hit, Hit)
        assert set(hit.terms) == {"solar", "electricity"}
        assert "bm25" in hit.why()
        assert "solar" in hit.why()
        assert "guide.md" in hit.summary()
        assert hit.similarity is None, "no embed means no cosine to report"

        payload = hit.to_dict()
        assert payload["id"] == hit.id
        assert payload["terms"] == list(hit.terms)
        assert payload["why"] == hit.why()
        assert payload["when"].startswith("20")

        record = hit.to_record()
        assert (record.id, record.text, record.source) == (hit.id, hit.text, hit.source)


def test_the_result_set_describes_itself():
    with Memory() as memory:
        memory.add_many(["solar one", "solar two"])
        hits = memory.search("solar")
        assert "2 hits" in hits.summary()
        assert "'solar'" in hits.summary()
        payload = hits.to_dict()
        assert payload["count"] == 2
        assert payload["mode"] == "lexical BM25"
        assert len(payload["items"]) == 2
        assert len(hits.to_list()) == 2


# --------------------------------------------------------------------- filtering


def test_where_filters_on_metadata_including_nested_keys_and_lists():
    with Memory() as memory:
        memory.add(
            "solar roof array",
            source="a.md",
            metadata={"project": "roof", "user": {"name": "ada"}},
        )
        memory.add(
            "solar garden lamp",
            source="b.md",
            metadata={"project": "garden", "user": {"name": "bob"}},
        )

        assert [h.source for h in memory.search("solar", where={"project": "roof"})] == [
            "a.md"
        ]
        both = memory.search("solar", where={"project": ["roof", "garden"]})
        assert {h.source for h in both} == {"a.md", "b.md"}
        assert [h.source for h in memory.search("solar", where={"user.name": "bob"})] == [
            "b.md"
        ]
        assert memory.search("solar", where={"project": "attic"}) == []


def test_where_accepts_a_predicate_and_matches_source_and_kind():
    with Memory() as memory:
        memory.add("solar roof array", source="a.md", metadata={"watts": 400})
        memory.add("solar garden lamp", source="b.md", metadata={"watts": 5})
        memory.remember("user", "a solar question")

        big = memory.search("solar", where={"watts": lambda v: bool(v) and v > 100})
        assert [hit.source for hit in big] == ["a.md"]
        assert [h.source for h in memory.search("solar", where={"source": "b.md"})] == [
            "b.md"
        ]
        turns = memory.search("solar", where={"kind": "turn"})
        assert [hit.kind for hit in turns] == ["turn"]
        assert [h.metadata["role"] for h in memory.search("solar", where={"role": "user"})] == [
            "user"
        ]


def test_since_keeps_only_memories_at_or_after_a_moment():
    with Memory() as memory:
        memory.add("old solar note", timestamp="2020-01-01")
        recent = memory.add("new solar note", timestamp="2026-01-01")

        assert [hit.id for hit in memory.search("solar", since="2025-01-01")] == [recent]
        assert len(memory.search("solar", since=dt.date(2019, 1, 1))) == 2
        assert memory.search("solar", since="2030-01-01") == []


def test_where_and_since_combine():
    with Memory() as memory:
        memory.add("old solar", timestamp="2020-01-01", metadata={"keep": True})
        memory.add("new solar", timestamp="2026-01-01", metadata={"keep": False})
        wanted = memory.add("new solar kept", timestamp="2026-01-01", metadata={"keep": True})
        hits = memory.search("solar", where={"keep": True}, since="2025-01-01")
        assert [hit.id for hit in hits] == [wanted]


def test_a_bad_where_says_so():
    with Memory() as memory:
        memory.add("solar")
        with pytest.raises(ValueError, match="where must be a dict"):
            memory.search("solar", where=["project"])


# ------------------------------------------------------------- the vector blend


def _bag_embed(dimension: int = 24):
    """A tiny deterministic hashing embedder - no model, no download."""

    def embed(texts):
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                matrix[row, sum(map(ord, token)) % dimension] += 1.0
        return matrix

    return embed


def test_the_blend_weight_is_a_documented_constant():
    assert VECTOR_WEIGHT == 0.5
    assert 0.0 <= VECTOR_FLOOR <= 1.0


def test_vectors_are_stored_and_cosine_is_blended_into_the_score():
    with Memory(embed=_bag_embed()) as memory:
        memory.add("cats sleep on warm laptops")
        memory.add("dogs run around the park")

        hits = memory.search("cats")
        assert hits[0].text.startswith("cats")
        assert hits[0].similarity is not None
        assert 0.0 <= hits[0].similarity <= 1.0
        assert hits.info["mode"] == "BM25 + cosine blend"
        assert hits.info["vector_weight"] == VECTOR_WEIGHT

        # score = (1 - w) * normalised_bm25 + w * cosine, and the best is 1.0 lexically
        expected = (1.0 - VECTOR_WEIGHT) * 1.0 + VECTOR_WEIGHT * hits[0].similarity
        assert hits[0].score == pytest.approx(expected, abs=1e-5)


def test_the_blend_weight_can_be_moved_per_store():
    embed = _bag_embed()
    with Memory(embed=embed) as memory:
        memory.add("cats sleep on warm laptops")
        memory.add("dogs run around the park")
        memory.vector_weight = 0.0
        lexical = memory.search("cats")[0]
        assert lexical.score == pytest.approx(1.0), "weight 0 is pure BM25"
        memory.vector_weight = 1.0
        semantic = memory.search("cats")[0]
        assert semantic.score == pytest.approx(semantic.similarity, abs=1e-5)


def test_vectors_are_not_stored_without_an_embed_function():
    with Memory() as memory:
        memory.add("no vector here")
        row = memory._connection.execute("SELECT vector FROM memories").fetchone()
        assert row[0] is None
        assert memory.search("vector")[0].similarity is None


def test_vectors_survive_a_reopen(tmp_path):
    store = str(tmp_path / "vectors.db")
    embed = _bag_embed()
    with Memory(store, embed=embed) as memory:
        memory.add("cats sleep on warm laptops")
    with Memory(store, embed=embed) as reopened:
        hit = reopened.search("cats")[0]
        assert hit.similarity is not None and hit.similarity > 0.0


def test_a_broken_embed_is_reported_clearly():
    with Memory(embed=lambda texts: [[1.0, 2.0, 3.0]]) as memory:
        with pytest.raises(ValueError, match="one vector per text"):
            memory.add_many(["one", "two"])

    with Memory(embed=lambda texts: [[float("nan"), 1.0]] * len(texts)) as memory:
        with pytest.raises(ValueError, match="NaN"):
            memory.add("anything")

    with pytest.raises(ValueError, match="embed must be a callable"):
        Memory(embed="not callable")


def test_changing_vector_width_mid_store_is_refused_with_a_useful_message(tmp_path):
    store = str(tmp_path / "dims.db")
    with Memory(store, embed=_bag_embed(8)) as memory:
        memory.add("first model")
    with Memory(store, embed=_bag_embed(16)) as memory:
        with pytest.raises(ValueError, match="dimensional"):
            memory.add("second model")


# ------------------------------------------------------------------ performance


def test_ten_thousand_documents_search_in_well_under_a_second():
    vocabulary = [f"w{n}" for n in range(400)]
    documents = []
    for n in range(10_000):
        picks = [vocabulary[(n * 7 + step * 13) % len(vocabulary)] for step in range(20)]
        documents.append(" ".join(picks))
    documents.append("the one and only quokka sighting")

    with Memory() as memory:
        memory.add_many(documents)
        assert memory.count() == 10_001

        slowest = 0.0
        for query in ("quokka", "w17", "w17 w23 w99", "w5 quokka w300"):
            start = time.perf_counter()
            hits = memory.search(query, k=5)
            slowest = max(slowest, time.perf_counter() - start)
            assert len(hits) > 0

        assert slowest < 0.25, f"slowest search took {slowest:.3f}s"
