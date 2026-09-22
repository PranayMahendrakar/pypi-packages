"""Public API: chunk, chunk_documents, Chunker, and the result objects."""
import pytest

import rag_chunker
from conftest import MARKDOWN, NO_PUNCTUATION, PROSE, UNICODE


# --------------------------------------------------------------------------- #
# the easy path
# --------------------------------------------------------------------------- #
def test_three_lines_give_a_useful_answer():
    result = rag_chunker.chunk(PROSE, size=40, overlap=8)
    assert result.n_chunks > 1
    assert all(isinstance(text, str) and text for text in result.to_list())
    assert "rag-chunker" in result.summary()


@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_every_method_produces_chunks(method):
    result = rag_chunker.chunk(PROSE, size=30, overlap=5, method=method)
    assert result.n_chunks >= 1
    assert result.method == method
    assert [c.index for c in result] == list(range(result.n_chunks))


def test_result_is_a_container():
    result = rag_chunker.chunk(PROSE, size=30, overlap=5)
    assert len(result) == result.n_chunks
    assert result[0] is result.chunks[0]
    assert list(iter(result))[-1] is result.chunks[-1]


def test_result_explains_itself():
    result = rag_chunker.chunk(PROSE, size=30, overlap=5)
    summary = result.summary()
    assert "words" in summary
    assert "at most 30 words per chunk" in summary
    assert summary.isascii(), "summary must stay plain ASCII"
    dist = result.size_distribution
    assert dist["count"] == result.n_chunks
    assert dist["min"] <= dist["median"] <= dist["max"]
    assert result.mean_size > 0
    assert result.n_words > 0


def test_to_dict_is_json_safe():
    import json

    result = rag_chunker.chunk(UNICODE, size=20, overlap=4, metadata={"doc": "u"})
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["n_chunks"] == result.n_chunks
    assert payload["reproduces_source"] is True
    assert payload["chunks"][0]["metadata"] == {"doc": "u"}
    assert payload["unit"] == "words"


def test_to_list_matches_texts_and_chunk_text():
    result = rag_chunker.chunk(PROSE, size=25, overlap=5)
    assert result.to_list() == result.texts == [c.text for c in result]


def test_chunk_to_dict_round_trips_fields():
    result = rag_chunker.chunk(MARKDOWN, size=30, overlap=5, method="structural")
    first = result.chunks[0].to_dict()
    assert set(first) == {
        "index", "text", "start", "end", "tokens", "n_chars",
        "heading_path", "overlap_with", "metadata",
    }
    assert first["n_chars"] == len(result.chunks[0])


# --------------------------------------------------------------------------- #
# counting: words vs tokens
# --------------------------------------------------------------------------- #
def test_words_are_the_default_unit():
    result = rag_chunker.chunk(PROSE, size=30, overlap=5)
    assert result.unit == "words"
    assert "counted in words" in result.summary()


def test_counter_switches_the_unit_to_tokens():
    calls = []

    def counter(text):
        calls.append(text)
        return len(text.split())

    result = rag_chunker.chunk(PROSE, size=30, overlap=5, counter=counter)
    assert result.unit == "tokens"
    assert calls, "the counter must actually be used"
    assert all(c.tokens == len(c.text.split()) for c in result)


def test_counter_must_be_callable_returning_int():
    with pytest.raises(TypeError, match="callable"):
        rag_chunker.chunk(PROSE, counter="nope")
    with pytest.raises(TypeError, match="int"):
        rag_chunker.chunk(PROSE, counter=lambda text: text)


def test_a_chunk_stays_within_size():
    for method in rag_chunker.METHODS:
        result = rag_chunker.chunk(PROSE, size=30, overlap=6, method=method)
        assert max(c.tokens for c in result) <= 30, method


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_text_shorter_than_one_chunk_is_a_single_chunk():
    result = rag_chunker.chunk("Two short sentences. That is all.", size=512)
    assert result.n_chunks == 1
    assert result.chunks[0].text == "Two short sentences. That is all."


def test_empty_text_is_one_empty_chunk_not_an_empty_list():
    result = rag_chunker.chunk("")
    assert result.n_chunks == 1
    assert result.chunks[0].text == ""
    assert result.to_list() == [""]


@pytest.mark.parametrize("bad", [0, -1, -512])
def test_size_must_be_positive(bad):
    with pytest.raises(ValueError, match="size must be a positive integer"):
        rag_chunker.chunk(PROSE, size=bad, overlap=0)


@pytest.mark.parametrize("bad", ["512", 12.5, None])
def test_size_must_be_an_integer(bad):
    with pytest.raises(ValueError, match="size must be a positive integer"):
        rag_chunker.chunk(PROSE, size=bad, overlap=0)


def test_overlap_must_be_smaller_than_size_and_the_error_names_both():
    with pytest.raises(ValueError) as excinfo:
        rag_chunker.chunk(PROSE, size=100, overlap=100)
    message = str(excinfo.value)
    assert "100" in message and "size" in message and "overlap" in message

    with pytest.raises(ValueError) as excinfo:
        rag_chunker.chunk(PROSE, size=64, overlap=90)
    message = str(excinfo.value)
    assert "overlap (90)" in message and "size (64)" in message


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap must be a non-negative integer"):
        rag_chunker.chunk(PROSE, size=100, overlap=-1)


def test_unknown_method_lists_the_valid_ones():
    with pytest.raises(ValueError, match="method must be one of"):
        rag_chunker.chunk(PROSE, method="magic")


def test_text_without_sentence_punctuation_still_chunks_by_words():
    result = rag_chunker.chunk(NO_PUNCTUATION, size=20, overlap=4, method="sentence")
    assert result.n_chunks > 5
    assert max(c.tokens for c in result) <= 20
    assert result.reassemble() == NO_PUNCTUATION


def test_unicode_text_survives_intact(unicode_text):
    result = rag_chunker.chunk(unicode_text, size=15, overlap=3)
    assert result.reassemble() == unicode_text
    joined = "".join(result.to_list())
    assert "\U0001f9ea" in joined
    assert "日本語" in joined


def test_metadata_is_copied_onto_every_chunk():
    meta = {"doc_id": 7, "source": "wiki"}
    result = rag_chunker.chunk(PROSE, size=25, overlap=5, metadata=meta)
    assert result.metadata == meta
    assert all(c.metadata == meta for c in result)
    result.chunks[0].metadata["doc_id"] = 999
    assert result.chunks[1].metadata["doc_id"] == 7, "chunks must not share one dict"


def test_metadata_must_be_a_dict():
    with pytest.raises(TypeError, match="metadata must be a dict"):
        rag_chunker.chunk(PROSE, metadata=["not", "a", "dict"])


def test_bad_input_type_is_reported_clearly():
    with pytest.raises(TypeError, match="text must be a string or a path"):
        rag_chunker.chunk(12345)


def test_missing_file_is_reported_clearly(tmp_path):
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        rag_chunker.chunk(Path(tmp_path / "nope.md"))


def test_unsupported_suffix_is_reported_clearly(tmp_path):
    path = tmp_path / "data.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="not a supported document"):
        rag_chunker.chunk(path)


def test_overlap_repeats_the_previous_chunk():
    result = rag_chunker.chunk(PROSE, size=30, overlap=8, method="fixed")
    assert result.n_chunks > 2
    second = result.chunks[1]
    assert second.overlap_with == (0,)
    assert second.start < result.chunks[0].end
    repeated = PROSE[second.start:result.chunks[0].end]
    assert repeated and second.text.startswith(repeated)


def test_zero_overlap_means_no_repeated_text():
    result = rag_chunker.chunk(PROSE, size=30, overlap=0, method="fixed")
    assert all(c.overlap_with == () for c in result)
    assert "".join(result.to_list()) == PROSE


# --------------------------------------------------------------------------- #
# many documents
# --------------------------------------------------------------------------- #
def test_chunk_documents_returns_one_result_per_document():
    results = rag_chunker.chunk_documents([PROSE, MARKDOWN], size=30, overlap=5)
    assert len(results) == 2
    assert all(isinstance(r, rag_chunker.ChunkResult) for r in results)
    assert results[0].reassemble() == PROSE
    assert results[1].reassemble() == MARKDOWN


def test_chunk_documents_accepts_dicts_with_metadata():
    docs = [
        {"text": PROSE, "metadata": {"id": 1}},
        {"text": MARKDOWN, "metadata": {"id": 2}},
    ]
    results = rag_chunker.chunk_documents(docs, size=30, overlap=5, method="sentence")
    assert results[0].chunks[0].metadata == {"id": 1}
    assert results[1].chunks[0].metadata == {"id": 2}


def test_chunk_documents_accepts_paths(md_file, txt_file):
    results = rag_chunker.chunk_documents([md_file, txt_file], size=40, overlap=8)
    assert len(results) == 2
    assert results[0].origin and results[0].origin.endswith("handbook.md")
    assert results[1].source_kind == "text"


def test_chunk_documents_rejects_a_bare_string():
    with pytest.raises(TypeError, match="not a single string"):
        rag_chunker.chunk_documents(PROSE)


def test_chunk_documents_reports_a_dict_without_text():
    with pytest.raises(KeyError):
        rag_chunker.chunk_documents([{"body": "oops"}])


# --------------------------------------------------------------------------- #
# the class underneath
# --------------------------------------------------------------------------- #
def test_chunker_is_reusable_and_takes_extra_knobs():
    chunker = rag_chunker.Chunker(
        size=40, overlap=8, method="semantic", sensitivity=60, min_fill=0.25
    )
    first = chunker.chunk(PROSE)
    second = chunker.chunk(PROSE)
    assert [c.text for c in first] == [c.text for c in second], "must be deterministic"
    assert first.reassemble() == PROSE


def test_chunker_per_document_metadata_merges_with_its_own():
    chunker = rag_chunker.Chunker(size=40, overlap=8, metadata={"corpus": "docs"})
    result = chunker.chunk(PROSE, {"doc_id": 3})
    assert result.chunks[0].metadata == {"corpus": "docs", "doc_id": 3}


def test_chunker_rejects_bad_knobs():
    with pytest.raises(ValueError, match="sensitivity"):
        rag_chunker.Chunker(sensitivity=500)
    with pytest.raises(ValueError, match="min_fill"):
        rag_chunker.Chunker(min_fill=5)
    with pytest.raises(ValueError, match="source must be one of"):
        rag_chunker.Chunker(source="pdf")


def test_version_is_exported():
    assert rag_chunker.__version__ == "0.1.0"
    assert set(rag_chunker.METHOD_HELP) == set(rag_chunker.METHODS)
