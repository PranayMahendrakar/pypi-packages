"""Regressions for the issues an independent review found in 0.1.0.

Each test names the behaviour that was wrong, so a reappearance is immediately
recognisable.
"""
import copy
import json
from pathlib import Path

import pytest

import rag_chunker
from rag_chunker._semantic import analyse_shifts, boundary_similarities, topic_shifts
from conftest import HTML, MARKDOWN, NO_PUNCTUATION, PROSE, UNICODE

# A page whose sections are all far shorter than the default overlap of 64
# words: the shape (API reference, FAQ, changelog, glossary) that used to make
# every chunk a strict prefix of the next.
SHORT_SECTIONS = "# Reference\n\n" + "".join(
    "## %s\n\n%s\n\n" % (name, body)
    for name, body in [
        ("install", "Install it with pip."),
        ("usage", "Import the module and call the function."),
        ("options", "Every option has a sensible default."),
        ("logging", "Logging goes to the standard logger."),
        ("errors", "Errors are plain ValueErrors."),
        ("testing", "Run pytest from the project root."),
        ("licence", "The project is MIT licensed."),
    ]
)

HANDBOOK = "# Handbook\n\n" + "".join(
    "## Section %d\n\n%s\n\n"
    % (i, "This section covers one narrow topic. It is deliberately brief.")
    for i in range(60)
)

DOCUMENTS = {
    "short_sections": SHORT_SECTIONS,
    "handbook": HANDBOOK,
    "markdown": MARKDOWN,
    "prose": PROSE,
    "html": HTML,
    "unicode": UNICODE,
    "no_punctuation": NO_PUNCTUATION,
}


def _contained_pairs(result):
    """Pairs ``(i, j)`` where chunk i's character span sits inside chunk j's."""
    spans = [(c.start, c.end) for c in result.chunks]
    return [
        (i, j)
        for i, (a0, a1) in enumerate(spans)
        for j, (b0, b1) in enumerate(spans)
        if i != j and b0 <= a0 and a1 <= b1
    ]


# --------------------------------------------------------------------------- #
# blocker: structural at the library defaults emitted chunks that were strict
# prefixes of one another, because the overlap walk-back ran past several
# earlier group boundaries and re-absorbed them.
# --------------------------------------------------------------------------- #
def test_structural_at_default_size_and_overlap_does_not_repeat_whole_chunks():
    result = rag_chunker.chunk(SHORT_SECTIONS, method="structural")  # 512 / 64
    assert (result.size, result.overlap) == (512, 64)
    starts = [c.start for c in result.chunks]
    assert starts[0] == 0
    assert starts == sorted(set(starts)), "each chunk must start further in than the last"
    assert not any(
        chunk.text in other.text
        for chunk in result.chunks
        for other in result.chunks
        if chunk is not other
    ), "no chunk may be wholly contained in another"
    assert result.chunks[-1].text != result.text, (
        "the last chunk must not be the whole document"
    )
    # The old bug stored this 330-character document eight times over.
    assert sum(c.n_chars for c in result.chunks) < 2 * len(result.text)
    assert result.reassemble() == result.text


def test_structural_overlap_only_repeats_the_chunk_directly_before():
    result = rag_chunker.chunk(SHORT_SECTIONS, method="structural")
    assert result.chunks[0].overlap_with == ()
    for chunk in result.chunks[1:]:
        assert chunk.overlap_with == (chunk.index - 1,), (
            "a chunk may only repeat the chunk immediately before it"
        )
        previous = result.chunks[chunk.index - 1]
        assert chunk.start > previous.start, (
            "the overlap must never reach past the previous chunk's own start"
        )


def test_a_long_document_of_short_sections_is_not_duplicated():
    result = rag_chunker.chunk(HANDBOOK, method="structural")
    assert result.n_chunks > 20
    assert _contained_pairs(result) == []
    assert sum(c.n_chars for c in result.chunks) < 2 * len(result.text)
    assert result.reassemble() == HANDBOOK


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
@pytest.mark.parametrize("size,overlap", [(512, 64), (200, 50), (60, 10), (25, 5)])
@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_no_chunk_is_ever_contained_in_another(method, size, overlap, name):
    text = DOCUMENTS[name]
    result = rag_chunker.chunk(text, size=size, overlap=overlap, method=method)
    assert _contained_pairs(result) == [], (
        f"{method} at {size}/{overlap} on {name} indexes the same span twice"
    )


def test_the_packages_own_readme_structures_cleanly_at_the_defaults():
    """The review chunked this very README and got chunks contained in others."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    if not readme.is_file():  # not shipped inside the wheel
        pytest.skip("README not present")
    result = rag_chunker.chunk(readme.read_text(encoding="utf-8"), method="structural")
    assert result.n_chunks > 1
    assert _contained_pairs(result) == []
    assert sum(c.n_chars for c in result.chunks) < 2 * len(result.text)
    assert result.reassemble() == result.text


def test_overlap_start_never_walks_past_the_previous_group():
    from rag_chunker._segment import Unit, overlap_start

    text = "a b c d e f g h"
    units = [Unit(i * 2, i * 2 + 2, False, 1) for i in range(8)]
    cost = lambda piece: len(piece.split())
    # Groups start at 0, 2, 4, 6. An overlap of 5 would happily swallow four
    # earlier units; the floor keeps the walk inside the group starting at 2,
    # so the chunk repeats part of the previous chunk and never the whole of it.
    start = overlap_start(units, 4, 5, text, cost, floor=2)
    assert start > units[2].start
    assert start == units[3].start
    # The floor moves the wall: a lower floor lets the walk reach further back,
    # and even at floor=0 the very first unit is never consumed, so no chunk can
    # ever grow to contain the one before it.
    assert overlap_start(units, 4, 5, text, cost, floor=0) == units[1].start
    assert overlap_start(units, 4, 5, text, cost) == units[1].start


# --------------------------------------------------------------------------- #
# major: a flat similarity series was read as a topic shift at every boundary
# --------------------------------------------------------------------------- #
def test_identical_units_are_maximally_similar_and_yield_no_topic_shift():
    units = ["the same sentence repeated here"] * 12
    assert boundary_similarities(units) == pytest.approx([1.0] * 11)
    shifts, _sims = topic_shifts(units)
    assert shifts == set(), "identical text has no topic shift by construction"
    _shifts, _series, reason = analyse_shifts(units)
    assert reason and "equally similar" in reason


def test_units_sharing_no_vocabulary_also_yield_no_topic_shift():
    units = ["alpha%d beta%d" % (i, i) for i in range(12)]
    assert boundary_similarities(units) == pytest.approx([0.0] * 11)
    assert topic_shifts(units)[0] == set()


def test_semantic_on_repeated_text_matches_sentence_packing_and_warns():
    text = "the same sentence repeated here. " * 60
    semantic = rag_chunker.chunk(text, size=100, overlap=20, method="semantic")
    sentence = rag_chunker.chunk(text, size=100, overlap=20, method="sentence")
    assert semantic.to_list() == sentence.to_list(), (
        "with no topic shift to find, semantic must fall back to packing sentences"
    )
    assert semantic.n_chunks == 4
    assert semantic.size_distribution["mean"] == 90.0
    assert semantic.warnings, "a degenerate heuristic result must be recorded"
    note = semantic.warnings[0]
    assert "semantic" in note and "fall back" in note
    assert "warning: " + note in semantic.summary()
    assert semantic.to_dict()["warnings"] == [note]
    assert semantic.summary().isascii()


def test_semantic_on_text_without_punctuation_does_not_over_split():
    semantic = rag_chunker.chunk(NO_PUNCTUATION, size=100, overlap=20, method="semantic")
    sentence = rag_chunker.chunk(NO_PUNCTUATION, size=100, overlap=20, method="sentence")
    assert semantic.n_chunks == sentence.n_chunks
    assert semantic.size_distribution["mean"] == sentence.size_distribution["mean"]
    assert semantic.warnings
    assert semantic.reassemble() == NO_PUNCTUATION


def test_a_real_topic_shift_is_still_found_and_carries_no_warning():
    topic_a = "Solar panels turn light into power. Inverters change the current. "
    topic_b = "Cats sleep most of the day. Kittens sleep even longer. "
    text = topic_a * 3 + topic_b * 3
    result = rag_chunker.chunk(text, size=60, overlap=0, method="semantic")
    assert result.n_chunks >= 2
    assert result.warnings == []


def test_structural_without_headings_records_its_fallback_too():
    result = rag_chunker.chunk(PROSE, size=25, overlap=5, method="structural")
    assert any("no headings" in note for note in result.warnings)
    assert "warning: " in result.summary()
    assert result.to_dict()["warnings"] == result.warnings


# --------------------------------------------------------------------------- #
# major: a missing path given as a str was silently chunked as literal text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["quarterly_report.md", "notes.txt", "page.html"])
def test_a_missing_document_path_raises_whether_it_is_a_str_or_a_path(name, tmp_path):
    missing = tmp_path / name
    for value in (str(missing), missing, name):
        with pytest.raises(FileNotFoundError) as excinfo:
            rag_chunker.chunk(value)
        assert name in str(excinfo.value), "the error must name the path"


def test_chunk_documents_reports_a_missing_path_too(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag_chunker.chunk_documents([PROSE, str(tmp_path / "gone.md")])


@pytest.mark.parametrize(
    "prose",
    [
        "See figure 3.txt",
        "Read the notes in config.md for details.",
        "https://example.com/guide.md",
        "The file was called report.html and nobody could find it.",
    ],
)
def test_prose_that_merely_ends_in_a_document_suffix_is_still_text(prose):
    result = rag_chunker.chunk(prose, size=50, overlap=5)
    assert result.text == prose
    assert result.origin is None


def test_an_existing_path_is_still_read_from_disk(md_file):
    assert rag_chunker.chunk(str(md_file), size=40, overlap=8).origin


# --------------------------------------------------------------------------- #
# minor: nested metadata was shared by reference with the caller
# --------------------------------------------------------------------------- #
def test_nested_metadata_is_not_shared_with_the_caller():
    meta = {"nested": {"a": 1}, "tags": ["x"]}
    before = copy.deepcopy(meta)
    result = rag_chunker.chunk(PROSE, size=25, overlap=5, metadata=meta)
    result.chunks[0].metadata["nested"]["a"] = 999
    result.chunks[0].metadata["tags"].append("y")
    assert meta == before, "the caller's dict must be untouched"
    assert result.chunks[1].metadata == before, "chunks must not share nested values"
    assert result.metadata == before


def test_a_reused_chunker_does_not_leak_nested_metadata_between_documents():
    meta = {"nested": {"n": 0}}
    chunker = rag_chunker.Chunker(size=30, overlap=5, metadata=meta)
    first = chunker.chunk(PROSE)
    first.chunks[0].metadata["nested"]["n"] = 42
    second = chunker.chunk(PROSE)
    assert second.chunks[0].metadata == {"nested": {"n": 0}}
    assert meta == {"nested": {"n": 0}}


# --------------------------------------------------------------------------- #
# minor: the CLI reported a raw JSON decoder message without naming the option
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["not json", "{'doc': 'guide'}", "{", "[1, 2]"])
def test_cli_metadata_errors_always_name_the_option(md_file, capsys, bad):
    from rag_chunker.cli import main

    code = main([str(md_file), "--metadata", bad])
    err = capsys.readouterr().err
    assert code == 1
    assert "rag-chunker: error: --metadata must be a JSON object" in err
    assert "Traceback" not in err


def test_cli_metadata_still_accepts_a_json_object(md_file, capsys):
    from rag_chunker.cli import main

    code = main([str(md_file), "--size", "40", "--overlap", "8",
                 "--metadata", '{"doc": "handbook"}', "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"][0]["metadata"] == {"doc": "handbook"}
