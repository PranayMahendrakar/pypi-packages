"""The guarantee that matters: chunking never drops or reorders content.

The chunk bodies, minus the text repeated as overlap, must join back into the
exact source string for every method and every setting.
"""
import pytest

import rag_chunker
from conftest import HTML, MARKDOWN, NO_PUNCTUATION, PROSE, UNICODE

CASES = {
    "prose": PROSE,
    "markdown": MARKDOWN,
    "unicode": UNICODE,
    "no_punctuation": NO_PUNCTUATION,
    "single_sentence": "Just the one sentence here.",
    "one_word": "word",
    "empty": "",
    "whitespace": "   \n\n   \t  \n",
    "no_trailing_newline": "First. Second. Third",
    "leading_whitespace": "\n\n   Leading whitespace then text. And more text.",
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("method", rag_chunker.METHODS)
@pytest.mark.parametrize("size,overlap", [(8, 2), (25, 5), (60, 0), (512, 64)])
def test_chunks_minus_overlap_reproduce_the_source(name, method, size, overlap):
    text = CASES[name]
    result = rag_chunker.chunk(
        text, size=size, overlap=overlap, method=method
    )
    assert result.reassemble() == text
    assert result.reproduces_source is True


@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_every_chunk_is_an_exact_slice_of_the_source(method):
    result = rag_chunker.chunk(MARKDOWN, size=30, overlap=6, method=method)
    for piece in result:
        assert MARKDOWN[piece.start:piece.end] == piece.text


@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_chunk_bodies_tile_the_source_without_gaps(method):
    result = rag_chunker.chunk(PROSE, size=20, overlap=5, method=method)
    cursor = 0
    for piece in result:
        assert piece.start <= cursor <= piece.end
        cursor = piece.end
    assert cursor == len(PROSE)


def test_reassemble_holds_with_a_token_counter():
    counter = lambda text: max(1, len(text) // 4)
    result = rag_chunker.chunk(PROSE, size=40, overlap=8, counter=counter)
    assert result.unit == "tokens"
    assert result.reassemble() == PROSE


def test_reassemble_holds_for_html_extracted_text():
    result = rag_chunker.chunk(HTML, size=20, overlap=4)
    assert result.source_kind == "html"
    assert result.reassemble() == result.text
    assert "ignored" not in result.text  # <style>/<script>/<title> dropped


def test_reassemble_holds_for_a_file_on_disk(md_file):
    result = rag_chunker.chunk(str(md_file), size=30, overlap=5)
    assert result.reassemble() == MARKDOWN


def test_summary_reports_the_check():
    result = rag_chunker.chunk(PROSE, size=30, overlap=5)
    assert "reproduce the source exactly: yes" in result.summary()
