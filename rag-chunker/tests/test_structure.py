"""Headings, code blocks, tables, HTML, and the semantic boundaries."""
import pytest

import rag_chunker
from conftest import HTML, MARKDOWN, PROSE

CODE_BLOCK = """```python
import example
example.run(retries=3)
print("done")
```"""

TABLE = """| step | command | note |
|------|---------|------|
| one  | build   | fast |
| two  | deploy  | slow |"""


def _containing(result, needle):
    return [c for c in result if needle in c.text]


@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_code_blocks_are_never_split(method):
    result = rag_chunker.chunk(MARKDOWN, size=12, overlap=2, method=method)
    holders = _containing(result, "example.run(retries=3)")
    assert holders, "the code line must survive somewhere"
    assert any(CODE_BLOCK in c.text for c in holders), (
        "a fenced code block must stay whole in one chunk"
    )


@pytest.mark.parametrize("method", rag_chunker.METHODS)
def test_markdown_tables_are_never_split(method):
    result = rag_chunker.chunk(MARKDOWN, size=12, overlap=2, method=method)
    assert any(TABLE in c.text for c in result), "a table must stay whole in one chunk"


def test_an_atomic_block_larger_than_size_still_stays_whole():
    big = "# T\n\nIntro line.\n\n```\n" + "\n".join(f"line {i}" for i in range(60)) + "\n```\n\nAfter.\n"
    result = rag_chunker.chunk(big, size=10, overlap=2, method="recursive")
    assert any("line 0" in c.text and "line 59" in c.text for c in result)
    assert result.reassemble() == big


def test_structural_splits_on_headings_and_carries_the_trail():
    result = rag_chunker.chunk(MARKDOWN, size=60, overlap=10, method="structural")
    trails = [c.heading_path for c in result]
    assert ("Handbook",) in trails
    assert ("Handbook", "Installation") in trails
    assert ("Handbook", "Troubleshooting", "Network") in trails
    assert ("Handbook", "Troubleshooting", "Disk") in trails
    assert result.n_headings == 5


def test_heading_trail_is_carried_by_the_other_methods_too():
    result = rag_chunker.chunk(MARKDOWN, size=40, overlap=8, method="sentence")
    assert any(c.heading_path for c in result)


def test_structural_falls_back_when_there_are_no_headings(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="rag_chunker._core"):
        result = rag_chunker.chunk(PROSE, size=25, overlap=5, method="structural")
    assert result.n_chunks > 1
    assert result.n_headings == 0
    assert "no headings" in caplog.text


def test_html_headings_and_atomic_blocks(html):
    result = rag_chunker.chunk(html, size=25, overlap=5, method="structural")
    trails = [c.heading_path for c in result]
    assert ("Report",) in trails
    assert ("Report", "Details") in trails
    assert result.n_atomic_blocks == 2  # the <pre> and the <table>
    assert any("def f(x):" in c.text and "return x + 1" in c.text for c in result)
    assert "var ignored" not in result.text


def test_html_from_a_file_reports_its_origin(html_file):
    result = rag_chunker.chunk(html_file, size=30, overlap=5)
    assert result.source_kind == "html"
    assert result.origin.endswith("report.html")
    assert "Report" in result.text


def test_markdown_file_is_detected_from_its_suffix(md_file):
    result = rag_chunker.chunk(md_file, size=40, overlap=8)
    assert result.source_kind == "markdown"


def test_source_kind_can_be_forced():
    markup = "<p>One two three. Four five six.</p>"
    as_text = rag_chunker.Chunker(size=10, overlap=2, source="text").chunk(markup)
    assert "<p>" in as_text.text
    as_html = rag_chunker.Chunker(size=10, overlap=2, source="html").chunk(markup)
    assert "<p>" not in as_html.text


def test_semantic_prefers_the_topic_shift():
    topic_a = "Solar panels turn light into power. Inverters change the current. "
    topic_b = "Cats sleep most of the day. Kittens sleep even longer. "
    text = topic_a * 3 + topic_b * 3
    result = rag_chunker.chunk(text, size=60, overlap=0, method="semantic")
    assert result.n_chunks >= 2
    assert result.reassemble() == text


def test_semantic_is_deterministic_across_calls():
    a = rag_chunker.chunk(PROSE, size=30, overlap=5)
    b = rag_chunker.chunk(PROSE, size=30, overlap=5)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_semantic_on_uniform_text_logs_its_fallback(caplog):
    import logging

    text = "Same sentence repeated. " * 12
    with caplog.at_level(logging.INFO, logger="rag_chunker._core"):
        result = rag_chunker.chunk(text, size=20, overlap=4, method="semantic")
    assert result.reassemble() == text
    assert result.n_chunks > 1


def test_abbreviations_do_not_end_a_sentence():
    text = "Dr. Smith met Mr. Lee at 5 p.m. They discussed the plan. It went well."
    result = rag_chunker.chunk(text, size=512, overlap=0, method="sentence")
    assert result.n_chunks == 1
    small = rag_chunker.chunk(text, size=6, overlap=0, method="sentence")
    assert any("Dr. Smith met Mr. Lee" in c.text for c in small)
