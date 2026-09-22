"""Reading texts from a list or from a .txt / .csv / .tsv / .jsonl / .json file."""
import json

import pytest

from semantic_dedup import dedupe, find_duplicates
from semantic_dedup._io import load_texts, write_texts

LINES = [
    "The meeting was postponed.",
    "the meeting was postponed",
    "Lunch is at noon.",
]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_txt_file_is_one_text_per_line(tmp_path):
    path = _write(tmp_path / "notes.txt", "\n".join(LINES) + "\n")
    assert load_texts(path) == LINES
    result = dedupe(path)
    assert result.n_texts == 3
    assert result.groups == [[0, 1]]


def test_txt_file_with_unicode_round_trips(tmp_path):
    path = _write(tmp_path / "uni.txt", "会議は延期されました\n会議は延期されました\nCafé fermé\n")
    assert load_texts(path) == ["会議は延期されました", "会議は延期されました", "Café fermé"]
    assert dedupe(path).groups == [[0, 1]]


def test_csv_picks_a_text_column_automatically(tmp_path):
    path = _write(
        tmp_path / "faq.csv",
        "id,question\n1,How do I reset my password?\n2,How can I reset the password?\n3,Where is the invoice?\n",
    )
    assert load_texts(path) == [
        "How do I reset my password?",
        "How can I reset the password?",
        "Where is the invoice?",
    ]
    result = dedupe(path, threshold=0.7)
    assert result.groups == [[0, 1]]


def test_csv_column_can_be_named(tmp_path):
    path = _write(tmp_path / "two.csv", "title,body\nA,alpha beta\nB,alpha beta\n")
    assert load_texts(path, "title") == ["A", "B"]
    assert load_texts(path, "BODY") == ["alpha beta", "alpha beta"]
    assert dedupe(path, column="body").groups == [[0, 1]]


def test_csv_unknown_column_says_what_is_available(tmp_path):
    path = _write(tmp_path / "x.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="nope"):
        load_texts(path, "nope")


def test_csv_header_only_and_ragged_rows(tmp_path):
    header_only = _write(tmp_path / "empty.csv", "id,text\n")
    assert load_texts(header_only) == []
    assert dedupe(header_only).n_texts == 0

    ragged = _write(tmp_path / "ragged.csv", "id,text\n1,hello\n2\n3,hello\n")
    assert load_texts(ragged) == ["hello", "", "hello"]


def test_tsv_uses_tabs(tmp_path):
    path = _write(tmp_path / "t.tsv", "id\ttext\n1\ta, with a comma\n2\ta, with a comma\n")
    assert load_texts(path) == ["a, with a comma", "a, with a comma"]
    assert dedupe(path).groups == [[0, 1]]


def test_jsonl_one_object_per_line(tmp_path):
    body = "\n".join(json.dumps({"id": i, "text": line}, ensure_ascii=False)
                     for i, line in enumerate(LINES))
    path = _write(tmp_path / "notes.jsonl", body + "\n")
    assert load_texts(path) == LINES
    assert dedupe(path).groups == [[0, 1]]


def test_jsonl_key_can_be_named_and_missing_keys_are_reported(tmp_path):
    path = _write(tmp_path / "k.jsonl", '{"q": "one"}\n{"q": "one"}\n')
    assert load_texts(path, "q") == ["one", "one"]
    with pytest.raises(ValueError, match="missing"):
        load_texts(path, "missing")


def test_json_array_file_is_accepted(tmp_path):
    path = _write(tmp_path / "arr.json", json.dumps(LINES, ensure_ascii=False))
    assert load_texts(path) == LINES
    objects = _write(tmp_path / "objs.json", json.dumps([{"text": t} for t in LINES]))
    assert load_texts(objects) == LINES


def test_bad_jsonl_line_names_the_line(tmp_path):
    path = _write(tmp_path / "bad.jsonl", '{"text": "ok"}\nnot json at all\n')
    with pytest.raises(ValueError, match="line 2"):
        load_texts(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_texts(str(tmp_path / "nope.txt"))


def test_non_string_items_are_rejected_with_the_position():
    with pytest.raises(ValueError, match=r"texts\[2\]"):
        load_texts(["ok", "ok", 7])
    with pytest.raises(TypeError, match="bytes"):
        load_texts(b"raw bytes")


def test_tuples_and_generators_are_accepted():
    assert load_texts(("a", "b")) == ["a", "b"]
    assert load_texts(iter(["a", "b"])) == ["a", "b"]
    assert find_duplicates(("same", "same")).groups == [[0, 1]]


def test_write_texts_is_utf8_and_one_per_line(tmp_path):
    target = tmp_path / "out.txt"
    write_texts(str(target), ["会議は延期されました", "two\nlines become one"])
    content = target.read_text(encoding="utf-8")
    assert content == "会議は延期されました\ntwo lines become one\n"
