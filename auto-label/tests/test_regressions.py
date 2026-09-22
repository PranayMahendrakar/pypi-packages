"""Regressions for the issues QA round 1 found.

One test per finding, named after the symptom, so a reappearance is obvious.
"""

import json
import logging
import pathlib
import re

import pandas as pd
import pytest

import auto_label
from auto_label import Labeler, Rule
from auto_label.cli import main
from auto_label.models import looks_like_free_text

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

TICKETS = (
    ["refund my invoice", "the invoice is wrong", "please refund me", "my invoice is late"] * 10
    + ["the app crashed", "error on export", "it crashed again", "error 500 here"] * 10
    + ["invoice question about a refund", "crashed with an error again"] * 10
    + ["money back on the wrong charge", "it freezes when I click save"] * 10
)
RULES = {"billing": ["invoice", "refund"], "bug": ["crashed", "error"]}


# --- major: a text column inside a DataFrame was one-hot encoded ---------------


def test_text_column_in_a_dataframe_gets_the_same_result_as_a_series():
    """The same column, wrapped in a DataFrame, must not degrade the model."""
    series = pd.Series(TICKETS)
    from_series = auto_label.label(series, RULES)
    from_frame = auto_label.label(series.to_frame("text"), RULES)

    assert from_frame.coverage == from_series.coverage
    assert from_frame.labels == from_series.labels
    assert from_frame.source == from_series.source
    assert from_frame.coverage > 0.9
    assert any("free text" in note for note in from_frame.notes), from_frame.notes


def test_a_short_repeated_string_column_is_still_one_hot_encoded():
    """The free-text heuristic must not swallow genuine categories."""
    frame = pd.DataFrame({"region": ["north", "south"] * 30, "spend": range(60)})
    assert looks_like_free_text(frame["region"]) is False
    assert looks_like_free_text(pd.Series(TICKETS)) is True
    # a column the caller declared Categorical is taken at their word
    assert looks_like_free_text(pd.Series(TICKETS, dtype="category")) is False


def test_free_text_columns_are_named_in_the_notes():
    # "amount" cycles, so it reads as a measurement; a monotonic near-unique column is
    # dropped as a row id instead (see test_regressions_round2.py).
    frame = pd.DataFrame({"text": TICKETS, "amount": [10.0, 250.0, 7.5, 99.0] * (len(TICKETS) // 4)})
    result = auto_label.label(frame, RULES)
    note = [n for n in result.notes if "free text" in n]
    assert note and "'text'" in note[0] and "numeric 'amount'" in note[0]


# --- major: README table cells were cut at the first inner pipe ----------------


def _label_result_table_rows():
    text = README.read_text(encoding="utf-8")
    section = text.split("`LabelResult`", 1)[1]
    return [line for line in section.splitlines() if line.startswith("|")]


def test_readme_table_pipes_inside_code_spans_are_escaped():
    """GFM cuts a cell at an unescaped '|', even inside a code span."""
    rows = _label_result_table_rows()
    assert rows, "the README no longer has a LabelResult table"
    for row in rows:
        cells = re.split(r"(?<!\\)\|", row)
        assert len(cells) == 4, f"row splits into {len(cells) - 2} cells, not 2: {row}"
    joined = "\n".join(rows)
    assert r"list[str \| None]" in joined
    assert r'list["rule" \| "model" \| "llm" \| None]' in joined


def test_readme_renders_the_two_core_rows_in_full():
    """The check above, confirmed through the renderer PyPI actually uses."""
    md = pytest.importorskip("readme_renderer.markdown")
    pytest.importorskip("cmarkgfm")
    html = md.render(README.read_text(encoding="utf-8"))
    assert html is not None, "README is not valid Markdown for PyPI"
    assert "<code>list[str | None]</code>, one per item" in html
    assert '<code>list["rule" | "model" | "llm" | None]</code>' in html


# --- major: a frame with rows but zero columns ---------------------------------


def test_dataframe_with_rows_but_no_columns_raises_at_the_entry_point():
    frame = pd.DataFrame({"amount": [10, 20, 30], "qty": [1, 2, 3]})
    text_only = frame.select_dtypes(include="object")  # shape (3, 0)
    assert text_only.shape == (3, 0)
    for degenerate in (text_only, frame.drop(columns=frame.columns), frame[[]]):
        with pytest.raises(ValueError, match="no columns"):
            auto_label.label(degenerate, {"any": {"func": lambda row: True}})


def test_a_frame_with_no_rows_and_no_columns_is_still_an_empty_result():
    result = auto_label.label(pd.DataFrame(), {"a": ["x"]})
    assert len(result) == 0 and result.coverage == 0.0
    assert result.to_frame().empty and result.to_dict()["records"] == []


def test_every_result_list_has_one_entry_per_item():
    """The zero-column bug showed up as items shorter than labels."""
    frame = pd.DataFrame({"text": ["refund please", "app crashed", "hello"]})
    result = auto_label.label(frame, RULES)
    n = len(result.labels)
    assert len(result.items) == n and len(result.index) == n
    assert len(result.confidence) == n and len(result.source) == n
    assert len(result.to_frame()) == n
    assert len(result.to_dict()["records"]) == result.to_dict()["n_items"] == n


# --- minor: dict input raised bare pandas errors -------------------------------


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"text": "free prize"}, "column 'text' is a str"),
        ({"a": [1, 2, 3], "b": [1, 2]}, "same length"),
        ({"n": 5}, "column 'n' is a int"),
    ],
)
def test_bad_dict_input_is_rejected_with_auto_label_context(data, expected):
    with pytest.raises(ValueError, match="auto_label") as exc:
        auto_label.label(data, {"s": ["free"]})
    assert expected in str(exc.value)


# --- minor: llm= argument and log volume ---------------------------------------


def test_non_callable_llm_raises_type_error():
    with pytest.raises(TypeError, match="llm must be a callable"):
        auto_label.label(["hi"], {"a": ["x"]}, llm="nope")


def test_a_failing_llm_hook_warns_once_not_once_per_item(caplog):
    def boom(text, candidates):
        raise RuntimeError("api down")

    with caplog.at_level(logging.WARNING, logger="auto_label.labeler"):
        result = auto_label.label([f"item {i}" for i in range(200)], {"a": ["zzz"]}, llm=boom)

    warnings = [r for r in caplog.records if "api down" in r.getMessage()]
    assert len(warnings) == 1, f"one warning per distinct failure, got {len(warnings)}"
    assert result.notes[-1] == "llm: asked 200, labeled 0, failed 200"
    assert result.coverage == 0.0


def test_two_distinct_llm_failures_warn_once_each(caplog):
    def flaky(text, candidates):
        raise RuntimeError("boom " + text[-1])

    with caplog.at_level(logging.WARNING, logger="auto_label.labeler"):
        auto_label.label(["a 1", "b 1", "c 2"], {"x": ["zzz"]}, llm=flaky)

    assert len([r for r in caplog.records if "boom" in r.getMessage()]) == 2


# --- minor: random_state validation --------------------------------------------


@pytest.mark.parametrize("bad", ["x", 1.5, None, True])
def test_random_state_must_be_an_integer(bad):
    with pytest.raises(ValueError, match="random_state must be an integer"):
        Labeler(random_state=bad)


# --- minor: columns 1 and "1" are distinct -------------------------------------


def test_int_and_string_columns_that_look_alike_are_not_duplicates():
    frame = pd.DataFrame({1: ["a", "b"], "1": ["c", "d"]})
    result = auto_label.label(frame, {"x": ["a"]}, model=False)
    assert result.labels == ["x", None]


def test_genuinely_duplicate_columns_still_raise():
    frame = pd.DataFrame([[1, 2]], columns=["dup", "dup"])
    with pytest.raises(ValueError, match="duplicate column names"):
        auto_label.label(frame, {"x": ["a"]}, model=False)


# --- minor: set input was non-deterministic ------------------------------------


def test_set_input_is_rejected_rather_than_ordered_at_random():
    with pytest.raises(TypeError, match="stable iteration order"):
        Labeler().add_rule("spam", keywords=["free"]).label({"Free prize", "Team meeting"})
    with pytest.raises(TypeError, match="stable iteration order"):
        auto_label.label(frozenset({"a", "b"}), {"x": ["a"]})


# --- minor: Rule() with an empty keyword ---------------------------------------


@pytest.mark.parametrize("keywords", [("",), ("ok", ""), ("   ",)])
def test_rule_with_an_empty_keyword_raises_value_error(keywords):
    with pytest.raises(ValueError, match="non-empty"):
        Rule(label="x", keywords=keywords)


def test_rule_built_directly_still_works_for_good_keywords():
    rule = Rule(label="x", keywords=("free",))
    assert rule.keyword_pattern is not None
    assert rule.keyword_pattern.search("a free lunch")


# --- minor: a user func that raises --------------------------------------------


def test_a_raising_func_rule_names_the_rule_and_the_item():
    with pytest.raises(ValueError, match=r"rule 'x': func raised on item 0"):
        Labeler().add_rule("x", func=lambda t: 1 / 0).label(["a", "b"])


def test_a_raising_func_rule_reports_the_original_error():
    with pytest.raises(ValueError, match="ZeroDivisionError"):
        auto_label.label(["a"], {"x": lambda t: 1 / 0})


# --- CLI ------------------------------------------------------------------------


@pytest.fixture
def tickets_csv(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("text,amount\nRefund please,10\nApp crash,0\n", encoding="utf-8")
    return path


def test_cli_directory_input_says_it_is_a_directory(tmp_path, capsys):
    assert main([str(tmp_path), "--rule", "a=x"]) == 2
    err = capsys.readouterr().err
    assert "is a directory" in err and "Permission denied" not in err


def test_cli_bad_rules_json_names_the_file(tickets_csv, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert main([str(tickets_csv), "--rules", str(bad), "--rule", "a=x"]) == 2
    err = capsys.readouterr().err
    assert "bad.json" in err and "not valid JSON" in err


def test_cli_missing_rules_file_names_the_file(tickets_csv, tmp_path, capsys):
    assert main([str(tickets_csv), "--rules", str(tmp_path / "nope.json")]) == 2
    assert "nope.json" in capsys.readouterr().err


def test_cli_unknown_output_extension_is_rejected(tickets_csv, tmp_path, capsys):
    out = tmp_path / "weird.txt"
    assert main([str(tickets_csv), "--rule", "a=refund", "--output", str(out)]) == 2
    err = capsys.readouterr().err
    assert "unsupported extension" in err and ".csv" in err
    assert not out.exists()


def test_cli_tabular_output_is_joinable_with_the_input(tickets_csv, tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written.columns) == ["index", "text", "amount", "label", "confidence", "source"]
    assert written["text"].tolist() == ["Refund please", "App crash"]
    assert written["amount"].tolist() == [10, 0]
    assert written["label"].tolist()[0] == "bill"
    assert "{" not in out.read_text(encoding="utf-8")


def test_cli_tabular_output_does_not_overwrite_a_label_column(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text,label\nRefund please,mine\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert main([str(src), "--rule", "bill=refund", "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert written["label"].tolist() == ["mine"]
    assert written["label_2"].tolist() == ["bill"]


def test_cli_text_output_keeps_the_item_column(tmp_path):
    lines = tmp_path / "lines.txt"
    lines.write_text("Free prize\nTeam meeting\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert main([str(lines), "--rule", "spam=free", "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written.columns) == ["index", "item", "label", "confidence", "source"]


def test_cli_parquet_output_without_pyarrow_names_the_extra(tickets_csv, tmp_path, monkeypatch, capsys):
    def no_engine(self, *args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    assert main([str(tickets_csv), "--rule", "a=refund", "--output", str(tmp_path / "o.parquet")]) == 2
    err = capsys.readouterr().err
    assert "auto-label[parquet]" in err
    assert "conda" not in err


def test_cli_keeps_source_line_numbers_when_skipping_blank_lines(tmp_path, capsys):
    path = tmp_path / "blanks.txt"
    path.write_text("Free prize\n\n   \nTeam meeting\nnothing\n", encoding="utf-8")
    assert main([str(path), "--rule", "spam=free", "--rule", "work=meeting", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["n_items"] == 3
    # "Team meeting" is line 4 of the file, so index 3 - not renumbered to 1
    assert [(r["index"], r["item"]) for r in payload["records"]] == [
        (0, "Free prize"),
        (3, "Team meeting"),
        (4, "nothing"),
    ]
    assert any("skipped 2 blank line(s)" in note for note in payload["notes"])


def test_cli_blank_line_skip_is_logged(tmp_path, caplog):
    path = tmp_path / "blanks.txt"
    path.write_text("Free prize\n\nTeam meeting\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="auto_label.cli"):
        assert main([str(path), "--rule", "spam=free"]) == 0
    assert any("blank line" in r.getMessage() for r in caplog.records)


def test_cli_file_without_blank_lines_records_no_note(tmp_path, capsys):
    path = tmp_path / "clean.txt"
    path.write_text("Free prize\nTeam meeting\n", encoding="utf-8")
    assert main([str(path), "--rule", "spam=free", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert not any("blank line" in note for note in payload["notes"])


# --- minor: a non-UTF-8 CSV ------------------------------------------------------


@pytest.fixture
def latin_csv(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("text\nfacture en double\ncaf\xe9 cr\xe8me\n".encode("cp1252"))
    return path


def test_non_utf8_csv_still_labels_and_warns(latin_csv, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="auto_label._data"):
        assert main([str(latin_csv), "--rule", "a=facture"]) == 0
    assert "1/2 items labeled" in capsys.readouterr().out
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "not valid UTF-8" in message and "--encoding" in message


def test_encoding_flag_reads_the_file_correctly(latin_csv, capsys):
    assert main([str(latin_csv), "--rule", "a=facture", "--encoding", "cp1252", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["item"]["text"] for r in payload["records"]] == ["facture en double", "café crème"]


def test_wrong_encoding_flag_says_which_encoding_failed(latin_csv, capsys):
    assert main([str(latin_csv), "--rule", "a=facture", "--encoding", "utf-8"]) == 2
    err = capsys.readouterr().err
    assert "latin.csv" in err and "utf-8" in err
    assert err.count("auto_label") == 0  # the CLI prefix is not printed twice


def test_encoding_flag_now_applies_to_a_text_file_too(tmp_path, capsys):
    """Round 1 rejected --encoding here; round 2 found a .txt is exactly where it is needed."""
    path = tmp_path / "lines.txt"
    path.write_bytes("caf\xe9 cr\xe8me\n".encode("cp1252"))
    assert main([str(path), "--rule", "coffee=caf\xe9", "--encoding", "cp1252", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["item"] for r in payload["records"]] == ["caf\xe9 cr\xe8me"]
    assert payload["records"][0]["label"] == "coffee"
