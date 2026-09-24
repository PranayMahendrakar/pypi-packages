"""Regressions for the issues QA round 2 found.

One test per finding, named after the symptom, so a reappearance is obvious.
Round 1's regressions live in ``test_regressions.py``.
"""

import io
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import auto_label
from auto_label import Labeler
from auto_label.cli import main, write_output
from auto_label.models import looks_like_row_id


# --- major: stdin was decoded with the locale encoding, not UTF-8 ----------------


class FakeStdin:
    """``sys.stdin`` as Windows hands it over: a text view over the real bytes.

    The text layer is cp1252 with ``errors="surrogateescape"``, which is what turns
    piped UTF-8 into mojibake; ``.buffer`` still holds what the user actually sent.
    """

    def __init__(self, data: bytes, encoding: str = "cp1252") -> None:
        self.buffer = io.BytesIO(data)
        self._text = io.TextIOWrapper(
            io.BytesIO(data), encoding=encoding, errors="surrogateescape"
        )

    def read(self) -> str:
        return self._text.read()


CN_BYTES = "发票退款\nTeam meeting\n".encode("utf-8")


def test_piped_utf8_stdin_is_decoded_as_utf8_not_as_the_locale(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", FakeStdin(CN_BYTES))
    assert main(["-", "--rule", "billing=退款", "--rule", "work=meeting", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["item"] for r in payload["records"]] == ["发票退款", "Team meeting"]
    assert [r["label"] for r in payload["records"]] == ["billing", "work"]
    assert payload["n_labeled"] == 2


def test_the_stdin_and_the_file_route_agree_on_the_same_bytes(tmp_path, monkeypatch, capsys):
    """The defect: (a) as a file labeled 2/2, (b) piped labeled 1/2 and exited 0."""
    path = tmp_path / "cn.txt"
    path.write_bytes(CN_BYTES)
    assert main([str(path), "--rule", "billing=退款", "--rule", "work=meeting"]) == 0
    from_file = capsys.readouterr().out

    monkeypatch.setattr(sys, "stdin", FakeStdin(CN_BYTES))
    assert main(["-", "--rule", "billing=退款", "--rule", "work=meeting"]) == 0
    from_stdin = capsys.readouterr().out

    assert "2/2 items labeled (100.0%)" in from_file
    assert from_stdin.splitlines()[0] == from_file.splitlines()[0]


def test_the_locale_text_layer_really_would_have_corrupted_it():
    """Guard for the guard: prove FakeStdin reproduces the mojibake it stands in for."""
    assert FakeStdin(CN_BYTES).read() != CN_BYTES.decode("utf-8")


def test_stdin_honours_an_explicit_encoding(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", FakeStdin("caf\xe9 cr\xe8me\n".encode("cp1252")))
    assert main(["-", "--rule", "coffee=caf\xe9", "--encoding", "cp1252", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["item"] for r in payload["records"]] == ["caf\xe9 cr\xe8me"]
    assert payload["records"][0]["label"] == "coffee"


def test_a_stdin_double_without_a_buffer_still_works(monkeypatch, capsys):
    """An embedder or a test may hand us an already-decoded text stream."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("Free prize now\n"))
    assert main(["-", "--rule", "spam=prize"]) == 0
    assert "1/1 items labeled" in capsys.readouterr().out


def test_non_utf8_bytes_on_stdin_are_replaced_and_reported(monkeypatch, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="auto_label._data"):
        monkeypatch.setattr(sys, "stdin", FakeStdin("facture\ncaf\xe9\n".encode("cp1252")))
        assert main(["-", "--rule", "a=facture"]) == 0
    out = capsys.readouterr().out
    assert "stdin is not valid UTF-8" in " ".join(r.getMessage() for r in caplog.records)
    assert "not valid UTF-8" in out and "--encoding" in out  # also recorded as a note


# --- major: a failed --output write truncated the destination -------------------


PRECIOUS = "PRECIOUS EXISTING DATA\n"


@pytest.fixture
def keeper(tmp_path):
    def make(name):
        path = tmp_path / name
        path.write_text(PRECIOUS, encoding="utf-8")
        return path

    return make


@pytest.fixture
def tickets_csv(tmp_path):
    path = tmp_path / "tickets.csv"
    path.write_text("text\nRefund please\nApp crash\n", encoding="utf-8")
    return path


def _fail_after_opening(monkeypatch, attr_owner, attr):
    """Replace a writer with one that half-writes its target, then fails.

    That is the shape of every real mid-write failure - a full disk, an encode error,
    a revoked permission: pandas and json.dump open the destination and start emitting
    before anything goes wrong. Injecting the failure *before* the open would not
    reproduce the defect at all, because the old code's damage was done by the open.
    """

    def half_written(_self, target, *args, **kwargs):
        with open(str(target), "w", encoding="utf-8") as fh:
            fh.write("index,item,label\n")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(attr_owner, attr, half_written)


def test_a_failed_csv_write_leaves_the_existing_file_untouched(
    tickets_csv, keeper, monkeypatch, capsys
):
    out = keeper("keep.csv")
    _fail_after_opening(monkeypatch, pd.DataFrame, "to_csv")
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == PRECIOUS
    assert "auto-label: error:" in capsys.readouterr().err


def test_a_failed_json_write_leaves_the_existing_file_untouched(
    tickets_csv, keeper, monkeypatch, capsys
):
    out = keeper("keep.json")

    def half_written(payload, fh, **kwargs):
        fh.write('{\n  "records": [\n    {\n      "index": 0,\n      "item": ')
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(json, "dump", half_written)
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == PRECIOUS
    capsys.readouterr()


def test_a_failed_write_leaves_no_temporary_file_behind(
    tickets_csv, keeper, tmp_path, monkeypatch, capsys
):
    out = keeper("keep.csv")
    _fail_after_opening(monkeypatch, pd.DataFrame, "to_csv")
    before = sorted(p.name for p in tmp_path.iterdir())
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    capsys.readouterr()


def test_a_failed_write_does_not_create_the_destination_when_it_was_absent(
    tickets_csv, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "brand-new.csv"
    _fail_after_opening(monkeypatch, pd.DataFrame, "to_csv")
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 2
    assert not out.exists()
    capsys.readouterr()


def test_undecodable_text_names_the_cause_and_keeps_the_old_file(keeper):
    """The raw codec message ("can't encode '\\udc8f'") told the user nothing to act on."""
    out = keeper("keep.csv")
    result = auto_label.label(["ok", "bad \udc8f text"], {"a": ["ok"]}, model=False)

    with pytest.raises(ValueError) as excinfo:
        write_output(result, str(out))

    message = str(excinfo.value)
    assert "--encoding" in message and "not valid UTF-8" in message
    assert "left unchanged" in message
    assert out.read_text(encoding="utf-8") == PRECIOUS


def test_a_successful_write_still_replaces_the_destination(tickets_csv, keeper, capsys):
    out = keeper("keep.csv")
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written["text"]) == ["Refund please", "App crash"]
    assert written["label"].tolist()[0] == "bill"
    capsys.readouterr()


def test_output_into_a_missing_folder_says_so(tickets_csv, tmp_path, capsys):
    out = tmp_path / "nope" / "labeled.csv"
    assert main([str(tickets_csv), "--rule", "bill=refund", "--output", str(out)]) == 2
    assert "does not exist" in capsys.readouterr().err


# --- major: an id column steered the model --------------------------------------


ID_TEXT = [
    "invoice overdue payment", "refund overdue payment", "invoice overdue billing",
    "refund overdue account", "crash login error", "fatal error crash",
    "error crash save", "crash error checkout",
    "overdue payment reminder",  # no rule keyword; plainly a billing ticket
]
ID_RULES = {"billing": ["invoice", "refund"], "bug": ["crash", "error"]}


def test_an_id_column_does_not_flip_the_label_the_text_supports():
    """The export is sorted by type, so ticket_id separated the classes perfectly."""
    text_only = auto_label.label(pd.DataFrame({"text": ID_TEXT}), ID_RULES)
    with_id = auto_label.label(
        pd.DataFrame({"ticket_id": range(1001, 1010), "text": ID_TEXT}), ID_RULES
    )
    assert text_only.labels[-1] == "billing"
    assert with_id.labels[-1] == text_only.labels[-1]
    assert with_id.labels == text_only.labels


def test_the_ignored_id_column_is_named_in_the_notes():
    result = auto_label.label(
        pd.DataFrame({"ticket_id": range(1001, 1010), "text": ID_TEXT}), ID_RULES
    )
    note = [n for n in result.notes if "row identifier" in n]
    assert note and "'ticket_id'" in note[0]
    # and it is not presented as a feature the model understood
    assert not any("numeric 'ticket_id'" in n for n in result.notes)


def test_the_id_heuristic_is_logged_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="auto_label.models"):
        auto_label.label(pd.DataFrame({"ticket_id": range(1001, 1010), "text": ID_TEXT}), ID_RULES)
    assert "row identifiers" in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "values, expected",
    [
        (list(range(1001, 1010)), True),          # a plain ascending id
        (list(range(1009, 1000, -1)), True),      # descending counts too
        ([10.0, 250.0, 7.5, 99.0, 12.0], False),  # a genuine measurement
        ([1, 2, 3], False),                       # too few rows to tell
        ([1, 1, 2, 2, 3, 3, 4, 4], False),        # repeats: not an identifier
        ([1.5, 2.5, 3.5, 4.5, 5.5], False),       # ordered, but not whole numbers
    ],
)
def test_looks_like_row_id_only_fires_on_id_shaped_columns(values, expected):
    assert looks_like_row_id(pd.Series(values, dtype="float64")) is expected


def test_a_real_numeric_column_is_still_used_as_a_feature():
    frame = pd.DataFrame({"text": ID_TEXT, "amount": [10.0, 250.0, 7.5, 99.0] * 2 + [42.0]})
    result = auto_label.label(frame, ID_RULES)
    assert any("numeric 'amount'" in n for n in result.notes), result.notes


# --- major: a column empty in the training rows killed the whole tabular model ---


def _csat_frame():
    """16 tickets; 'csat' was added to the export later, so only the newest 4 rows have it -
    exactly the rows the rules cannot label."""
    return pd.DataFrame(
        {
            "note": ["refund the invoice please", "the app crashed with an error"] * 6
            + [
                "money back on a wrong charge", "it freezes when I click save",
                "you took too much money", "the page will not load at all",
            ],
            "channel": ["email", "phone"] * 8,
            "csat": [np.nan] * 12 + [4.0, 2.0, 5.0, 1.0],
        }
    )


def _csat_labeler():
    lab = Labeler(min_confidence=0.4)
    lab.add_rule("billing", keywords=["refund", "invoice"])
    lab.add_rule("bug", keywords=["crashed", "error"])
    return lab


def test_a_column_empty_in_the_training_rows_does_not_switch_the_model_off():
    frame = _csat_frame()
    with_csat = _csat_labeler().label(frame)
    without_csat = _csat_labeler().label(frame.drop(columns=["csat"]))

    assert with_csat.model_trained is True
    assert with_csat.coverage == without_csat.coverage == 1.0
    assert with_csat.labels == without_csat.labels


def test_the_useless_column_is_named_in_the_notes_not_a_sklearn_message():
    result = _csat_labeler().label(_csat_frame())
    note = [n for n in result.notes if "empty for every rule-labeled row" in n]
    assert note and "'csat'" in note[0]
    assert not any("StandardScaler" in n for n in result.notes)
    assert not any("training failed" in n for n in result.notes)


def test_no_scikit_learn_warning_escapes_into_the_callers_warning_stream():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _csat_labeler().label(_csat_frame())
    leaked = [w for w in caught if "observed values" in str(w.message)]
    assert leaked == [], [str(w.message) for w in leaked]


def test_a_table_whose_every_column_is_useless_degrades_instead_of_crashing():
    frame = pd.DataFrame(
        {
            "note": ["refund the invoice", "the app crashed error"] * 6
            + ["money back wrong charge", "freezes when I click save",
               "you took too much money", "the page will not load"],
            "csat": [np.nan] * 12 + [4.0, 2.0, 5.0, 1.0],
        }
    )
    lab = _csat_labeler()
    result = lab.label(frame[["csat"]].assign(csat=frame["csat"]))
    # nothing for the rules to match, so the model never runs; it must not raise
    assert result.model_trained is False
    assert all(lab is None for lab in result.labels)


# --- major: a non-UTF-8 text file was mojibaked silently -------------------------


@pytest.fixture
def latin_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(
        "facture en double\ncaf\xe9 cr\xe8me\nr\xe9sum\xe9 du probl\xe8me\n".encode("cp1252")
    )
    return path


def test_a_non_utf8_text_file_warns_the_way_a_csv_does(latin_txt, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="auto_label._data"):
        assert main([str(latin_txt), "--rule", "a=facture"]) == 0
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "notes.txt is not valid UTF-8" in logged and "--encoding" in logged
    assert "not valid UTF-8" in capsys.readouterr().out  # recorded as a note too


def test_an_accented_keyword_matches_once_the_text_file_is_read_correctly(latin_txt, capsys):
    assert main([str(latin_txt), "--rule", "coffee=caf\xe9", "--encoding", "cp1252", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["item"] for r in payload["records"]] == [
        "facture en double", "caf\xe9 cr\xe8me", "r\xe9sum\xe9 du probl\xe8me",
    ]
    assert payload["records"][1]["label"] == "coffee"


def test_a_wrong_encoding_on_a_text_file_says_which_one_failed(tmp_path, capsys):
    path = tmp_path / "cn.txt"
    path.write_bytes(CN_BYTES)
    assert main([str(path), "--rule", "a=x", "--encoding", "ascii"]) == 2
    err = capsys.readouterr().err
    assert "cn.txt" in err and "ascii" in err


# --- minor: text mode said "kept 0" with no explanation --------------------------


def test_a_text_model_that_keeps_nothing_explains_why():
    texts = [
        "invoice refund a", "refund invoice b", "invoice pay c", "crash error d",
        "error crash e", "zzz aaa", "qqq bbb", "www ccc",
    ]
    result = auto_label.label(texts, {"billing": ["invoice", "refund"], "bug": ["crash", "error"]})
    assert "kept 0" in " ".join(result.notes)
    advice = [n for n in result.notes if "min_confidence" in n]
    assert advice and "--min-confidence" in advice[0]
    assert result.source.count(None) > 0


def test_the_text_advice_is_logged_as_a_warning(caplog):
    texts = [
        "invoice refund a", "refund invoice b", "invoice pay c", "crash error d",
        "error crash e", "zzz aaa", "qqq bbb", "www ccc",
    ]
    with caplog.at_level(logging.WARNING, logger="auto_label.labeler"):
        auto_label.label(texts, {"billing": ["invoice", "refund"], "bug": ["crash", "error"]})
    assert "no prediction reached" in " ".join(r.getMessage() for r in caplog.records)


# --- minor: a func rule saw the positional index, not the caller's ---------------


def test_a_func_rule_sees_the_callers_row_label():
    df = pd.DataFrame({"amount": [10, 500]}, index=["T-100", "T-200"])
    seen = []

    def vip(row):
        seen.append(row.name)
        return row.name == "T-200"

    result = Labeler().add_rule("vip", func=vip).label(df, model=False)
    assert seen == ["T-100", "T-200"]
    assert result.labels == [None, "vip"]
    assert result.index == ["T-100", "T-200"]


def test_a_func_rule_still_sees_the_row_values():
    df = pd.DataFrame({"amount": [10, 500]}, index=["T-100", "T-200"])
    result = Labeler().add_rule("big", func=lambda row: row["amount"] > 100).label(df, model=False)
    assert result.labels == [None, "big"]


def test_a_func_rule_on_a_default_index_is_unchanged():
    df = pd.DataFrame({"amount": [10, 500]})
    seen = []
    Labeler().add_rule("x", func=lambda r: seen.append(r.name) or False).label(df, model=False)
    assert seen == [0, 1]


def test_a_func_rule_on_text_input_still_gets_the_string():
    result = Labeler().add_rule("long", func=lambda t: len(t) > 5).label(["hi", "hello there"], model=False)
    assert result.labels == [None, "long"]


# --- minor: --output CSV wrote two columns named 'index' -------------------------


def test_an_input_index_column_does_not_collide_with_the_written_index(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("index,text\n7,Refund please\n8,App crash\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert main([str(src), "--rule", "bill=refund", "--output", str(out)]) == 0
    capsys.readouterr()

    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(header) == len(set(header)), header
    read_back = pd.read_csv(out)
    assert not any(str(c).endswith(".1") for c in read_back.columns), list(read_back.columns)
    assert list(read_back["index"]) == [7, 8]
    # the tool must be able to read back the shape it just wrote
    auto_label.label(read_back, {"bill": ["refund"]}, model=False)


def test_the_ordinary_case_still_writes_a_column_called_index(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("text\nRefund please\nApp crash\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert main([str(src), "--rule", "bill=refund", "--output", str(out)]) == 0
    capsys.readouterr()
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("index,")


# --- minor: weight ties were compared with exact float equality ------------------


def test_accumulated_rounding_does_not_break_the_documented_tie_break():
    lab = Labeler()
    lab.add_rule("b", keywords=["y"], weight=0.3)   # added FIRST
    lab.add_rule("a", keywords=["x"], weight=0.1)
    lab.add_rule("a", keywords=["x2"], weight=0.2)  # 0.1 + 0.2 == 0.30000000000000004
    assert 0.1 + 0.2 != 0.3                          # the noise this test is about
    result = lab.label(["x x2 y"], model=False)
    assert result.labels == ["b"]


def test_a_genuinely_larger_weight_still_wins():
    lab = Labeler()
    lab.add_rule("b", keywords=["y"], weight=0.3)
    lab.add_rule("a", keywords=["x"], weight=0.4)
    assert lab.label(["x y"], model=False).labels == ["a"]
