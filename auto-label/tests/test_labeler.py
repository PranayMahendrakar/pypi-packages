"""End-to-end behaviour of Labeler / label(): rules, model, LLM hook, inputs, edge cases."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

import auto_label
from auto_label import Labeler, LabelResult, label

SPAM = ["Free prize, click now to claim it", "Win a free voucher today", "Claim your prize before midnight",
        "Free gift for the winner", "Prize draw closes tonight"]
WORK = ["Team meeting at 10 in room B", "Agenda for the meeting is attached", "Meeting notes from the team",
        "Quarterly agenda review", "Meeting moved to Friday"]
UNSEEN = ["Click now and claim your voucher", "Team notes from room B attached", "Voucher for the winner tonight"]
RULES = {"spam": ["free", "prize", "win"], "work": ["meeting", "agenda"]}


def spam_work_labeler(**kw):
    lab = Labeler(**kw)
    lab.add_rule("spam", keywords=["free", "prize", "win"])
    lab.add_rule("work", keywords=["meeting", "agenda"])
    return lab


# --- rules only ---------------------------------------------------------------------


def test_rules_only_when_too_few_examples_for_model():
    result = spam_work_labeler().label(["Free prize", "Team meeting", "nothing here"])
    assert result.labels == ["spam", "work", None]
    assert result.source == ["rule", "rule", None]
    assert result.confidence == [1.0, 1.0, 0.0]
    assert result.model_trained is False
    assert result.coverage == pytest.approx(2 / 3)
    assert any("at least 5" in n for n in result.notes)
    assert set(result.source) <= {"rule", None}


def test_conflicts_resolve_by_summed_weight():
    lab = Labeler()
    lab.add_rule("a", keywords=["x"], weight=1.0)
    lab.add_rule("b", keywords=["y"], weight=2.0)
    lab.add_rule("a", keywords=["z"], weight=0.5)
    result = lab.label(["x y", "x y z", "x z", "y"])
    assert result.labels == ["b", "b", "a", "b"]
    assert result.confidence[0] == pytest.approx(2 / 3)
    assert result.confidence[1] == pytest.approx(2 / 3.5)
    assert result.confidence[2] == 1.0
    assert result.source == ["rule"] * 4


def test_tie_goes_to_the_rule_added_first():
    lab = Labeler().add_rule("later", keywords=["x"]).add_rule("first", keywords=["y"])
    # 'later' was added first, so it wins the tie even though 'first' sorts earlier.
    result = lab.label(["x y"])
    assert result.labels == ["later"]
    assert result.confidence == [0.5]

    swapped = Labeler().add_rule("first", keywords=["y"]).add_rule("later", keywords=["x"])
    assert swapped.label(["x y"]).labels == ["first"]


def test_negative_weight_vetoes():
    lab = Labeler().add_rule("a", keywords=["x"]).add_rule("a", keywords=["not"], weight=-1)
    result = lab.label(["x", "x not"])
    assert result.labels == ["a", None]


def test_model_disabled():
    result = spam_work_labeler().label(SPAM + WORK + UNSEEN, model=False)
    assert result.model_trained is False
    assert result.labels[-3:] == [None, None, None]
    assert any("disabled" in n for n in result.notes)


# --- model ---------------------------------------------------------------------------


def test_model_trains_and_predicts_the_rest():
    lab = spam_work_labeler(min_confidence=0.5)
    result = lab.label(SPAM + WORK + UNSEEN)
    assert result.model_trained is True
    assert lab.model_ is not None
    assert result.source[:10] == ["rule"] * 10
    assert result.source[10:] == ["model"] * 3
    assert result.labels[10:] == ["spam", "work", "spam"]
    for c in result.confidence[10:]:
        assert 0.5 <= c <= 1.0
    assert result.coverage == 1.0


def test_model_skipped_with_fewer_than_five_examples():
    result = spam_work_labeler().label(SPAM[:2] + WORK[:2] + UNSEEN)
    assert result.model_trained is False
    assert result.labels[4:] == [None] * 3
    assert any("at least 5 rule-labeled" in n for n in result.notes)


def test_model_skipped_with_a_single_label():
    lab = Labeler().add_rule("spam", keywords=["free", "prize", "win"])
    result = lab.label(SPAM + UNSEEN)
    assert result.model_trained is False
    assert set(result.source) == {"rule", None}
    assert any("at least 2 labels" in n for n in result.notes)


def test_model_skipped_when_everything_is_rule_labeled():
    result = spam_work_labeler().label(SPAM + WORK)
    assert result.model_trained is False
    assert result.coverage == 1.0
    assert any("every item" in n for n in result.notes)


def test_min_confidence_bounds_model_predictions():
    everything = spam_work_labeler(min_confidence=0.0).label(SPAM + WORK + UNSEEN)
    assert everything.coverage == 1.0
    strict = spam_work_labeler(min_confidence=1.0).label(SPAM + WORK + UNSEEN)
    assert strict.labels[10:] == [None] * 3
    assert strict.confidence[10:] == [0.0] * 3
    assert strict.model_trained is True


def test_results_are_deterministic():
    a = label(SPAM + WORK + UNSEEN, RULES, min_confidence=0.0)
    b = label(SPAM + WORK + UNSEEN, RULES, min_confidence=0.0)
    assert a.labels == b.labels
    assert a.confidence == b.confidence
    assert a.source == b.source


def test_model_failure_degrades_to_rules(caplog):
    # Rule-labeled texts that produce an empty TF-IDF vocabulary (only punctuation).
    lab = Labeler().add_rule("a", func=lambda t: t.startswith("!")).add_rule("b", func=lambda t: t.startswith("?"))
    texts = ["!", "!!", "!!!", "?", "??", "???", "..."]
    with caplog.at_level(logging.WARNING, logger="auto_label"):
        result = lab.label(texts)
    assert result.labels[:6] == ["a", "a", "a", "b", "b", "b"]
    assert result.labels[6] is None
    assert result.model_trained is False
    assert any("training failed" in n for n in result.notes)


# --- llm hook -------------------------------------------------------------------------


def test_llm_is_called_for_low_confidence_items():
    calls = []

    def llm(text, candidates):
        calls.append((text, list(candidates)))
        return "spam"

    result = spam_work_labeler().label(["Free prize", "Team meeting", "no idea"], llm=llm)
    assert calls == [("no idea", ["spam", "work"])]
    assert result.labels == ["spam", "work", "spam"]
    assert result.source == ["rule", "rule", "llm"]
    assert result.confidence[2] == 1.0
    assert any(n.startswith("llm: asked 1, labeled 1") for n in result.notes)


def test_llm_raising_does_not_crash(caplog):
    def llm(text, candidates):
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger="auto_label"):
        result = spam_work_labeler().label(["Free prize", "no idea"], llm=llm)
    assert result.labels == ["spam", None]
    assert result.source == ["rule", None]
    assert "boom" in caplog.text
    assert any("failed 1" in n for n in result.notes)


@pytest.mark.parametrize("answer", [None, "", "   ", 123, ["spam"]])
def test_llm_unusable_answers_leave_item_unlabeled(answer):
    result = spam_work_labeler().label(["no idea"], llm=lambda t, c: answer)
    assert result.labels == [None]
    assert result.source == [None]
    assert result.confidence == [0.0]


def test_llm_answer_is_normalised_to_candidate_case():
    result = spam_work_labeler().label(["no idea"], llm=lambda t, c: "  SPAM ")
    assert result.labels == ["spam"]


def test_llm_open_label_set_accepts_new_label_closed_set_rejects_it():
    open_result = spam_work_labeler().label(["no idea"], llm=lambda t, c: "other")
    assert open_result.labels == ["other"]

    closed = spam_work_labeler(labels=["spam", "work"])
    closed_result = closed.label(["no idea"], llm=lambda t, c: "other")
    assert closed_result.labels == [None]
    assert closed_result.candidates == ["spam", "work"]


def test_llm_is_asked_about_rule_ties():
    lab = Labeler().add_rule("a", keywords=["x"]).add_rule("b", keywords=["y"])
    seen = []

    def llm(text, candidates):
        seen.append(text)
        return "b"

    result = lab.label(["x y", "x"], llm=llm)
    assert seen == ["x y"]
    assert result.labels == ["b", "a"]
    assert result.source == ["llm", "rule"]

    kept = lab.label(["x y"], llm=lambda t, c: None)
    assert kept.labels == ["a"]
    assert kept.source == ["rule"]
    assert kept.confidence == [0.5]


def test_llm_sees_row_text_for_tables():
    seen = []
    lab = Labeler().add_rule("big", query="amount > 100")
    df = pd.DataFrame({"amount": [5, 500], "note": ["hello", None]})
    lab.label(df, llm=lambda t, c: seen.append(t))
    assert seen == ["amount: 5; note: hello"]


# --- inputs and edge cases ------------------------------------------------------------


@pytest.mark.parametrize("empty", [[], (), pd.Series([], dtype=object), pd.DataFrame(), np.array([]), {}])
def test_empty_input_returns_empty_result(empty):
    result = spam_work_labeler().label(empty)
    assert len(result) == 0
    assert result.labels == [] and result.confidence == [] and result.source == []
    assert result.coverage == 0.0
    frame = result.to_frame()
    assert list(frame.columns) == ["item", "label", "confidence", "source"]
    assert len(frame) == 0
    assert result.to_dict()["n_items"] == 0
    assert "0/0" in result.summary()


def test_single_item_and_single_row():
    text = spam_work_labeler().label(["Free stuff"])
    assert text.labels == ["spam"] and text.coverage == 1.0
    row = Labeler().add_rule("big", query="amount > 1").label(pd.DataFrame({"amount": [2]}))
    assert row.labels == ["big"]
    assert row.items == [{"amount": 2}]


def test_series_index_is_preserved():
    series = pd.Series(["Free prize", "Team meeting", None], index=["a", "b", "c"])
    result = spam_work_labeler().label(series)
    assert result.index == ["a", "b", "c"]
    assert list(result.to_frame().index) == ["a", "b", "c"]
    assert result.labels == ["spam", "work", None]
    assert result.items[2] == ""


def test_non_string_items_are_cast_to_text():
    result = Labeler().add_rule("num", regex=r"^\d+$").label([1, 22, "abc", None, float("nan")])
    assert result.labels == ["num", "num", None, None, None]
    assert result.items == ["1", "22", "abc", "", ""]


def test_unsupported_inputs_raise():
    with pytest.raises(TypeError):
        spam_work_labeler().label(42)
    with pytest.raises(TypeError):
        spam_work_labeler().label(b"bytes")
    with pytest.raises(ValueError):
        spam_work_labeler().label(np.zeros((2, 2)))


def test_path_inputs(tmp_path):
    df = pd.DataFrame({"text": ["Free prize", "Team meeting"], "amount": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = spam_work_labeler().label(path)
    assert result.labels == ["spam", "work"]
    assert result.mode == "tabular"
    assert spam_work_labeler().label(str(path)).labels == ["spam", "work"]
    with pytest.raises(FileNotFoundError):
        spam_work_labeler().label(tmp_path / "missing.csv")
    other = tmp_path / "data.txt"
    other.write_text("x")
    with pytest.raises(ValueError, match="csv"):
        spam_work_labeler().label(other)


def test_dict_of_columns_is_tabular():
    result = Labeler().add_rule("big", query="amount > 10").label({"amount": [1, 100]})
    assert result.mode == "tabular"
    assert result.labels == [None, "big"]


def test_unicode_text_and_keywords():
    texts = ["Un café, s'il vous plaît", "CAFÉ noir", "cafeteria", "naïve approach", "今日は東京へ行きます", "大阪です"]
    lab = Labeler().add_rule("coffee", keywords=["café"]).add_rule("naive", keywords=["naïve"]).add_rule("tokyo", keywords=["東京"])
    result = lab.label(texts)
    assert result.labels == ["coffee", "coffee", None, "naive", "tokyo", None]
    frame = result.to_frame()
    assert frame.loc[4, "item"] == "今日は東京へ行きます"
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_tabular_mixed_dtypes_all_nan_column_and_model():
    n = 12
    df = pd.DataFrame(
        {
            "amount": [5, 500, 7, 800, 9, 900, 4, 700, 6, 600, 8, 650],
            "kind": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "flag": [True, False] * 6,
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "note": ["x", None] * 6,
            "empty": [np.nan] * n,
            "cat": pd.Categorical(["p", "q"] * 6),
        }
    )
    df.loc[10, "amount"] = np.nan
    lab = Labeler(min_confidence=0.5)
    lab.add_rule("small", query="amount < 10 and index < 8")
    lab.add_rule("large", query="amount > 100 and index < 8")
    result = lab.label(df)
    assert result.mode == "tabular"
    assert result.labels[:8] == ["small", "large"] * 4
    assert result.source[:8] == ["rule"] * 8
    assert result.model_trained is True
    assert result.source[8:] == ["model"] * 4
    assert result.labels[8:] == ["small", "large", "small", "large"]
    frame = result.to_frame()
    assert list(frame.columns) == ["item", "label", "confidence", "source"]
    assert isinstance(frame.loc[0, "item"], dict)
    payload = json.dumps(result.to_dict())
    assert "2024-01-01" in payload


def test_tabular_keywords_see_string_columns_and_func_gets_row():
    df = pd.DataFrame({"text": ["Please refund me", "hello"], "amount": [10, 200], "free": [0, 1]})
    lab = Labeler()
    lab.add_rule("billing", keywords=["refund"])
    lab.add_rule("big", func=lambda row: row["amount"] > 100)
    result = lab.label(df)
    assert result.labels == ["billing", "big"]
    # a column *named* "free" must not trigger the keyword "free"
    assert Labeler().add_rule("spam", keywords=["free"]).label(df).labels == [None, None]


def test_labeler_constructor_validation_and_repr():
    with pytest.raises(ValueError):
        Labeler(min_confidence=1.5)
    with pytest.raises(ValueError):
        Labeler(min_confidence="high")
    with pytest.raises(ValueError):
        Labeler(labels=["", " "])
    lab = Labeler(labels="only")
    assert lab.labels == ["only"]
    assert "rules=0" in repr(lab)
    assert lab.candidates == ["only"]
    assert Labeler().candidates == []


def test_label_without_rules_warns_and_labels_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="auto_label"):
        result = Labeler().label(["a", "b"])
    assert result.labels == [None, None]
    assert "no rules" in caplog.text


# --- convenience function ---------------------------------------------------------------


def test_label_convenience_accepts_every_rule_form():
    rules = {
        "kw": ["alpha"],
        "rx": r"(?i)^beta",
        "fn": lambda t: t.endswith("!"),
        "dict": {"keywords": ["gamma"], "weight": 3},
        "many": [{"keywords": ["delta"]}, {"regex": "epsilon"}],
    }
    result = label(["alpha", "Beta test", "wow!", "gamma", "delta", "epsilon", "none"], rules, model=False)
    assert result.labels == ["kw", "rx", "fn", "dict", "many", "many", None]
    assert result.candidates == ["kw", "rx", "fn", "dict", "many"]


def test_label_convenience_rejects_bad_specs():
    with pytest.raises(TypeError):
        label(["a"], {"x": 42})
    with pytest.raises(TypeError):
        label(["a"], ["not", "a", "mapping"])
    with pytest.raises(ValueError):
        label(["a"], {"x": {"weight": 2}})


def test_label_convenience_forwards_options():
    seen = []
    result = label(["no idea"], RULES, labels=["spam", "work"], min_confidence=0.9, random_state=7,
                   llm=lambda t, c: seen.append(c) or "work")
    assert seen == [["spam", "work"]]
    assert result.labels == ["work"]
    assert isinstance(result, LabelResult)
    assert auto_label.__version__ == "0.1.0"


# --- result object -------------------------------------------------------------------


def test_result_frame_dict_and_summary():
    result = spam_work_labeler().label(["Free prize", "Team meeting", "no idea"])
    frame = result.to_frame()
    assert frame["label"].tolist() == ["spam", "work", None]
    assert frame["confidence"].tolist() == [1.0, 1.0, 0.0]
    assert frame["source"].tolist() == ["rule", "rule", None]
    assert frame["item"].tolist() == ["Free prize", "Team meeting", "no idea"]

    d = result.to_dict()
    assert d["n_items"] == 3 and d["n_labeled"] == 2
    assert d["coverage"] == pytest.approx(2 / 3)
    assert d["labels"] == {"spam": 1, "work": 1}
    assert d["sources"] == {"rule": 2, "model": 0, "llm": 0, "unlabeled": 1}
    assert d["candidates"] == ["spam", "work"]
    assert d["records"][2] == {"index": 2, "item": "no idea", "label": None, "confidence": 0.0, "source": None}
    json.dumps(d)

    text = result.summary()
    assert "2/3 items labeled (66.7%)" in text
    assert "rule 2" in text and "unlabeled 1" in text
    assert "spam 1" in text and "work 1" in text
    assert result.counts == {"spam": 1, "work": 1}
    assert len(result) == 3


# --- conventions: duplicate columns and degenerate auto-detection ----------------------


def test_duplicate_column_names_raise_a_clear_value_error():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column names: 'a'"):
        spam_work_labeler().label(df)


def test_table_with_no_text_column_falls_back_and_records_a_note(caplog):
    df = pd.DataFrame({"amount": [1, 22, 333], "other": [4, 5, 6]})
    lab = Labeler().add_rule("two", keywords=["22"])
    with caplog.at_level(logging.WARNING, logger="auto_label"):
        result = lab.label(df)
    assert result.labels == [None, "two", None]
    assert any("no text-like column" in n for n in result.notes)
    assert "no text-like column" in caplog.text


def test_no_text_column_fallback_is_silent_without_keyword_or_regex_rules():
    df = pd.DataFrame({"amount": [1, 200]})
    result = Labeler().add_rule("big", query="amount > 100").label(df)
    assert result.labels == [None, "big"]
    assert not any("no text-like column" in n for n in result.notes)


def test_to_dict_is_json_safe_for_exotic_cell_types():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-03-01", "2024-03-02"]),
            "took": pd.to_timedelta(["1 days", "2 days"]),
            "ratio": [np.float64("inf"), np.float64(1.5)],
            "flag": np.array([True, False]),
            "note": ["ok", None],
        }
    )
    result = Labeler().add_rule("first", query="ratio > 1.4 and index == 0").label(df)
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    first = payload["records"][0]["item"]
    assert first["when"] == "2024-03-01T00:00:00"
    assert first["ratio"] is None  # inf is not JSON-safe
    assert first["flag"] is True
    assert payload["records"][1]["item"]["note"] is None


def test_model_attribute_is_reset_between_calls():
    lab = spam_work_labeler(min_confidence=0.5)
    assert lab.model_ is None
    lab.label(SPAM + WORK + UNSEEN)
    assert lab.model_ is not None
    lab.label(["Free prize", "Team meeting"])  # too few examples, model skipped
    assert lab.model_ is None
