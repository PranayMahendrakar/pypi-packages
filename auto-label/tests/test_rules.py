"""Rule construction and matching."""

import re

import pandas as pd
import pytest

from auto_label import Labeler, Rule
from auto_label._data import prepare
from auto_label.rules import compile_keywords, make_rule, match_rule


def hits(rule, data):
    return match_rule(rule, prepare(data)).tolist()


def test_keywords_are_whole_words_and_case_insensitive():
    rule = make_rule("pet", keywords=["cat"])
    assert hits(rule, ["a Cat!", "category", "CAT", "concatenate", "the cat sat"]) == [True, False, True, False, True]


def test_keyword_phrase_matches_any_whitespace_run():
    rule = make_rule("spam", keywords=["free money"])
    assert hits(rule, ["FREE   money now", "free\tmoney", "free, money", "freemoney"]) == [True, True, False, False]


def test_keyword_with_punctuation_edges():
    rule = make_rule("lang", keywords=["c++", "#promo"])
    assert hits(rule, ["I like c++ a lot", "c++x", "use #promo today", "c+"]) == [True, False, True, False]


def test_single_keyword_string_is_accepted():
    rule = make_rule("x", keywords="hello")
    assert rule.keywords == ("hello",)
    assert hits(rule, ["Hello there", "hell"]) == [True, False]


def test_compile_keywords_prefers_nothing_but_matches_all():
    pat = compile_keywords(["ab", "abc"])
    assert pat.search("xx abc yy") is not None
    assert pat.search("xx ab yy") is not None
    assert pat.search("xx abcd yy") is None


def test_regex_rule_is_searched_as_written():
    rule = make_rule("phone", regex=r"\d{3}-\d{4}")
    assert hits(rule, ["call 555-1234", "no digits"]) == [True, False]
    upper = make_rule("x", regex="ABC")
    assert hits(upper, ["abc"]) == [False]
    insensitive = make_rule("x", regex="(?i)ABC")
    assert hits(insensitive, ["abc"]) == [True]
    compiled = make_rule("x", regex=re.compile("z+"))
    assert hits(compiled, ["zzz", "y"]) == [True, False]


def test_invalid_regex_raises_value_error():
    with pytest.raises(ValueError, match="invalid regex"):
        make_rule("x", regex="(unclosed")
    with pytest.raises(ValueError):
        make_rule("x", regex="")
    with pytest.raises(ValueError):
        make_rule("x", regex=123)


def test_rule_needs_at_least_one_condition():
    with pytest.raises(ValueError, match="at least one"):
        make_rule("x")
    with pytest.raises(ValueError, match="empty"):
        make_rule("x", keywords=[])
    with pytest.raises(ValueError, match="empty"):
        make_rule("x", keywords=["", "  "])


def test_label_and_weight_validation():
    with pytest.raises(ValueError, match="label"):
        make_rule("", keywords=["a"])
    with pytest.raises(ValueError, match="label"):
        make_rule(None, keywords=["a"])
    with pytest.raises(ValueError, match="weight"):
        make_rule("x", keywords=["a"], weight=float("nan"))
    with pytest.raises(ValueError, match="weight"):
        make_rule("x", keywords=["a"], weight="heavy")
    with pytest.raises(ValueError, match="weight"):
        make_rule("x", keywords=["a"], weight=True)
    with pytest.raises(ValueError, match="query"):
        make_rule("x", query="   ")
    with pytest.raises(ValueError, match="callable"):
        make_rule("x", func="not callable")


def test_query_rule_on_text_raises_clear_error():
    rule = make_rule("big", query="amount > 10")
    with pytest.raises(ValueError, match="needs a DataFrame"):
        match_rule(rule, prepare(["a", "b"]))


def test_query_rule_on_frame():
    df = pd.DataFrame({"amount": [5, 50, 500], "kind": ["a", "b", "a"]}, index=[10, 20, 30])
    rule = make_rule("big", query="amount > 10 and kind == 'a'")
    assert hits(rule, df) == [False, False, True]


def test_query_supports_string_methods():
    df = pd.DataFrame({"text": ["Refund please", "hello", "REFUND now"]})
    rule = make_rule("billing", query="text.str.contains('refund', case=False)")
    assert hits(rule, df) == [True, False, True]


def test_bad_query_raises_value_error_with_label():
    df = pd.DataFrame({"amount": [1, 2]})
    rule = make_rule("big", query="nope > 1")
    with pytest.raises(ValueError, match="'big'"):
        match_rule(rule, prepare(df))


def test_func_rule_gets_text_or_row():
    text_rule = make_rule("long", func=lambda t: len(t) > 5)
    assert hits(text_rule, ["short", "long enough"]) == [False, True]

    df = pd.DataFrame({"amount": [1, 100], "name": ["a", "b"]})
    row_rule = make_rule("big", func=lambda row: row["amount"] > 10 and row.name >= 0)
    assert hits(row_rule, df) == [False, True]


def test_conditions_are_combined_with_and():
    rule = make_rule("x", keywords=["alpha"], func=lambda t: "beta" in t)
    assert hits(rule, ["alpha beta", "alpha", "beta"]) == [True, False, False]
    assert rule.conditions == ["keywords", "func"]


def test_describe_and_repr():
    rule = make_rule("spam", keywords=["free"], regex="x", query="a > 1", func=len, weight=2)
    text = rule.describe()
    assert text.startswith("spam:")
    assert "keywords=('free',)" in text
    assert "regex='x'" in text
    assert "query='a > 1'" in text
    assert "func=len" in text
    assert "weight=2" in text
    assert isinstance(rule, Rule)
    assert "spam" in repr(rule)


def test_empty_input_mask_is_empty():
    rule = make_rule("x", keywords=["a"], func=len)
    assert hits(rule, []) == []


def test_labeler_add_rule_is_chainable_and_validates_labels():
    labeler = Labeler(labels=["a", "b"])
    assert labeler.add_rule("a", keywords=["x"]).add_rule("b", regex="y") is labeler
    assert [r.label for r in labeler.rules] == ["a", "b"]
    with pytest.raises(ValueError, match="not in labels"):
        labeler.add_rule("c", keywords=["z"])
