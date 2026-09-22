"""Cell-level parsers: numbers, booleans, missing tokens and day-first inference."""
import numpy as np
import pandas as pd
import pytest

from smartclean_df.parsing import (
    infer_dayfirst,
    is_missing_token,
    is_na_scalar,
    looks_like_date,
    parse_bool,
    parse_number,
    to_datetime_quiet,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1,234", 1234.0),
        ("$12.50", 12.5),
        ("€1.234,00", None),
        ("45%", 45.0),
        ("  7 ", 7.0),
        ("(300)", -300.0),
        ("-$5", -5.0),
        ("$-5", -5.0),
        ("1e3", 1000.0),
        (".5", 0.5),
        ("0.5", 0.5),
        ("00123", None),
        ("12,34", None),
        ("abc", None),
        ("", None),
        ("(3", None),
        ("+-3", None),
        ("£1,000,000.25", 1000000.25),
        ("₹99", 99.0),
    ],
)
def test_parse_number_strings(text, expected):
    assert parse_number(text) == expected


def test_parse_number_non_strings():
    assert parse_number(3) == 3.0
    assert parse_number(np.int64(4)) == 4.0
    assert parse_number(2.5) == 2.5
    assert parse_number(float("nan")) is None
    assert parse_number(True) is None
    assert parse_number(None) is None
    assert parse_number([1]) is None


@pytest.mark.parametrize(
    "value, expected",
    [
        ("yes", True),
        (" No ", False),
        ("TRUE", True),
        ("f", False),
        ("1", True),
        ("0", False),
        (True, True),
        (np.bool_(False), False),
        ("maybe", None),
        (1, None),
        (None, None),
    ],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_missing_tokens_and_na_scalars():
    for token in ["", "NA", "n/a", "Null", "NONE", "-", "?", "nan", "  na  "]:
        assert is_missing_token(token)
    assert not is_missing_token("nah")
    assert is_na_scalar(None) and is_na_scalar(float("nan")) and is_na_scalar(pd.NaT) and is_na_scalar(pd.NA)
    assert is_na_scalar(np.datetime64("NaT"))
    assert not is_na_scalar("") and not is_na_scalar(0)


def test_looks_like_date_and_dayfirst():
    assert looks_like_date("2024-01-05") and looks_like_date("5 Jan 2024") and looks_like_date("Jan 5, 2024")
    assert looks_like_date(pd.Timestamp("2024-01-01"))
    assert not looks_like_date("hello") and not looks_like_date(5) and not looks_like_date("2024")
    assert infer_dayfirst(["05/01/2024", "13/02/2024"]) is True
    assert infer_dayfirst(["05/01/2024", "02/13/2024"]) is False
    assert infer_dayfirst(["2024-01-05", "x", 3]) is False


def test_to_datetime_quiet_never_raises():
    series = pd.Series(["2024-01-05", "garbage", "13/02/2024"], dtype=object)
    parsed = to_datetime_quiet(series, dayfirst=True)
    assert parsed is not None and parsed.isna().sum() >= 1
    weird = to_datetime_quiet(pd.Series([object(), object()]))
    assert weird is None or bool(weird.isna().all())
