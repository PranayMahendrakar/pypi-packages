"""parse_number: spoken English numbers, and refusing things that are not numbers."""

import pytest

from voice_commands_ai import parse_number


@pytest.mark.parametrize(
    "text, expected",
    [
        # the three the brief names
        ("a hundred and five", 105),
        ("two point five", 2.5),
        ("minus three", -3),
        # cardinals
        ("zero", 0),
        ("seven", 7),
        ("thirteen", 13),
        ("twenty one", 21),
        ("twenty-one", 21),
        ("ninety nine", 99),
        ("one hundred", 100),
        ("hundred", 100),
        ("one hundred one", 101),
        ("twelve hundred", 1200),
        ("twenty five hundred", 2500),
        ("a thousand and five", 1005),
        ("two thousand and twenty six", 2026),
        ("one thousand two hundred and thirty four", 1234),
        ("three million four hundred thousand", 3_400_000),
        ("a dozen", 12),
        ("two dozen", 24),
        # digits and mixed forms
        ("21", 21),
        ("-3", -3),
        ("\u22123", -3),  # unicode minus sign
        ("+4", 4),
        ("2.5", 2.5),
        ("1,000", 1000),
        ("2,5", 2.5),  # decimal comma
        ("21 thousand", 21_000),
        ("1.5 million", 1_500_000),
        ("\uff12\uff11", 21),  # full-width digits
        # signs and decimals
        ("negative two point five", -2.5),
        ("minus twenty one", -21),
        ("plus five", 5),
        ("point five", 0.5),
        ("three point one four", 3.14),
        ("one point twenty five", 1.25),
        ("zero point oh five", 0.05),
        # fractions
        ("two and a half", 2.5),
        ("a half", 0.5),
        ("a quarter", 0.25),
        ("three quarters", 0.75),
        ("one and a quarter", 1.25),
        # ordinals
        ("first", 1),
        ("third", 3),
        ("twenty first", 21),
        ("21st", 21),
        ("3rd", 3),
        ("hundredth", 100),
        # digit-by-digit and years
        ("four five six", 456),
        ("one oh one", 101),
        ("nineteen eighty four", 1984),
        ("twenty twenty six", 2026),
        ("nineteen oh five", 1905),
        # case, punctuation and hesitations
        ("TWENTY One", 21),
        ("A Hundred And Five", 105),
        ("twenty, um, one", 21),
        ("  twenty one!  ", 21),
    ],
)
def test_spoken_numbers(text, expected):
    value = parse_number(text)
    assert isinstance(value, float)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello",
        "twenty one degrees",  # the whole text has to be the number
        "the twenty",
        "five hundred hundred",
        "thousand thousand",
        "twenty and five",
        "one hundred and",
        "minus",
        "point",
        "and",
        "a",
        "minus -3",
        "um",
        "\u2615",
    ],
)
def test_not_numbers(text):
    assert parse_number(text) is None


def test_zero_is_not_none():
    assert parse_number("zero") == 0.0
    assert parse_number("zero") is not None


def test_unicode_digits_from_other_scripts():
    # Arabic-Indic digits are decimal digits too
    assert parse_number("\u0662\u0661") == 21.0


def test_non_string_is_a_type_error():
    with pytest.raises(TypeError, match="str"):
        parse_number(21)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_number(None)  # type: ignore[arg-type]
