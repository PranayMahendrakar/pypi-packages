"""Every identifier is validated, not just pattern-matched."""
import pytest

from privacy_scan_ml._validators import (
    aadhaar_valid,
    credit_card_valid,
    email_valid,
    ipv4_valid,
    ipv6_valid,
    luhn_valid,
    pan_valid,
    parse_ambiguous,
    phone_valid,
    plausible_birth_date,
    url_valid,
    verhoeff_check_digit,
    verhoeff_valid,
)
from conftest import (
    INVALID_AADHAAR,
    INVALID_CARDS,
    INVALID_PANS,
    VALID_AADHAAR,
    VALID_CARDS,
    VALID_PANS,
)


@pytest.mark.parametrize("number", VALID_AADHAAR)
def test_aadhaar_known_valid(number):
    assert verhoeff_valid(number)
    assert aadhaar_valid(number)
    assert aadhaar_valid(f"{number[:4]} {number[4:8]} {number[8:]}")
    assert aadhaar_valid(f"{number[:4]}-{number[4:8]}-{number[8:]}")


@pytest.mark.parametrize("number", INVALID_AADHAAR)
def test_aadhaar_known_invalid(number):
    assert not aadhaar_valid(number)


def test_aadhaar_rejects_repeated_digits_and_leading_0_or_1():
    assert not aadhaar_valid("222222222222")
    assert not aadhaar_valid("0" * 12)
    # a Verhoeff-correct number that starts with 1 is still not an Aadhaar
    base = "12345678901"
    assert verhoeff_valid(base + verhoeff_check_digit(base))
    assert not aadhaar_valid(base + verhoeff_check_digit(base))


def test_verhoeff_check_digit_round_trip():
    for base in ["23456789012", "98765432109", "40000000000"]:
        assert verhoeff_valid(base + verhoeff_check_digit(base))
        wrong = str((int(verhoeff_check_digit(base)) + 1) % 10)
        assert not verhoeff_valid(base + wrong)


@pytest.mark.parametrize("number", VALID_CARDS)
def test_card_luhn_valid(number):
    assert luhn_valid(number)
    assert credit_card_valid(number)
    spaced = " ".join(number[i : i + 4] for i in range(0, len(number), 4))
    assert credit_card_valid(spaced)


@pytest.mark.parametrize("number", INVALID_CARDS)
def test_card_luhn_invalid(number):
    assert not credit_card_valid(number)


@pytest.mark.parametrize("value", VALID_PANS)
def test_pan_valid(value):
    assert pan_valid(value)


@pytest.mark.parametrize("value", INVALID_PANS)
def test_pan_invalid(value):
    assert not pan_valid(value)


def test_pan_fourth_letter_is_the_holder_type():
    for letter in "PCHFATBLJG":
        assert pan_valid(f"ABC{letter}D1234E"), letter
    for letter in "DEIKMNOQRSUVWXYZ":
        assert not pan_valid(f"ABC{letter}D1234E"), letter


@pytest.mark.parametrize(
    "value",
    ["9876543210", "6123456789", "+91 98765 43210", "98765-43210", "09876543210",
     "+1 415 555 2671", "415-555-2671", "+44 20 7946 0958"],
)
def test_phone_valid(value):
    assert phone_valid(value)


@pytest.mark.parametrize(
    "value", ["1234567890", "5876543210", "1299", "45999", "0000000000", "12", "123456789012345678"]
)
def test_phone_invalid(value):
    assert not phone_valid(value)


@pytest.mark.parametrize(
    "value", ["asha@example.com", "a.b+tag@sub.example.co.in", "x_1@example-site.org"]
)
def test_email_valid(value):
    assert email_valid(value)


@pytest.mark.parametrize(
    "value", ["asha@example", "asha@@example.com", "@example.com", "asha@.com", "a b@example.com",
              ".asha@example.com", "asha@example.123"]
)
def test_email_invalid(value):
    assert not email_valid(value)


def test_ip_validation():
    assert ipv4_valid("192.168.1.1") and ipv4_valid("8.8.8.8") and ipv4_valid("255.255.255.255")
    assert not ipv4_valid("256.1.1.1")
    assert not ipv4_valid("192.168.01.1")
    assert not ipv4_valid("1.2.3")
    assert ipv6_valid("2001:db8::1") and ipv6_valid("::1")
    assert not ipv6_valid("2001:db8") and not ipv6_valid("hello")


def test_url_validation():
    assert url_valid("https://example.com/a?b=1")[0]
    assert url_valid("www.example.org")[0]
    assert not url_valid("justtext")[0]
    ok, cleaned = url_valid("https://example.com.")
    assert ok and cleaned == "https://example.com"


def test_birth_dates():
    from datetime import date

    assert parse_ambiguous("14", "02", "1985") == date(1985, 2, 14)
    assert parse_ambiguous("02", "14", "1985") == date(1985, 2, 14)  # falls back to month-first
    assert parse_ambiguous("31", "02", "1985") is None
    assert plausible_birth_date(date(1985, 2, 14))
    assert not plausible_birth_date(date(date.today().year + 1, 1, 1))
    assert not plausible_birth_date(date(1700, 1, 1))
