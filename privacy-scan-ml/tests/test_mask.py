"""Masking: the three strategies, the salt contract, and keeping the frame intact."""
import pandas as pd
import pytest

from privacy_scan_ml import PIIScanner, mask, scan
from privacy_scan_ml._mask import SALT_KEY


@pytest.fixture
def people():
    return pd.DataFrame(
        {
            "email": ["asha@example.com", "vikram@example.org"],
            "phone": ["9876543210", "9123456780"],
            "order_amount": [1299, 45999],
            "units": [2, 7],
        }
    )


# --------------------------------------------------------------------------- strategies


def test_redact_replaces_matches_with_a_type_token(people):
    out = mask(people)
    assert list(out["email"]) == ["[EMAIL]", "[EMAIL]"]
    assert list(out["phone"]) == ["[PHONE]", "[PHONE]"]


def test_partial_keeps_only_the_last_four_characters(people):
    out = mask(people, strategy="partial")
    for value in out["phone"]:
        assert value.endswith(("3210", "6780"))
        assert value[:6] == "******"
    for original, value in zip(people["email"], out["email"]):
        assert value != original and value.endswith(original[-4:])


def test_hash_is_a_twelve_character_digest(people):
    out = mask(people, strategy="hash", salt="pepper")
    for value in out["email"]:
        assert len(value) == 12 and all(c in "0123456789abcdef" for c in value)
    assert out["email"].iloc[0] != out["email"].iloc[1]


def test_hash_with_the_same_salt_is_stable_across_calls(people):
    a = mask(people, strategy="hash", salt="pepper")
    b = mask(people, strategy="hash", salt="pepper")
    assert list(a["email"]) == list(b["email"])


def test_an_unknown_strategy_is_rejected(people):
    with pytest.raises(ValueError, match="strategy must be one of"):
        mask(people, strategy="shred")


# --------------------------------------------------------------------------- the salt contract


def test_hash_without_a_salt_generates_one_and_records_it_on_the_frame(people):
    out = mask(people, strategy="hash")
    salt = out.attrs[SALT_KEY]
    assert isinstance(salt, str) and len(salt) >= 16
    again = mask(people, strategy="hash", salt=salt)
    assert list(again["email"]) == list(out["email"]), "the recorded salt must reproduce it"


def test_two_unsalted_hash_calls_do_not_accidentally_share_a_mapping(people):
    first, second = mask(people, strategy="hash"), mask(people, strategy="hash")
    assert first.attrs[SALT_KEY] != second.attrs[SALT_KEY]
    assert list(first["email"]) != list(second["email"])


def test_masked_text_carries_the_salt_too():
    out = mask("write to asha@example.com", strategy="hash")
    assert isinstance(out, str) and out.salt
    assert mask("write to asha@example.com", strategy="hash", salt=out.salt) == out


def test_a_masked_list_carries_the_salt():
    out = mask(["asha@example.com", "vikram@example.org"], strategy="hash")
    assert isinstance(out, list) and len(out) == 2 and out.salt


def test_strategies_without_a_salt_record_nothing(people):
    assert SALT_KEY not in mask(people).attrs
    assert SALT_KEY not in mask(people, strategy="partial").attrs
    assert getattr(mask("asha@example.com"), "salt", None) is None


def test_the_scanner_remembers_the_salt_it_last_used(people):
    scanner = PIIScanner()
    out = scanner.mask(people, strategy="hash")
    assert scanner.last_salt == out.attrs[SALT_KEY]


# --------------------------------------------------------------------------- shape and dtype


def test_masking_preserves_dtypes_of_columns_it_did_not_touch(people):
    out = mask(people)
    assert out["order_amount"].dtype == people["order_amount"].dtype
    assert out["units"].dtype == people["units"].dtype
    assert list(out["order_amount"]) == [1299, 45999]


def test_a_categorical_column_stays_categorical():
    df = pd.DataFrame({"gender": pd.Categorical(["F", "M", "F"])})
    out = mask(df)
    assert isinstance(out["gender"].dtype, pd.CategoricalDtype)


def test_missing_values_survive_masking():
    df = pd.DataFrame({"email": ["asha@example.com", None, float("nan")]})
    out = mask(df)
    assert out["email"].iloc[0] == "[EMAIL]"
    assert pd.isna(out["email"].iloc[1]) and pd.isna(out["email"].iloc[2])


def test_the_input_frame_is_not_modified(people):
    before = people.copy()
    mask(people)
    pd.testing.assert_frame_equal(people, before)


def test_masking_returns_the_type_that_went_in(tmp_path):
    series = pd.Series(["asha@example.com"], name="email")
    out = mask(series)
    assert isinstance(out, pd.Series) and out.name == "email" and out.iloc[0] == "[EMAIL]"

    assert isinstance(mask("asha@example.com"), str)
    assert isinstance(mask(["asha@example.com"]), list)
    assert isinstance(mask(("asha@example.com",)), tuple)

    path = tmp_path / "people.csv"
    pd.DataFrame({"email": ["asha@example.com"]}).to_csv(path, index=False, encoding="utf-8")
    out = mask(str(path))
    assert isinstance(out, pd.DataFrame) and out["email"].iloc[0] == "[EMAIL]"


def test_free_text_keeps_its_wording_around_the_match():
    out = mask("please write to asha@example.com before friday")
    assert out == "please write to [EMAIL] before friday"


def test_unicode_around_a_match_is_preserved():
    out = mask("अमित को asha@example.com पर लिखें")
    assert out == "अमित को [EMAIL] पर लिखें"


# --------------------------------------------------------------------------- column choice


def test_columns_limits_what_is_masked(people):
    out = mask(people, columns=["email"])
    assert list(out["email"]) == ["[EMAIL]", "[EMAIL]"]
    assert list(out["phone"]) == list(people["phone"]), "phone was not asked for"


def test_masking_a_column_the_scan_did_not_flag_still_masks_it(people):
    out = mask(people, columns=["units"])
    assert list(out["units"]) == ["[PII]", "[PII]"]


def test_an_unknown_column_is_a_clear_value_error(people):
    # caller error, so ValueError: `except ValueError` around the public API must catch it
    with pytest.raises(ValueError, match="columns not in the frame"):
        mask(people, columns=["nope"])


def test_columns_is_meaningless_for_free_text():
    with pytest.raises(ValueError, match="columns is only meaningful"):
        mask("asha@example.com", columns=["email"])


def test_non_string_column_names_are_handled():
    df = pd.DataFrame({5: ["asha@example.com", "vikram@example.org"]})
    assert list(mask(df)[5]) == ["[EMAIL]", "[EMAIL]"]


def test_masking_a_frame_leaves_nothing_the_scan_would_still_flag(people):
    assert scan(mask(people)).pii_columns == []
