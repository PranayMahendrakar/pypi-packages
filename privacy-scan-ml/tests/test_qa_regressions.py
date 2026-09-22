"""Regressions for what independent QA found. One test per finding, named after it.

The two that matter most are at the top: a column the scan flags that masking quietly
left alone, and an identifier column the scan called a phone number because of how it
happened to be stored.
"""
import json
import re

import pandas as pd
import pytest

from privacy_scan_ml import SENSITIVE_TYPES, PIIScanner, mask, scan
from privacy_scan_ml._mask import WARNINGS_KEY
from privacy_scan_ml._validators import pan_valid
from privacy_scan_ml.cli import main

# --------------------------------------------------------------------------- blocker
# mask() left date_of_birth and postal_code columns byte-identical, under every strategy,
# while the report printed in the same run said those columns carried personal data.

CONTEXT_GATED = pd.DataFrame(
    {
        "dob": ["1990-05-14", "1985-02-14", "1977-03-03"],
        "pincode": ["560087", "110011", "400051"],
        "email": ["asha@example.com", "vikram@example.org", "meera@example.net"],
    }
)


@pytest.mark.parametrize("strategy", ["redact", "hash", "partial"])
def test_a_flagged_dob_or_postal_column_is_actually_masked(strategy):
    out = mask(CONTEXT_GATED, strategy=strategy, salt="pepper")
    for column in ("dob", "pincode"):
        original = list(CONTEXT_GATED[column])
        assert list(out[column]) != original, f"{column} came back unmasked under {strategy}"
        for value in original:
            assert value not in list(out[column])


def test_masking_a_flagged_frame_leaves_nothing_the_scan_would_still_flag():
    """The guard the old fixture was too narrow to catch: it held no gated types."""
    assert scan(CONTEXT_GATED).pii_columns == ["dob", "pincode", "email"]
    assert scan(mask(CONTEXT_GATED)).pii_columns == []


def test_naming_the_gated_columns_explicitly_masks_them_too():
    out = mask(CONTEXT_GATED, columns=["dob", "pincode"])
    assert list(out["dob"]) == ["[DATE_OF_BIRTH]"] * 3
    assert list(out["pincode"]) == ["[POSTAL_CODE]"] * 3


def test_free_text_that_is_nothing_but_a_date_or_a_pin_is_masked():
    """scan() flags a whole-value date; mask() has to agree, or the two contradict."""
    assert [f.type for f in scan("1990-05-14").findings] == ["date_of_birth"]
    assert mask("1990-05-14") == "[DATE_OF_BIRTH]"
    assert [f.type for f in scan("560087").findings] == ["postal_code"]
    assert mask("560087") == "[POSTAL_CODE]"


def test_the_cli_mask_file_has_no_raw_dates_or_pins(tmp_path, capsys):
    source = tmp_path / "in.csv"
    CONTEXT_GATED.to_csv(source, index=False, encoding="utf-8")
    clean = tmp_path / "out.csv"
    assert main([str(source), "--mask", str(clean)]) == 0
    capsys.readouterr()
    body = clean.read_text(encoding="utf-8")
    for column in CONTEXT_GATED.columns:
        for value in CONTEXT_GATED[column]:
            assert value not in body, f"{value} survived --mask"


def test_a_column_masking_could_not_touch_is_never_silent():
    """The safety net: if masking ever changes nothing, it has to say so."""
    scanner = PIIScanner()
    out = scanner.mask_frame(CONTEXT_GATED, columns=["dob"])
    assert scanner.last_warnings == [] and out.attrs[WARNINGS_KEY] == []

    already = pd.DataFrame({"email": ["[EMAIL]", "[EMAIL]"]})
    out = scanner.mask_frame(already, columns=["email"])
    assert scanner.last_warnings and "email" in scanner.last_warnings[0]
    assert out.attrs[WARNINGS_KEY] == scanner.last_warnings


# --------------------------------------------------------------------------- major
# An ASCII-only email pattern meant mask() shipped unicode addresses verbatim.

UNICODE_EMAILS = [
    "andré@example.fr",
    "müller@example.de",
    "josé@example.es",
    "आशा@example.in",
    "李雷@example.cn",
    "user@münchen.de",
]


@pytest.mark.parametrize("address", UNICODE_EMAILS)
def test_a_unicode_or_idn_email_is_detected(address):
    assert [f.type for f in scan([address]).findings] == ["email"]


@pytest.mark.parametrize("address", UNICODE_EMAILS)
def test_a_unicode_or_idn_email_is_masked(address):
    masked = mask(f"write to {address} today")
    assert masked == "write to [EMAIL] today"
    assert address not in masked


def test_a_mixed_ascii_and_unicode_email_column_is_fully_flagged_and_fully_masked():
    df = pd.DataFrame({"email": ["asha@example.com", "müller@example.de", "आशा@example.in"]})
    finding = scan(df).columns["email"]
    assert finding.primary == "email" and finding.types["email"] == 1.0
    assert list(mask(df)["email"]) == ["[EMAIL]"] * 3


def test_ascii_email_behaviour_is_unchanged():
    """The widened pattern must not start swallowing things that are not addresses."""
    for good in ["Asha.Rao+tag@sub.example.co.uk", "a@b.io", "x_y%z@example-host.museum"]:
        assert [f.type for f in scan([good]).findings] == ["email"], good
    for bad in ["not-an-email", "a@@b.com", "@example.com", "asha@", "asha@example",
                "e-mail me", "price: 12@3"]:
        assert [f for f in scan([bad]).findings if f.type == "email"] == [], bad


# --------------------------------------------------------------------------- major
# The id/amount phone guard only fired on numeric dtype, so the same ids stored as
# strings were reported as phone numbers. README says user_id stays clean; it must.

ID_HEADERS = ["user_id", "transaction_id", "txn_ref", "employee_id", "policy_number",
              "serial_number", "account_number", "order_no", "row_number", "invoice_amount"]


@pytest.mark.parametrize("name", ID_HEADERS)
@pytest.mark.parametrize("as_text", [False, True], ids=["int64", "object"])
def test_an_id_header_is_never_a_phone_whatever_the_dtype(name, as_text):
    values = [9876543210 + i for i in range(60)]
    if as_text:
        values = [str(v) for v in values]
    finding = scan(pd.DataFrame({name: values})).columns.get(name)
    assert finding is None or "phone" not in finding.types, f"{name} was called a phone"


def test_a_realistic_finance_table_carries_no_personal_data():
    n = 60
    df = pd.DataFrame(
        {
            "txn_ref": [str(9000000000 + i) for i in range(n)],
            "debit": [100.5 + i for i in range(n)],
            "credit": [0.0] * n,
            "branch_code": [f"{560000 + i}" for i in range(n)],
        }
    )
    report = scan(df)
    assert report.pii_columns == [] and report.has_pii is False


@pytest.mark.parametrize("name", ["mobile", "phone", "contact_number", "mobile_no",
                                  "whatsapp", "alt_contact"])
@pytest.mark.parametrize("as_text", [False, True], ids=["int64", "object"])
def test_a_real_phone_column_is_still_found_either_way(name, as_text):
    values = [9876543210 + i for i in range(60)]
    if as_text:
        values = [str(v) for v in values]
    assert scan(pd.DataFrame({name: values})).columns[name].primary == "phone"


# --------------------------------------------------------------------------- major
# `--strategy hash` with no `--salt` generated a salt, used it and threw it away, so the
# masked file could never be reproduced and nothing said so.


def _hash_run(tmp_path, out_name, capsys, extra=()):
    source = tmp_path / "in.csv"
    pd.DataFrame({"email": ["asha@example.com", "b@example.org"]}).to_csv(
        source, index=False, encoding="utf-8"
    )
    out = tmp_path / out_name
    assert main([str(source), "--mask", str(out), "--strategy", "hash", *extra]) == 0
    captured = capsys.readouterr()
    return out.read_text(encoding="utf-8"), captured.out + captured.err


def test_cli_hash_without_a_salt_prints_the_salt_it_generated(tmp_path, capsys):
    first, printed = _hash_run(tmp_path, "out1.csv", capsys)
    assert "salt" in printed.lower(), "the generated salt was never reported"
    found = re.search(r"--salt\): ([0-9a-f]{8,})", printed)
    assert found, printed
    salt = found.group(1)

    second, _ = _hash_run(tmp_path, "out2.csv", capsys)
    assert second != first, "two unsalted runs must not silently share a mapping"

    again, _ = _hash_run(tmp_path, "out3.csv", capsys, ["--salt", salt])
    assert again == first, "the reported salt must reproduce the run it came from"


def test_cli_hash_with_an_explicit_salt_says_nothing_about_generating_one(tmp_path, capsys):
    _, printed = _hash_run(tmp_path, "out.csv", capsys, ["--salt", "s3cret"])
    assert "generated" not in printed


def test_cli_json_stdout_stays_parseable_when_a_salt_is_reported(tmp_path, capsys):
    source = tmp_path / "in.csv"
    pd.DataFrame({"email": ["asha@example.com"]}).to_csv(source, index=False, encoding="utf-8")
    out = tmp_path / "out.csv"
    assert main([str(source), "--mask", str(out), "--strategy", "hash", "--json"]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)  # must be one document, salt notice goes to stderr
    assert "salt" in captured.err.lower()


# --------------------------------------------------------------------------- minor


def test_a_missing_path_with_a_data_extension_raises_instead_of_reporting_clean():
    with pytest.raises(ValueError, match="unsupported file type"):
        scan("customers.xlsx")
    with pytest.raises(FileNotFoundError):
        scan("customers.csv")


def test_a_sentence_that_merely_mentions_a_file_is_still_free_text():
    assert scan("please check the report.docx I sent").mode == "text"
    assert scan("just some free text").mode == "text"


def test_a_missing_parquet_says_the_file_is_missing_not_that_pyarrow_is(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan(str(tmp_path / "nope.parquet"))


def test_an_unknown_column_is_a_value_error():
    with pytest.raises(ValueError, match="columns not in the frame"):
        mask(pd.DataFrame({"email": ["a@b.com"]}), columns=["missing"])


def test_a_dict_of_scalars_gets_our_message_not_a_pandas_one():
    with pytest.raises(ValueError, match="must map each name to a sequence") as exc:
        scan({"a": 1})
    assert "pass an index" not in str(exc.value)
    with pytest.raises(ValueError, match="must map each name to a sequence"):
        scan({"a": "x"})
    with pytest.raises(ValueError, match="same"):
        scan({"a": [1, 2], "b": [1]})


@pytest.mark.parametrize("strategy", ["redact", "hash"])
def test_a_non_string_salt_names_the_parameter(strategy):
    with pytest.raises(TypeError, match="salt must be a str"):
        mask(pd.DataFrame({"email": ["a@b.com"]}), strategy=strategy, salt=123)


def test_sensitive_columns_can_be_asked_for_without_hard_coding_the_list():
    report = scan(pd.DataFrame({"gender": ["female", "male"], "religion": ["Hindu", "Sikh"]}))
    assert set(report.types) <= set(SENSITIVE_TYPES)
    assert [c for c, f in report.columns.items() if f.category == "sensitive_category"] == [
        "gender",
        "religion",
    ]
    assert report.to_dict()["columns"]["gender"]["category"] == "sensitive_category"


@pytest.mark.parametrize(
    "line", ["221B Baker Street", "12A Nehru Road", "5B Brigade Road", "42 Maple Street"]
)
def test_an_alphanumeric_house_number_still_reads_as_an_address(line):
    assert "address" in {f.type for f in scan([line]).findings}, line


def test_lowercase_pan_is_detected_and_masked():
    assert pan_valid("abcpd1234e") and not pan_valid("AbcPd1234E")
    df = pd.DataFrame({"pan": ["abcpd1234e", "xyzpk9876f", "aaact1234z"]})
    assert scan(df).columns["pan"].primary == "pan"
    assert list(mask(df)["pan"]) == ["[PAN]"] * 3
    assert [f.type for f in scan("my pan is abcpd1234e ok").findings] == ["pan"]


def test_an_empty_text_argument_is_not_reported_as_giving_both_or_neither(capsys):
    assert main(["--text", ""]) == 0
    assert "no personal data found" in capsys.readouterr().out


def test_free_text_spans_bound_the_value_they_name():
    """The old assertion, text[start:end] in text, was true of every possible span."""
    text = "write to asha@example.com or call 9876543210 about 2345 6789 0124"
    found = {f.type: text[f.span[0] : f.span[1]] for f in scan(text).findings}
    assert found["email"] == "asha@example.com"
    assert found["phone"] == "9876543210"
    assert found["aadhaar"] == "2345 6789 0124"


def test_no_masked_value_leaks_through_the_gated_types():
    report = scan(CONTEXT_GATED)
    blob = "\n".join(
        [report.summary(), json.dumps(report.to_dict(), ensure_ascii=False), report.to_markdown()]
    )
    for column in CONTEXT_GATED.columns:
        for value in CONTEXT_GATED[column]:
            assert value not in blob
