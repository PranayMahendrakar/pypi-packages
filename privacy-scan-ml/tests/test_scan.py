"""Column scanning: what gets flagged, what must not, and the awkward frames."""
import time

import numpy as np
import pandas as pd
import pytest

from privacy_scan_ml import PIIScanner, scan


def test_every_detector_fires_on_the_customer_table(customers):
    report = scan(customers)
    primary = {name: finding.primary for name, finding in report.columns.items()}
    assert primary["full_name"] == "person_name"
    assert primary["email"] == "email"
    assert primary["phone"] == "phone"
    assert primary["aadhaar"] == "aadhaar"
    assert primary["pan"] == "pan"
    assert primary["card_number"] == "credit_card"
    assert primary["ip_address"] == "ipv4"
    assert primary["website"] == "url"
    assert primary["dob"] == "date_of_birth"
    assert primary["pincode"] == "postal_code"
    assert primary["address"] == "address"
    assert primary["gender"] == "gender"
    assert report.risk == "high" and report.has_pii


def test_amount_and_id_columns_are_never_personal_data(customers):
    report = scan(customers)
    assert "order_amount" not in report.columns
    assert "order_id" not in report.columns


def test_a_clean_frame_is_reported_clean(clean_frame):
    report = scan(clean_frame)
    assert report.has_pii is False
    assert report.risk == "none"
    assert report.columns == {} and report.pii_columns == []
    assert report.n_columns == 4 and report.rows == 3


# --------------------------------------------------------------------- the phone guard


def _numeric(name, start=9876543210, n=200):
    return pd.DataFrame({name: [start + i for i in range(n)]})


@pytest.mark.parametrize("name", ["user_id", "transaction_id", "order_no", "row_number",
                                  "account_number", "invoice_amount", "total_amount"])
def test_phone_shaped_numbers_under_an_id_or_amount_header_are_not_phones(name):
    """The exact false positive the guard exists for: a 10-digit numeric row id."""
    report = scan(_numeric(name))
    flagged = report.columns.get(name)
    assert flagged is None or "phone" not in flagged.types, f"{name} was called a phone"


@pytest.mark.parametrize("name", ["mobile", "phone", "contact_number", "mobile_no", "whatsapp"])
def test_a_real_phone_column_is_still_found_even_when_stored_as_an_integer(name):
    report = scan(_numeric(name))
    assert report.columns[name].primary == "phone"


def test_a_string_phone_column_without_a_hinting_name_still_needs_phone_shaped_digits():
    good = pd.DataFrame({"alt_contact": [f"98765{i:05d}" for i in range(50)]})
    assert scan(good).columns["alt_contact"].primary == "phone"
    # five-digit codes are not phone-shaped, whatever else is true
    short = pd.DataFrame({"alt_code": [f"{10000 + i}" for i in range(50)]})
    assert "alt_code" not in scan(short).columns


def test_amounts_with_phone_shaped_digits_are_still_amounts():
    df = pd.DataFrame({"salary_amount": [9876543210 + i for i in range(50)]})
    assert "salary_amount" not in scan(df).columns


# --------------------------------------------------------------------- awkward frames


def test_empty_frame():
    report = scan(pd.DataFrame())
    assert report.has_pii is False and report.risk == "none"
    assert report.rows == 0 and report.columns == {}
    assert report.warnings and "nothing to scan" in report.warnings[0]
    assert isinstance(report.summary(), str) and isinstance(report.to_markdown(), str)


def test_frame_with_columns_but_no_rows():
    report = scan(pd.DataFrame({"email": pd.Series([], dtype=object)}))
    assert report.has_pii is False and report.rows == 0


def test_single_row_frame():
    report = scan(pd.DataFrame({"email": ["asha@example.com"], "qty": [3]}))
    assert report.pii_columns == ["email"]
    assert report.columns["email"].rows_checked == 1
    assert 0.0 < report.columns["email"].confidence <= 0.99


def test_all_nan_column_is_skipped_and_warned_about():
    report = scan(pd.DataFrame({"email": [None, None, np.nan], "qty": [1, 2, 3]}))
    assert report.pii_columns == []
    assert any("email" in w for w in report.warnings)


def test_a_column_named_like_pii_whose_values_never_validate_is_warned_about():
    report = scan(pd.DataFrame({"email": ["not-an-email", "also-not", "nope"]}))
    assert report.pii_columns == []
    assert any("email" in w and "no value validated" in w for w in report.warnings)


def test_mixed_dtypes_in_one_frame():
    df = pd.DataFrame(
        {
            "email": ["asha@example.com", "vikram@example.org"],
            "when": pd.to_datetime(["2020-01-01", "2020-06-01"]),
            "flag": [True, False],
            "score": [1.5, 2.5],
            "kind": pd.Categorical(["a", "b"]),
            "count": [1, 2],
        }
    )
    report = scan(df)
    assert report.pii_columns == ["email"]


def test_unicode_names_and_addresses_are_handled():
    df = pd.DataFrame(
        {
            "full_name": ["Aakanksha Rao", "अमित शर्मा", "李雷", "Zoë Müller"],
            "home_address": [
                "12 Brigade Road, Bengaluru",
                "45 एमजी Road, Pune",
                "78 Nehru Marg, Delhi",
                "9 Königsallee Street, Köln",
            ],
        }
    )
    report = scan(df)
    assert report.columns["full_name"].primary == "person_name"
    assert report.columns["home_address"].primary == "address"
    # the text outputs must not blow up on non-ASCII
    assert report.summary() and report.to_markdown() and report.to_dict()


def test_duplicate_column_names_raise_a_clear_value_error():
    df = pd.DataFrame([[1, 2]], columns=["email", "email"])
    with pytest.raises(ValueError, match="duplicate column names"):
        scan(df)


def test_min_share_controls_how_much_must_validate():
    values = ["asha@example.com"] + ["plain text"] * 9   # 10% validate
    df = pd.DataFrame({"note": values})
    assert "note" not in scan(df).columns
    assert "note" in scan(df, min_share=0.05).columns


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_min_share_is_range_checked(bad):
    with pytest.raises(ValueError, match="min_share"):
        scan(pd.DataFrame({"a": [1]}), min_share=bad)


def test_sample_is_range_checked():
    with pytest.raises(ValueError, match="sample"):
        scan(pd.DataFrame({"a": [1]}), sample=0)


def test_sampling_is_deterministic_and_reported(big_frame):
    a = scan(big_frame, sample=1_000)
    b = scan(big_frame, sample=1_000)
    assert a.rows == 50_000 and a.rows_scanned == 1_000
    assert a.to_dict() == b.to_dict()
    assert "1,000 of 50,000 rows scanned" in a.summary()


def test_fifty_thousand_rows_scan_in_a_few_seconds(big_frame):
    start = time.perf_counter()
    report = scan(big_frame)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"scanning 50k rows took {elapsed:.1f}s"
    assert report.pii_columns == ["email", "mobile"]


# --------------------------------------------------------------------- other inputs


def test_scan_accepts_a_series_a_dict_and_a_csv_path(tmp_path):
    series = pd.Series(["asha@example.com", "vikram@example.org"], name="email")
    assert scan(series).pii_columns == ["email"]
    assert scan({"email": ["asha@example.com"]}).pii_columns == ["email"]
    path = tmp_path / "people.csv"
    pd.DataFrame({"email": ["asha@example.com"]}).to_csv(path, index=False, encoding="utf-8")
    assert scan(str(path)).pii_columns == ["email"]


def test_scan_rejects_an_unsupported_file_type(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        scan(str(path))


def test_free_text_scanning_gives_spans_that_point_at_the_match():
    text = "write to asha@example.com or call 9876543210"
    report = scan(text)
    assert report.mode == "text" and report.documents == 1
    kinds = {f.type for f in report.findings}
    assert kinds == {"email", "phone"}
    # `text[start:end] in text` is true of any slice at all; name the values instead
    spans = {f.type: text[f.span[0] : f.span[1]] for f in report.findings}
    assert spans == {"email": "asha@example.com", "phone": "9876543210"}


def test_a_list_of_documents_keeps_the_document_index():
    report = scan(["asha@example.com", "nothing here", "vikram@example.org"])
    assert report.documents == 3
    assert sorted(f.doc for f in report.findings) == [0, 2]


def test_scanner_reuses_its_settings(customers):
    scanner = PIIScanner(sample=10, min_share=0.5)
    first, second = scanner.scan(customers), scanner.scan(customers)
    assert first.to_dict() == second.to_dict()
    assert first.min_share == 0.5
