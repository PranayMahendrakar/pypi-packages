"""The README quickstart, verbatim, is the contract."""
import pandas as pd

from privacy_scan_ml import mask, scan


def test_readme_quickstart_runs():
    df = pd.DataFrame({
        "full_name": ["Asha Rao", "Vikram Nair"],
        "email": ["asha@example.com", "vikram@example.org"],
        "phone": ["9876543210", "+91 91234 56780"],
        "order_amount": [1299, 45999],
    })
    report = scan(df)
    text = report.summary()
    assert isinstance(text, str) and "privacy-scan-ml" in text
    assert report.risk == "high"
    assert report.has_pii is True
    assert report.pii_columns == ["full_name", "email", "phone"]
    masked = mask(df)
    assert list(masked["email"]) == ["[EMAIL]", "[EMAIL]"]
    assert list(masked["order_amount"]) == [1299, 45999]


def test_readme_quickstart_matches_the_file():
    """The block in README.md is the one the test above runs."""
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    body = readme.read_text(encoding="utf-8")
    block = body.split("## Quickstart", 1)[1].split("```python", 1)[1].split("```", 1)[0]
    for needed in ["from privacy_scan_ml import scan, mask", "report = scan(df)", "mask(df)"]:
        assert needed in block
