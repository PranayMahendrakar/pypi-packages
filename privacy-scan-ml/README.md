# privacy-scan-ml

Find the personal data in a dataset before it leaks into a model: emails, phones, Aadhaar,
PAN, cards, IPs, addresses and more. Every identifier is checked, not just pattern-matched,
so an amount column is not reported as a phone number and a random 12-digit id is not
reported as an Aadhaar. Nothing the report prints is an unmasked value.

## Install

```bash
pip install privacy-scan-ml
```

## Quickstart

```python
import pandas as pd
from privacy_scan_ml import scan, mask

df = pd.DataFrame({
    "full_name": ["Asha Rao", "Vikram Nair"],
    "email": ["asha@example.com", "vikram@example.org"],
    "phone": ["9876543210", "+91 91234 56780"],
    "order_amount": [1299, 45999],
})
report = scan(df)
print(report.summary())
print(report.risk, report.has_pii, report.pii_columns)
print(mask(df))
```

## What it finds

- **email** - RFC-shaped address with a real domain label and an alphabetic TLD,
  including a non-ASCII local part and an IDN domain (RFC 6531).
- **phone** - international `+CC` numbers, Indian 10-digit mobiles starting 6-9 (with or
  without separators, `0` or `+91` prefix) and separator-formatted NANP numbers.
- **aadhaar** - 12 digits, first digit 2-9, **Verhoeff checksum**.
- **pan** - Indian PAN `AAAAA9999A` with the fourth letter one of P/C/H/F/A/T/B/L/J/G.
- **credit_card** - 13-19 digits, plausible issuer prefix, **Luhn checksum**.
- **ipv4 / ipv6 / url** - parsed, not guessed.
- **date_of_birth** - a real calendar date within the last 120 years, plus a column-name
  or cue-word hint ("born", "dob").
- **postal_code** - Indian 6-digit PIN and US ZIP / ZIP+4, with a column-name hint.
- **address** - house or flat number, `221B` and `12A` included, followed by street
  words (road, nagar, marg, avenue).
- **person_name** - column-name hint plus the share of capitalised, alphabetic tokens.
- **sensitive categories** - gender, religion, caste, ethnicity, nationality, sexual
  orientation, disability, political opinion, health condition, from the column name.
  Each is reported under its own type; `SENSITIVE_TYPES` and `ColumnFinding.category`
  (`"sensitive_category"`) let you ask for the family without listing the nine names.

Guardrails that matter in practice:

- A column whose header says id, ref, serial, count or amount is never called a phone,
  whether it is stored as integers or as strings, and any other numeric column needs the
  digit lengths themselves to look like phone numbers before it is flagged.
  `order_amount`, `user_id` and a string-typed `txn_ref` all stay clean.
- `date_of_birth`, `postal_code`, `person_name` and the sensitive categories need the
  column name to agree, because a six-digit number on its own is not evidence.
- Big frames are sampled evenly (50,000 rows by default), so a scan is deterministic
  and a million-row table still takes seconds.

## API

### `scan(data, *, sample=50_000, min_share=0.2) -> PIIReport`

`data` is a DataFrame, a Series, a dict of columns, a path to `.csv` / `.tsv` /
`.parquet`, a string of free text, or a list of strings.

`PIIReport`:

| member | what it is |
| --- | --- |
| `.columns` | `dict[str, ColumnFinding]` for tabular input |
| `.findings` | `list[Finding]` for free text: `type`, `value_masked`, `span`, `doc` |
| `.has_pii` | `True` when anything was flagged |
| `.risk` | `"high"`, `"medium"`, `"low"` or `"none"` |
| `.types`, `.pii_columns` | what was found, and where |
| `.summary()` | human-readable text |
| `.to_dict()` | JSON-safe dictionary |
| `.to_markdown()` | the same report as a Markdown document |
| `.warnings` | e.g. a column named `email` whose values never validated |

`ColumnFinding` carries `types` (type -> share of non-null values), three **masked**
`samples`, a `confidence` from 0 to 1, the `primary` type, the `dtype` and `rows_checked`.

### `mask(data, *, strategy="redact", columns=None, salt=None)`

Returns the same type that went in (a path gives you back the masked DataFrame).

- `"redact"` - `asha@example.com` becomes `[EMAIL]`
- `"hash"` - `sha256(salt + value)[:12]`, stable across files with the same salt
- `"partial"` - keeps the last four characters: `******3210`

With `strategy="hash"` and no `salt`, a fresh random salt is generated for that one call and
recorded on the result, so you never reproduce a mapping by accident. Read it back and pass it
in again when you do want the same mapping twice:

```python
masked = mask(df, strategy="hash")
salt = masked.attrs["privacy_scan_ml_salt"]      # a DataFrame or Series carries it here
same = mask(df, strategy="hash", salt=salt)      # identical output
```

Masked text comes back as a `str` (or `list`) carrying the salt on `.salt` instead. On the
CLI the generated salt is printed, because a CSV cannot carry it.

Every type the scan flags in a cell is replaced, and a column that masking could not change
is reported in `.attrs["privacy_scan_ml_warnings"]` rather than passing silently.

Only the matched span inside a cell is replaced, so a free-text note keeps its wording.
Columns that nothing matched in are left untouched, dtype included.

### `PIIScanner(*, sample=50_000, min_share=0.2)`

The same settings reused across many frames: `.scan()`, `.scan_frame()`, `.scan_text()`,
`.mask()`, `.mask_frame()`.

## CLI

```bash
privacy-scan-ml customers.csv                      # print the summary
privacy-scan-ml customers.csv --json -o out.json   # JSON report
privacy-scan-ml customers.csv --mask clean.csv --strategy hash --salt s3cret
privacy-scan-ml customers.csv --mask clean.csv --strategy hash   # prints the salt it made
privacy-scan-ml --text "write to asha@example.com"
privacy-scan-ml customers.csv --fail-on-pii        # exit 1 when PII is found
```

`--help` lists every flag, including `--sample`, `--min-share` and `--columns`.

## License

MIT
