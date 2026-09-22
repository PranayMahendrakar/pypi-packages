"""The command line: every flag, and the Windows pipe-encoding trap."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from privacy_scan_ml.cli import main

# Non-ASCII on purpose: this is what crashed sibling packages when piped.
UNICODE_ROWS = {
    "full_name": ["अमित शर्मा", "李雷", "Zoë Müller"],
    "email": ["amit@example.com", "lilei@example.org", "zoe@example.net"],
    "city": ["पुणे", "北京", "Köln"],
}


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "people.csv"
    pd.DataFrame(UNICODE_ROWS).to_csv(path, index=False, encoding="utf-8")
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "privacy-scan-ml" in capsys.readouterr().out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_scanning_a_file_prints_the_summary(csv_path, capsys):
    assert main([str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "privacy-scan-ml:" in out and "email" in out
    assert "amit@example.com" not in out


def test_json_output_is_valid_and_masked(csv_path, capsys):
    assert main([str(csv_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["has_pii"] is True and payload["risk"] in ("low", "medium", "high")
    assert "email" in payload["columns"]
    assert "amit@example.com" not in json.dumps(payload, ensure_ascii=False)


def test_output_flag_writes_the_report(csv_path, tmp_path, capsys):
    out_path = tmp_path / "report.json"
    assert main([str(csv_path), "-o", str(out_path)]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["has_pii"] is True
    assert "report written to" in capsys.readouterr().out


def test_text_mode(capsys):
    assert main(["--text", "write to asha@example.com"]) == 0
    out = capsys.readouterr().out
    assert "email" in out and "asha@example.com" not in out


def test_mask_writes_a_masked_copy(csv_path, tmp_path, capsys):
    clean = tmp_path / "clean.csv"
    assert main([str(csv_path), "--mask", str(clean)]) == 0
    body = clean.read_text(encoding="utf-8")
    assert "[EMAIL]" in body and "amit@example.com" not in body
    assert "पुणे" in body, "a column with nothing personal in it must survive untouched"


def test_mask_with_hash_and_an_explicit_salt(csv_path, tmp_path):
    clean = tmp_path / "clean.csv"
    assert main([str(csv_path), "--mask", str(clean), "--strategy", "hash", "--salt", "s3cret"]) == 0
    assert "amit@example.com" not in clean.read_text(encoding="utf-8")


def test_fail_on_pii_sets_the_exit_code(csv_path, capsys):
    assert main([str(csv_path), "--fail-on-pii"]) == 1
    capsys.readouterr()


def test_fail_on_pii_stays_zero_on_a_clean_file(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    pd.DataFrame({"sku": ["A-1"], "units": [3]}).to_csv(path, index=False, encoding="utf-8")
    assert main([str(path), "--fail-on-pii"]) == 0
    capsys.readouterr()


def test_giving_both_a_path_and_text_is_an_error(csv_path):
    with pytest.raises(SystemExit) as exc:
        main([str(csv_path), "--text", "hello"])
    assert exc.value.code == 2


def test_giving_neither_is_an_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_a_missing_file_reports_cleanly_instead_of_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 2
    assert "privacy-scan-ml:" in capsys.readouterr().err


# --------------------------------------------------------------------------- encoding


def _run(args, extra_env=None):
    """Run the CLI as a real subprocess with its stdout on a pipe."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "privacy_scan_ml", *args],
        capture_output=True,
        env=env,
    )


def test_piped_output_with_non_ascii_data_does_not_crash(csv_path):
    """A pipe, not a console: the classic UnicodeEncodeError on Windows."""
    done = _run([str(csv_path)])
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    done.stdout.decode("utf-8")  # must be valid UTF-8


def test_piped_output_survives_an_ascii_only_stdio_encoding(csv_path):
    """Even when the environment insists stdout is ASCII, the CLI must not blow up."""
    done = _run([str(csv_path), "--json"], {"PYTHONIOENCODING": "ascii"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr


def test_piped_text_mode_with_non_ascii_input():
    done = _run(["--text", "अमित को asha@example.com पर लिखें"])
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr


def test_python_m_entry_point_help():
    done = _run(["--help"])
    assert done.returncode == 0
    assert b"privacy-scan-ml" in done.stdout
