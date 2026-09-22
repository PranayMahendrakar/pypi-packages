"""The auto-label command line."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from auto_label.cli import build_parser, main


@pytest.fixture
def tickets(tmp_path):
    df = pd.DataFrame(
        {
            "text": ["Invoice overdue, refund please", "App crash on login", "Refund not received",
                     "Error 500 on checkout", "Another crash report", "Invoice for March",
                     "Crash when saving", "Where is my refund"],
            "amount": [10, 0, 25, 0, 0, 40, 0, 15],
        }
    )
    path = tmp_path / "tickets.csv"
    df.to_csv(path, index=False)
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: auto-label" in out
    assert "--rule" in out and "--query" in out and "--output" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_csv_column_rules_json(tickets, capsys):
    code = main([str(tickets), "--column", "text", "--rule", "billing=invoice,refund", "--rule", "bug=crash,error", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_items"] == 8
    assert payload["mode"] == "text"
    assert payload["labels"] == {"billing": 4, "bug": 4}
    assert payload["sources"]["rule"] == 8


def test_summary_and_output_csv(tickets, tmp_path, capsys):
    out = tmp_path / "labeled.csv"
    code = main([str(tickets), "--column", "text", "--rule", "billing=refund", "--regex", "bug=(?i)crash|error",
                 "--no-model", "--output", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "auto-label:" in printed and "wrote" in printed
    written = pd.read_csv(out)
    assert list(written.columns) == ["index", "item", "label", "confidence", "source"]
    assert len(written) == 8
    assert written["label"].tolist()[:2] == ["billing", "bug"]


def test_tabular_query_and_json_output(tickets, tmp_path):
    out = tmp_path / "result.json"
    code = main([str(tickets), "--query", "paid=amount > 0", "--query", "free=amount == 0", "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "tabular"
    assert payload["labels"] == {"paid": 4, "free": 4}
    assert payload["records"][0]["item"]["amount"] == 10


def test_text_file_lines_and_rules_file(tmp_path, capsys):
    lines = tmp_path / "lines.txt"
    lines.write_text("Free prize now\n\nTeam meeting\nCafé au lait\n", encoding="utf-8")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"spam": ["free"], "work": {"regex": "(?i)meeting", "weight": 2}, "coffee": ["café"]}),
                     encoding="utf-8")
    code = main([str(lines), "--rules", str(rules), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["label"] for r in payload["records"]] == ["spam", "work", "coffee"]


def test_stdin_input(monkeypatch, capsys):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("Free prize\nTeam meeting\n"))
    assert main(["-", "--rule", "spam=free", "--rule", "work=meeting", "--labels", "spam,work", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"] == ["spam", "work"]
    assert payload["n_labeled"] == 2


def test_errors_are_reported_with_exit_code_2(tickets, tmp_path, capsys):
    assert main([str(tickets), "--rule", "novalue"]) == 2
    assert "LABEL=VALUE" in capsys.readouterr().err

    assert main([str(tmp_path / "missing.csv"), "--rule", "a=b"]) == 2
    assert "no such file" in capsys.readouterr().err.lower()

    assert main([str(tickets), "--column", "nope", "--rule", "a=b"]) == 2
    assert "nope" in capsys.readouterr().err

    assert main([str(tickets), "--query", "big=nope > 1"]) == 2
    assert "query" in capsys.readouterr().err

    bad_rules = tmp_path / "bad.json"
    bad_rules.write_text("[1, 2]", encoding="utf-8")
    assert main([str(tickets), "--rules", str(bad_rules)]) == 2
    assert "JSON object" in capsys.readouterr().err


def test_no_rules_is_a_usage_error(tickets, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(tickets)])
    assert exc.value.code == 2
    assert "at least one rule" in capsys.readouterr().err


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["x.csv", "--rule", "a=b", "--min-confidence", "0.3"])
    assert args.min_confidence == 0.3


def test_module_runs_as_script():
    proc = subprocess.run([sys.executable, "-m", "auto_label.cli", "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "usage: auto-label" in proc.stdout


def test_package_runs_as_module():
    proc = subprocess.run([sys.executable, "-m", "auto_label", "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "usage: auto-label" in proc.stdout
