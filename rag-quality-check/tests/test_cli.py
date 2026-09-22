"""The command line interface, including the encoding path that pipes break."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from rag_quality_check.cli import build_parser, main, parse_cases

CASES = [
    {"query": "reset my password", "retrieved": ["p_reset", "p_billing"], "relevant": ["p_reset"]},
    {"query": "cancel my plan", "retrieved": ["p_faq", "p_billing"], "relevant": ["p_cancel"]},
]

UNICODE_CASES = [
    {
        "query": "パスワードを再設定",
        "retrieved": ["パスワードを再設定するには設定を開きます"],
        "answer": "設定を開きます。",
    },
    {"query": "café ouvert", "retrieved": ["le café est ouvert"], "relevant": ["le café est ouvert"]},
]


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_help_lists_every_metric():
    parser = build_parser()
    text = parser.format_help()
    assert "rag-quality-check" in text
    for name in ("precision_at_k", "ndcg_at_k", "groundedness", "citation_coverage"):
        assert name in text


def test_a_json_file_prints_the_summary(tmp_path, capsys):
    path = write_json(tmp_path, "cases.json", CASES)
    assert main([path, "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 case(s), top-2" in out
    assert "precision_at_k" in out
    assert "worst" in out and "in detail" in out


def test_json_flag_prints_the_dict(tmp_path, capsys):
    path = write_json(tmp_path, "cases.json", CASES)
    assert main([path, "--k", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_cases"] == 2
    assert payload["metrics"]["hit_rate"] == 0.5
    assert len(payload["per_case"]) == 2


def test_weakest_zero_drops_the_detail_block(tmp_path, capsys):
    path = write_json(tmp_path, "cases.json", CASES)
    assert main([path, "--weakest", "0"]) == 0
    assert "in detail" not in capsys.readouterr().out


def test_jsonl_input_is_one_case_per_line(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in CASES) + "\n",
        encoding="utf-8",
    )
    assert main([str(path), "--k", "2"]) == 0
    assert "2 case(s)" in capsys.readouterr().out


def test_stdin_is_read_when_the_input_is_a_dash(monkeypatch, capsys):
    payload = json.dumps(CASES, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))
    assert main(["-", "--k", "2"]) == 0
    assert "2 case(s)" in capsys.readouterr().out


def test_output_writes_json_or_text(tmp_path, capsys):
    path = write_json(tmp_path, "cases.json", CASES)
    as_json = tmp_path / "report.json"
    assert main([path, "--output", str(as_json)]) == 0
    assert json.loads(as_json.read_text(encoding="utf-8"))["n_cases"] == 2
    assert "wrote the report" in capsys.readouterr().out

    as_text = tmp_path / "report.txt"
    assert main([path, "--output", str(as_text)]) == 0
    assert "precision_at_k" in as_text.read_text(encoding="utf-8")


def test_unicode_output_is_written_unescaped(tmp_path):
    path = write_json(tmp_path, "cases.json", UNICODE_CASES)
    out = tmp_path / "report.json"
    assert main([path, "--k", "1", "--output", str(out)]) == 0
    assert "パスワード" in out.read_text(encoding="utf-8")


def test_threshold_option_changes_the_verdict(tmp_path, capsys):
    path = write_json(
        tmp_path,
        "cases.json",
        [{"query": "solar power output", "retrieved": ["solar output notes"]}],
    )
    main([path, "--k", "1", "--threshold", "0.9"])
    strict = capsys.readouterr().out
    main([path, "--k", "1", "--threshold", "0.1"])
    loose = capsys.readouterr().out
    assert "failures: none" in loose
    assert "failures (1)" in strict


@pytest.mark.parametrize(
    "payload, message",
    [
        ("", "empty"),
        ("[1, 2]", "not a JSON object"),
        ('"just a string"', "list of case objects"),
        ("{not json at all", "not valid JSON"),
    ],
)
def test_bad_input_files_fail_with_an_explained_message(tmp_path, capsys, payload, message):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "rag-quality-check: error:" in err
    assert message in err


def test_a_missing_file_is_reported_not_traced(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_a_bad_k_is_reported_not_traced(tmp_path, capsys):
    path = write_json(tmp_path, "cases.json", CASES)
    assert main([path, "--k", "0"]) == 1
    assert "at least 1" in capsys.readouterr().err


def test_parse_cases_accepts_a_single_object():
    assert parse_cases('{"query": "q", "retrieved": []}', "x") == [
        {"query": "q", "retrieved": []}
    ]


def test_parse_cases_skips_blank_jsonl_lines():
    raw = '{"query": "a", "retrieved": []}\n\n{"query": "b", "retrieved": []}\n'
    assert len(parse_cases(raw, "x")) == 2


# --------------------------------------------------------------------------
# the real subprocess: non-ASCII data through a pipe must not raise
# --------------------------------------------------------------------------

def _run(args, tmp_path):
    # cp1252 cannot encode any of the data below: without the stdout
    # reconfigure in main() this is a UnicodeEncodeError, not a report.
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    return subprocess.run(
        [sys.executable, "-m", "rag_quality_check.cli"] + args,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
    )


def test_piping_non_ascii_output_does_not_crash(tmp_path):
    path = write_json(tmp_path, "cases.json", UNICODE_CASES)
    done = _run([path, "--k", "1"], tmp_path)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    text = done.stdout.decode("utf-8", "replace")
    assert "2 case(s)" in text
    assert "UnicodeEncodeError" not in done.stderr.decode("utf-8", "replace")


def test_piping_non_ascii_json_does_not_crash(tmp_path):
    path = write_json(tmp_path, "cases.json", UNICODE_CASES)
    done = _run([path, "--k", "1", "--json"], tmp_path)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    payload = json.loads(done.stdout.decode("utf-8", "replace"))
    assert payload["n_cases"] == 2
