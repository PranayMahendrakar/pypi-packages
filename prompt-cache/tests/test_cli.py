"""The prompt-cache command, including its behaviour under a pipe."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from prompt_cache import Cache
from prompt_cache.cli import build_parser, main

NON_ASCII_PROMPT = "重力とは何ですか \U0001f680 - que se passe-t-il ?"
NON_ASCII_ANSWER = "レイリー散乱 — diffusion"


def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_help_works(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "prompt-cache" in capsys.readouterr().out


def test_version_works(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_parser_lists_every_command():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("stats", "list", "get", "set", "clear", "prune"):
        assert command in help_text


def test_bare_invocation_prints_stats(tmp_path, capsys):
    code, out, _err = run(["--path", str(tmp_path / "c")], capsys)
    assert code == 0
    assert "prompt-cache:" in out
    assert "namespace: default" in out


def test_set_get_and_list(tmp_path, capsys):
    target = str(tmp_path / "c")
    code, out, _err = run(["--path", target, "set", "why?", "because", "--param", "model=gpt-4o"], capsys)
    assert code == 0
    assert "stored" in out

    code, out, _err = run(["--path", target, "get", "why?", "--param", "model=gpt-4o"], capsys)
    assert code == 0
    assert out.strip() == "because"

    code, out, _err = run(["--path", target, "list"], capsys)
    assert code == 0
    assert "why?" in out


def test_get_exits_one_on_a_miss(tmp_path, capsys):
    code, out, err = run(["--path", str(tmp_path / "c"), "get", "nothing here"], capsys)
    assert code == 1
    assert out == ""
    assert "nothing cached" in err


def test_params_are_read_as_json_when_they_look_like_json(tmp_path, capsys):
    target = str(tmp_path / "c")
    run(["--path", target, "set", "p", "v", "--param", "temperature=0.7"], capsys)
    cache = Cache(target)
    assert cache.get("p", temperature=0.7) == "v"
    assert cache.get("p", temperature="0.7") is None


def test_a_malformed_param_is_reported(tmp_path, capsys):
    code, _out, err = run(["--path", str(tmp_path / "c"), "get", "p", "--param", "oops"], capsys)
    assert code == 1
    assert "KEY=VALUE" in err


def test_json_output_is_machine_readable(tmp_path, capsys):
    target = str(tmp_path / "c")
    run(["--path", target, "set", "p", "v"], capsys)
    code, out, _err = run(["--path", target, "stats", "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["entries"] == 1
    assert data["namespace"] == "default"

    code, out, _err = run(["--path", target, "list", "--json"], capsys)
    rows = json.loads(out)
    assert rows[0]["preview"] == "p"


def test_output_files_are_utf8(tmp_path, capsys):
    target = str(tmp_path / "c")
    run(["--path", target, "set", NON_ASCII_PROMPT, NON_ASCII_ANSWER], capsys)
    out_file = tmp_path / "entries.json"
    code, _out, _err = run(["--path", target, "list", "--json", "--output", str(out_file)], capsys)
    assert code == 0
    written = out_file.read_text(encoding="utf-8")
    assert NON_ASCII_PROMPT.split()[0] in written
    assert json.loads(written)[0]["preview"].startswith("重力")


def test_namespace_option_is_honoured(tmp_path, capsys):
    target = str(tmp_path / "c")
    run(["--path", target, "--namespace", "alpha", "set", "p", "v"], capsys)
    code, _out, _err = run(["--path", target, "get", "p"], capsys)
    assert code == 1
    code, out, _err = run(["--path", target, "--namespace", "alpha", "get", "p"], capsys)
    assert code == 0
    assert out.strip() == "v"


def test_ttl_zero_stores_nothing(tmp_path, capsys):
    target = str(tmp_path / "c")
    code, out, _err = run(["--path", target, "--ttl", "0", "set", "p", "v"], capsys)
    assert code == 0
    assert "nothing was stored" in out


def test_clear_and_prune(tmp_path, capsys):
    target = str(tmp_path / "c")
    run(["--path", target, "set", "p", "v"], capsys)
    code, out, _err = run(["--path", target, "prune"], capsys)
    assert code == 0
    assert "removed 0 expired entries" in out
    code, out, _err = run(["--path", target, "clear"], capsys)
    assert code == 0
    assert "removed 1 entry" in out
    code, out, _err = run(["--path", target, "clear", "--all"], capsys)
    assert "the whole file" in out


def test_list_says_when_a_namespace_is_empty(tmp_path, capsys):
    code, out, _err = run(["--path", str(tmp_path / "c"), "list"], capsys)
    assert code == 0
    assert "is empty" in out


def test_get_prints_structured_values_as_json(tmp_path, capsys):
    target = str(tmp_path / "c")
    Cache(target).set("p", {"choices": [1, 2]})
    code, out, _err = run(["--path", target, "get", "p"], capsys)
    assert code == 0
    assert json.loads(out) == {"choices": [1, 2]}


def _piped(args, cwd):
    """Run the CLI in a child process with a hostile console encoding."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-m", "prompt_cache"] + args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_piped_non_ascii_output_does_not_crash(tmp_path):
    target = str(tmp_path / "c")
    cache = Cache(target)
    cache.set(NON_ASCII_PROMPT, NON_ASCII_ANSWER)
    cache.close()

    for args in (["--path", target, "list"], ["--path", target, "list", "--json"],
                 ["--path", target, "get", NON_ASCII_PROMPT], ["--path", target, "stats"]):
        done = _piped(args, tmp_path)
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in done.stderr
        assert done.stderr.decode("utf-8", "replace").strip() == ""

    done = _piped(["--path", target, "get", NON_ASCII_PROMPT], tmp_path)
    assert done.stdout.decode("utf-8").strip() == NON_ASCII_ANSWER


def test_help_is_pipe_safe():
    done = subprocess.run(
        [sys.executable, "-m", "prompt_cache", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert done.returncode == 0
    assert b"usage: prompt-cache" in done.stdout
