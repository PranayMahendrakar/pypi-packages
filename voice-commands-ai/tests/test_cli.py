"""The voice-commands-ai command line, including non-ASCII text through pipes."""

import json
import os
import subprocess
import sys

import pytest

from voice_commands_ai import cli

THERMOSTAT = "thermostat = set temperature to {value:number} degrees"
SWITCH = "switch = turn {state:on|off} the {device:text}"
PLAY = "play = play {song:text}"


def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_help_mentions_that_speech_recognition_is_not_included(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "does not recognise speech" in out
    assert "--number" in out and "--json" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_match_prints_the_summary(capsys):
    code, out, _ = run_cli(capsys, "-c", THERMOSTAT, "-c", SWITCH,
                           "um set the temperature to twenty one degrees please")
    assert code == 0
    assert "Matched command 'thermostat'" in out
    assert "value = 21" in out


def test_no_match_reports_the_closest_candidate(capsys):
    code, out, _ = run_cli(capsys, "-c", THERMOSTAT, "what time is it")
    assert code == 0
    assert 'No command matched "what time is it"' in out
    assert "closest: 'thermostat'" in out
    code, _, _ = run_cli(capsys, "-c", THERMOSTAT, "--fail-on-no-match", "what time is it")
    assert code == 1
    code, _, _ = run_cli(capsys, "-c", THERMOSTAT, "--fail-on-no-match", "set temperature to 5 degrees")
    assert code == 0


def test_json_output(capsys):
    code, out, _ = run_cli(capsys, "-c", SWITCH, "--json", "turn the lights off", "hello there")
    assert code == 0
    data = json.loads(out)
    assert data[0]["matched"] is True
    assert data[0]["match"]["slots"] == {"state": "off", "device": "lights"}
    assert data[1] == {"text": "hello there", "matched": False, "match": None,
                       "closest": data[1]["closest"]}


def test_all_lists_every_command(capsys):
    code, out, _ = run_cli(capsys, "-c", THERMOSTAT, "-c", SWITCH, "--all", "turn on the fan")
    assert code == 0
    assert "All commands: 'switch'" in out and "'thermostat'" in out


def test_min_confidence(capsys):
    code, out, _ = run_cli(capsys, "-c", THERMOSTAT, "--min-confidence", "0.95", "set temperature to 21")
    assert "No command matched" in out
    code, _, err = run_cli(capsys, "-c", THERMOSTAT, "--min-confidence", "3", "x")
    assert code == 2 and "min_confidence" in err


def test_number_mode(capsys):
    code, out, _ = run_cli(capsys, "--number", "a hundred and five", "two point five", "minus three", "nope")
    assert code == 0
    assert out.splitlines() == [
        "a hundred and five = 105", "two point five = 2.5", "minus three = -3", "nope = not a number",
    ]
    code, out, _ = run_cli(capsys, "--number", "--json", "twenty one")
    assert json.loads(out) == [{"text": "twenty one", "value": 21}]


def test_commands_files_and_utterance_files(tmp_path, capsys):
    txt = tmp_path / "commands.txt"
    txt.write_text(
        "# smart home\n" + THERMOSTAT + "\n\nturn {state:on|off} the {device:text}\n",
        encoding="utf-8",
    )
    js = tmp_path / "commands.json"
    js.write_text(json.dumps([
        {"pattern": "play {song:text}", "name": "play", "examples": ["play some jazz"]},
        "stop the music",
    ]), encoding="utf-8")
    heard = tmp_path / "heard.txt"
    heard.write_text("set temperature to 20 degrees\n\nplay Für Elise\nstop the music\n",
                     encoding="utf-8")
    out_path = tmp_path / "results.json"
    code, out, _ = run_cli(capsys, "-f", str(txt), "-f", str(js), str(heard),
                           "--output", str(out_path))
    assert code == 0
    assert "Results written to" in out
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert [d["match"]["name"] for d in data] == ["thermostat", "play", "stop the music"]
    assert data[1]["match"]["slots"] == {"song": "Für Elise"}
    assert "Für Elise" in out_path.read_text(encoding="utf-8")  # written unescaped


def test_json_object_commands_file(tmp_path, capsys):
    js = tmp_path / "cmds.json"
    js.write_text(json.dumps({"lock": "lock the {door:front|back} door"}), encoding="utf-8")
    code, out, _ = run_cli(capsys, "-f", str(js), "--json", "lock the back door")
    assert json.loads(out)[0]["match"]["slots"] == {"door": "back"}


def test_bad_input_exits_with_status_2(tmp_path, capsys):
    code, _, err = run_cli(capsys, "turn on the lights")
    assert code == 2 and "at least one command" in err
    code, _, err = run_cli(capsys, "-c", "set {value:numbr}", "x")
    assert code == 2 and "unknown type" in err
    bad = tmp_path / "bad.txt"
    bad.write_text("ok = turn on the lights\nbroken = set {x\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "-f", str(bad), "x")
    assert code == 2 and "line 2" in err
    code, _, err = run_cli(capsys, "-f", str(tmp_path / "missing.json"), "x")
    assert code == 2 and "could not read" in err
    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{nope", encoding="utf-8")
    code, _, err = run_cli(capsys, "-f", str(broken_json), "x")
    assert code == 2 and "not valid JSON" in err
    code, _, err = run_cli(capsys, "-c", "stop", "--output", str(tmp_path / "no" / "dir.json"), "stop")
    assert code == 2 and "could not write" in err


def _plain_env():
    env = dict(os.environ)
    for key in ("PYTHONIOENCODING", "PYTHONUTF8"):
        env.pop(key, None)
    return env


def test_non_ascii_through_a_pipe_does_not_crash():
    """Piped stdout on Windows defaults to a legacy code page; the CLI must still work."""
    text = "play Café del Mar ☕ 東京 привет"
    proc = subprocess.run(
        [sys.executable, "-m", "voice_commands_ai.cli", "-c", PLAY, text],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_plain_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8")
    assert "Café del Mar ☕ 東京 привет" in out
    proc = subprocess.run(
        [sys.executable, "-m", "voice_commands_ai.cli", "-c", PLAY, "--json"],
        input=(text + "\n").encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_plain_env(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    data = json.loads(proc.stdout.decode("utf-8"))
    assert data[0]["match"]["slots"]["song"] == "Café del Mar ☕ 東京 привет"
