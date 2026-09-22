"""The command line interface.

The CLI is public surface, so it is tested like the library: every command,
both output shapes, the file and stdin paths, the error paths, and the
encoding behaviour that Windows consoles punish.
"""

import io
import json
import sys

import pytest

from llm_router_lite import __version__
from llm_router_lite.cli import (
    EXAMPLE_MODELS,
    build_parser,
    load_model_file,
    main,
    parse_model_spec,
    read_prompt,
    with_default_command,
)

HARD = "Explain step by step why this scales, compare it with a queue, and then recommend one."

TWO_MODELS = [
    "--model",
    "small:0.0002:0.4:100:local",
    "--model",
    "large:0.01:0.95:900:cloud",
]


def run(capsys, argv):
    """Run main() and hand back (exit code, stdout, stderr)."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ----------------------------------------------------------------- plumbing


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "llm-router-lite" in capsys.readouterr().out


def test_version_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_with_default_command_makes_complexity_the_default():
    assert with_default_command([]) == ["complexity"]
    assert with_default_command(["hello"]) == ["complexity", "hello"]
    assert with_default_command(["route", "hi"]) == ["route", "hi"]
    assert with_default_command(["--help"]) == ["--help"]


def test_build_parser_is_usable_on_its_own():
    args = build_parser().parse_args(["route", "hi", "--prefer", "cheap"])
    assert args.command == "route"
    assert args.prefer == "cheap"


# ----------------------------------------------------------------- commands


def test_bare_prompt_scores_complexity(capsys):
    code, out, _ = run(capsys, ["What is 2 + 2?"])
    assert code == 0
    assert "simple prompt" in out
    assert "signals:" in out


def test_complexity_json(capsys):
    code, out, _ = run(capsys, ["complexity", HARD, "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["band"] == "hard"
    assert set(payload["signals"]) == {
        "length",
        "questions",
        "reasoning",
        "code",
        "instructions",
    }


def test_route_picks_and_explains(capsys):
    code, out, _ = run(capsys, ["route", HARD] + TWO_MODELS)
    assert code == 0
    assert "route -> large" in out
    assert "considered:" in out
    assert "no model was called" in out


def test_route_json_carries_the_decision(capsys):
    code, out, _ = run(capsys, ["route", HARD, "--json"] + TWO_MODELS)
    assert code == 0
    payload = json.loads(out)
    assert payload["model"] == "large"
    assert payload["alternatives"] == ["small"]
    assert "small" in payload["demoted"]


def test_route_honours_prefer_cheap(capsys):
    code, out, _ = run(
        capsys, ["route", HARD, "--prefer", "cheap", "--json"] + TWO_MODELS
    )
    assert code == 0
    assert json.loads(out)["model"] == "small"


def test_route_honours_require_and_max_cost(capsys):
    code, out, _ = run(
        capsys, ["route", HARD, "--require", "local", "--json"] + TWO_MODELS
    )
    assert code == 0
    assert json.loads(out)["model"] == "small"

    code, out, _ = run(
        capsys, ["route", HARD, "--max-cost", "0.001", "--json"] + TWO_MODELS
    )
    assert code == 0
    assert json.loads(out)["model"] == "small"


def test_models_lists_the_line_up(capsys):
    code, out, _ = run(capsys, ["models"])
    assert code == 0
    assert "3 model(s):" in out
    for example in EXAMPLE_MODELS:
        assert example["name"] in out


def test_models_json(capsys):
    code, out, _ = run(capsys, ["models", "--json"])
    assert code == 0
    assert [entry["name"] for entry in json.loads(out)] == [
        example["name"] for example in EXAMPLE_MODELS
    ]


def test_the_example_line_up_stands_in_when_none_is_given(capsys):
    code, out, _ = run(capsys, ["route", HARD])
    assert code == 0
    assert "cloud-large" in out


# -------------------------------------------------------------------- specs


def _without_handler(options):
    """Specs carry a placeholder handler, since the CLI never calls a model."""
    rest = dict(options)
    assert callable(rest.pop("handler"))
    return rest


def test_parse_model_spec_fills_in_from_the_left():
    assert _without_handler(parse_model_spec("bare")) == {"name": "bare"}
    full = parse_model_spec("big:0.01:0.95:900:cloud,fast")
    assert full["name"] == "big"
    assert full["cost"] == 0.01
    assert full["quality"] == 0.95
    assert full["latency_ms"] == 900.0
    assert full["tags"] == ["cloud", "fast"]


def test_a_bad_number_in_a_spec_is_a_clear_error(capsys):
    code, _, err = run(capsys, ["route", "hi", "--model", "big:not-a-number"])
    assert code == 1
    assert "cost" in err
    assert "big:not-a-number" in err


def test_an_empty_spec_name_is_rejected(capsys):
    code, _, err = run(capsys, ["route", "hi", "--model", ":0.01"])
    assert code == 1
    assert err.startswith("error:")


# --------------------------------------------------------------- model file


def test_load_model_file_accepts_a_list(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps([{"name": "a", "cost": 0.1}]), encoding="utf-8")
    loaded = load_model_file(str(path))
    assert [_without_handler(entry) for entry in loaded] == [{"name": "a", "cost": 0.1}]


def test_load_model_file_accepts_an_object_of_name_to_options(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(json.dumps({"a": {"cost": 0.1}}), encoding="utf-8")
    loaded = load_model_file(str(path))
    assert [_without_handler(entry) for entry in loaded] == [{"cost": 0.1, "name": "a"}]


def test_load_model_file_rejects_bad_json(tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_model_file(str(path))
    assert "not valid JSON" in str(excinfo.value)


def test_a_missing_model_file_is_a_clean_error_not_a_traceback(capsys):
    code, _, err = run(capsys, ["route", "hi", "--models", "no-such-file.json"])
    assert code == 1
    assert err.startswith("error:")


def test_models_file_drives_the_route(capsys, tmp_path):
    path = tmp_path / "lineup.json"
    path.write_text(
        json.dumps(
            [
                {"name": "cheap", "cost": 0.0},
                {"name": "good", "cost": 1.0, "quality": 0.9},
            ]
        ),
        encoding="utf-8",
    )
    code, out, _ = run(capsys, ["route", HARD, "--models", str(path), "--json"])
    assert code == 0
    assert json.loads(out)["model"] == "good"


# -------------------------------------------------------------------- stdin


def test_read_prompt_prefers_the_argument():
    assert read_prompt("hello") == "hello"


def test_stdin_is_decoded_as_utf8_whatever_the_console_codepage(monkeypatch):
    """A piped UTF-8 prompt must survive, not arrive as mojibake."""
    text = "なぜ空は青いのですか \U0001f30f café"

    class _Cp1252Stdin:
        """stdin as Windows hands it over: bytes, and a text view that is wrong."""

        def __init__(self, raw):
            self.buffer = io.BytesIO(raw)

        def read(self):  # pragma: no cover - the raw buffer path wins
            raise AssertionError("the raw buffer should be preferred")

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _Cp1252Stdin(text.encode("utf-8")))
    assert read_prompt("-") == text


def test_stdin_without_a_buffer_still_works(monkeypatch):
    stream = io.StringIO("plain text\n")
    stream.isatty = lambda: False
    monkeypatch.setattr(sys, "stdin", stream)
    assert read_prompt("-") == "plain text"


def test_stdin_that_is_not_utf8_scores_instead_of_crashing(monkeypatch):
    class _Broken:
        buffer = io.BytesIO(b"caf\xe9 and more")

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _Broken())
    assert "caf" in read_prompt("-")


def test_no_prompt_and_a_tty_is_a_clear_error(monkeypatch):
    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", _Tty())
    with pytest.raises(ValueError) as excinfo:
        read_prompt(None)
    assert "no prompt given" in str(excinfo.value)


# ----------------------------------------------------------------- encoding


def test_non_ascii_model_names_survive_to_json(capsys):
    small = "モデル小:0.0:0.3:80:local"
    big = "modèle-café☕:0.002:0.7:300:cloud"
    code, out, _ = run(
        capsys, ["route", HARD, "--json", "--model", small, "--model", big]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["model"] == "modèle-café☕"
    assert "モデル小" in payload["demoted"]


def test_output_file_is_written_as_utf8(capsys, tmp_path):
    target = tmp_path / "route.json"
    big = "modèle-café☕:0.002:0.7:300:cloud"
    code, _, _ = run(
        capsys,
        ["route", HARD, "--json", "--output", str(target), "--model", big],
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model"] == "modèle-café☕"


def test_summary_output_is_plain_ascii_punctuation(capsys):
    code, out, _ = run(capsys, ["route", HARD] + TWO_MODELS)
    assert code == 0
    for bad in ("→", "•", "─", "—"):
        assert bad not in out


# ------------------------------------------------------------------- errors


def test_an_unknown_prefer_is_a_clean_error():
    with pytest.raises(SystemExit):
        main(["route", "hi", "--prefer", "nonsense"])


def test_a_max_cost_nothing_meets_names_the_cheapest(capsys):
    code, _, err = run(capsys, ["route", HARD, "--max-cost", "0.0000001"] + TWO_MODELS)
    assert code == 1
    assert "small" in err


def test_a_require_that_matches_nothing_is_a_clean_error(capsys):
    code, _, err = run(capsys, ["route", HARD, "--require", "vision"] + TWO_MODELS)
    assert code == 1
    assert err.startswith("error:")


# ------------------------------------------------------- reviewer follow-ups


def test_route_text_output_shows_the_demoted_block_not_only_skipped(capsys):
    # The README draws a line between skipped (out of the running) and demoted
    # (ranks behind, still a fallback). The text view used to show only the
    # first, while --json carried both.
    code, out, _ = run(capsys, ["route", HARD] + TWO_MODELS)
    assert code == 0
    assert "ranked behind, still a fallback:" in out
    assert "small" in out

    code, json_out, _ = run(capsys, ["route", HARD, "--json"] + TWO_MODELS)
    payload = json.loads(json_out)
    assert set(payload["demoted"]) == {"small"}
    for name in payload["demoted"]:
        assert name in out


def test_route_text_output_still_shows_the_skipped_block(capsys):
    code, out, _ = run(
        capsys, ["route", HARD, "--require", "cloud"] + TWO_MODELS
    )
    assert code == 0
    assert "set aside, out of the running:" in out
    assert "small" in out


def test_a_route_with_nothing_demoted_prints_no_demoted_block(capsys):
    code, out, _ = run(capsys, ["route", "hi"] + TWO_MODELS)
    assert code == 0
    assert "ranked behind" not in out


def test_a_command_word_stays_a_command_and_the_help_says_how_to_score_it(capsys):
    # `llm-router-lite models` must keep listing the line-up, so the three
    # command words are the one set of prompts the bare shorthand cannot
    # express. That is documented rather than silently surprising.
    for word in ("complexity", "route", "models"):
        assert with_default_command([word]) == [word]
    code, out, _ = run(capsys, ["models"])
    assert code == 0
    assert "model(s):" in out

    with pytest.raises(SystemExit):
        main(["--help"])
    helped = capsys.readouterr().out
    assert "complexity \"models\"" in helped

    code, out, _ = run(capsys, ["complexity", "models"])
    assert code == 0
    assert "simple prompt" in out


def test_complexity_text_output_carries_a_scorer_warning(capsys):
    thai = "อธิบายทีละขั้นตอนว่าทำไมสถาปัตยกรรมนี้จึงขยายตัวได้และเปรียบเทียบกับคิว"
    code, out, _ = run(capsys, ["complexity", thai, "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["warnings"]
    assert "script" in payload["warnings"][0]
