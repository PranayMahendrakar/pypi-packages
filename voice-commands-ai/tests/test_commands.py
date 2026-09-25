"""Commands.add / run / listen, handler errors, and the Match result object."""

import json

import pytest

from voice_commands_ai import Command, CommandError, Commands, Match


def make() -> Commands:
    calls = []

    def thermostat(value):
        calls.append(value)
        return "heating to %s" % value

    cmds = Commands()
    cmds.add("set temperature to {value:number} degrees", thermostat)
    cmds.add("turn {state:on|off} the {device:text}", name="switch")
    cmds.calls = calls  # type: ignore[attr-defined]
    return cmds


# ------------------------------------------------------------ add


def test_add_returns_self_for_chaining_and_names_default_sensibly():
    cmds = Commands().add("stop the music").add("volume {d:up|down}", lambda d: d)
    names = [c.name for c in cmds]
    assert names == ["stop the music", "volume {d:up|down}"]  # lambdas have no usable name
    assert len(cmds) == 2
    assert "stop the music" in cmds
    assert all(isinstance(c, Command) for c in cmds)

    def lights(state):
        return state

    named = Commands().add("lights {state:on|off}", lights)
    assert "lights" in named
    assert "2 commands" in repr(cmds) and "1 command," in repr(named)


def test_duplicate_names_are_refused():
    cmds = Commands().add("stop the music", name="stop")
    with pytest.raises(ValueError, match="already registered"):
        cmds.add("stop playing", name="stop")


def test_handler_that_cannot_take_the_slots_is_refused_at_add_time():
    with pytest.raises(ValueError, match="cannot be called with the pattern's slots"):
        Commands().add("set temperature to {value:number} degrees", lambda temperature: None)
    with pytest.raises(ValueError, match="value"):
        Commands().add("set temperature to {value:number} degrees", lambda: None)
    # **kwargs and defaults are fine
    Commands().add("set temperature to {value:number} degrees", lambda **slots: slots)
    Commands().add("set {value:number}", lambda value, unit="C": (value, unit))


def test_handler_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        Commands().add("stop the music", "not a function")  # type: ignore[arg-type]


def test_a_handler_with_no_inspectable_signature_is_accepted():
    """Some C callables cannot be introspected at all (inspect.signature raises), and
    registration must not fail just because it cannot check them. print() used to be
    such a callable on Python 3.10, but Python 3.11 made many builtins introspectable,
    and print's real signature correctly rejects a 'text' keyword - so this test now
    uses a genuinely signature-less object instead of relying on that accident."""
    import functools

    opaque = functools.reduce  # a builtin whose signature cannot be introspected
    cmds = Commands().add("say {text:text}", opaque, name="say")
    assert "say" in cmds


def test_a_handler_whose_real_signature_rejects_the_slots_is_caught_at_registration():
    """print() genuinely cannot be called with a 'text' keyword, and on interpreters
    that can introspect it, that mismatch should be caught now rather than at match
    time."""
    import pytest

    try:
        import inspect

        inspect.signature(print)
    except (TypeError, ValueError):
        pytest.skip("this interpreter cannot introspect print() at all")
    with pytest.raises(ValueError, match="cannot be called with the pattern's slots"):
        Commands().add("say {text:text}", print, name="say")


def test_examples_are_checked_against_their_own_pattern():
    cmds = Commands().add(
        "set temperature to {value:number} degrees",
        name="thermostat",
        examples=["set the temperature to twenty one degrees", "set temperature to 19 degrees"],
    )
    assert next(iter(cmds)).examples == (
        "set the temperature to twenty one degrees", "set temperature to 19 degrees",
    )
    with pytest.raises(ValueError, match="does not match its own pattern"):
        Commands().add("set temprature to {value:number}", examples=["play some jazz"])


def test_min_confidence_is_validated():
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Commands(min_confidence=bad)
    with pytest.raises(TypeError):
        Commands(min_confidence="high")  # type: ignore[arg-type]
    assert Commands(min_confidence=1).min_confidence == 1.0


def test_pattern_with_no_fixed_words_warns(caplog):
    with caplog.at_level("WARNING", logger="voice_commands_ai._commands"):
        Commands().add("{anything:text}")
    assert "no fixed words" in caplog.text


# ------------------------------------------------------------ run


def test_run_calls_the_handler_with_typed_slots():
    cmds = make()
    assert cmds.run("um set the temperature to twenty one degrees please") == "heating to 21"
    assert cmds.calls == [21]


def test_run_returns_default_when_nothing_matches_and_calls_nothing():
    cmds = make()
    assert cmds.run("what time is it") is None
    assert cmds.run("what time is it", default="sorry?") == "sorry?"
    assert cmds.run("") is None
    assert cmds.calls == []


def test_run_without_a_handler_returns_the_match():
    found = make().run("turn off the lights")
    assert isinstance(found, Match)
    assert found.name == "switch" and found.slots == {"state": "off", "device": "lights"}


def test_handler_that_raises_is_reported_with_the_command_name():
    def fragile(value):
        raise ValueError("%s degrees is too hot" % value)

    cmds = Commands().add("set temperature to {value:number} degrees", fragile, name="thermostat")
    with pytest.raises(CommandError) as info:
        cmds.run("set temperature to ninety degrees")
    err = info.value
    assert err.name == "thermostat"
    assert "thermostat" in str(err) and "too hot" in str(err)
    assert isinstance(err.original, ValueError)
    assert err.__cause__ is err.original
    assert err.match.slots == {"value": 90}
    assert isinstance(err, RuntimeError)


def test_handler_that_works_raises_nothing():
    cmds = Commands().add("set temperature to {value:number} degrees", lambda value: value * 2)
    assert cmds.run("set temperature to ninety degrees") == 180


def test_run_accepts_a_match():
    cmds = make()
    found = cmds.match("set temperature to 18 degrees")
    assert cmds.run(found) == "heating to 18"
    assert cmds.run(None, default="nothing") == "nothing"
    stranger = Commands().add("stop the music").match("stop the music")
    with pytest.raises(ValueError, match="does not belong"):
        cmds.run(stranger)


# ------------------------------------------------------------ listen


def test_listen_uses_the_supplied_transcriber():
    cmds = make()
    seen = []

    def transcribe(audio):
        seen.append(audio)
        return "please turn on the porch light"

    audio = b"\x00\x01" * 800  # stand-in for recorded audio
    found = cmds.listen(transcribe, audio)
    assert seen == [audio]
    assert found.name == "switch" and found.slots == {"state": "on", "device": "porch light"}


def test_listen_then_run():
    cmds = make()
    found = cmds.listen(lambda audio: "set temperature to twenty degrees", object())
    assert cmds.run(found) == "heating to 20"


def test_listen_with_alternative_transcripts_picks_the_best_match():
    cmds = make()
    nbest = ["set temper chair to twenty degrees", "set temperature to twenty degrees", "blah"]
    found = cmds.listen(lambda audio: nbest, None)
    assert found.text == "set temperature to twenty degrees"
    assert found.slots == {"value": 20}
    # ties go to the earlier (more likely) transcript
    tie = cmds.listen(lambda audio: ["turn on the fan", "turn on the van"], None)
    assert tie.slots["device"] == "fan"
    assert cmds.listen(lambda audio: ["what time is it", "hello"], None) is None
    assert cmds.listen(lambda audio: [], None) is None


def test_listen_with_silence_returns_none():
    cmds = make()
    assert cmds.listen(lambda audio: None, b"") is None
    assert cmds.listen(lambda audio: "", b"") is None


def test_listen_rejects_bad_transcribers():
    cmds = make()
    with pytest.raises(TypeError, match="callable"):
        cmds.listen("not callable", b"")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return a str"):
        cmds.listen(lambda audio: 42, b"")
    with pytest.raises(TypeError, match="expected strings"):
        cmds.listen(lambda audio: ["ok", 3], b"")


def test_listen_lets_transcriber_errors_through():
    def broken(audio):
        raise OSError("microphone unplugged")

    with pytest.raises(OSError, match="microphone unplugged"):
        make().listen(broken, b"")


# ------------------------------------------------------------ the result object


def test_match_summary_explains_itself():
    found = make().match("um could you set the tempreture to twenty one degrees please")
    text = found.summary()
    assert text.startswith("Matched command 'thermostat' with confidence")
    for piece in ("value = 21", 'from "twenty one"', "tempreture", "temperature",
                  "Ignored filler: um, could you, please", "floor 0.60"):
        assert piece in text
    assert str(found) == text
    assert text.isascii()  # plain ASCII when the input is ASCII
    assert repr(found).startswith("Match(name='thermostat'")


def test_match_to_dict_is_json_safe_and_complete():
    cmds = Commands()
    cmds.add("play {song:text}", name="play")
    cmds.add("play {song:text} by {artist:text}", name="play_by")
    found = cmds.match("play Déjà Vu by Beyoncé please")
    data = found.to_dict()
    text = json.dumps(data, ensure_ascii=False)
    assert "Beyoncé" in text
    assert data["name"] == "play_by"
    assert data["slots"] == {"song": "Déjà Vu", "artist": "Beyoncé"}
    assert data["alternatives"][0]["name"] == "play"
    assert "alternatives" not in data["alternatives"][0]
    for key in ("confidence", "pattern", "text", "slot_words", "corrections", "ignored",
                "extra", "missing", "reordered", "replaced", "coverage", "precision",
                "specificity", "threshold", "chosen_because"):
        assert key in data
    assert json.loads(text) == data


def test_match_is_truthy_and_exposes_its_command():
    found = make().match("turn off the lights")
    assert bool(found) is True
    assert found.command.name == "switch"
    assert found.command.pattern == "turn {state:on|off} the {device:text}"
