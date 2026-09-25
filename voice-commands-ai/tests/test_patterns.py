"""Pattern syntax: what is accepted, and clear errors for what is not."""

import pytest

from voice_commands_ai import Commands
from voice_commands_ai._pattern import parse_pattern


def test_slot_types_are_parsed():
    p = parse_pattern("turn {state:on|off} the {device} to {level:number} for {note:text}")
    assert p.slots == (("state", "choice"), ("device", "word"), ("level", "number"), ("note", "text"))
    kinds = [e.kind for e in p.elements]
    assert kinds == ["literal", "choice", "literal", "word", "literal", "number", "literal", "text"]


def test_optional_words_and_multi_word_choices():
    p = parse_pattern("open [the] {room:living room|kitchen} door")
    assert [e.kind for e in p.elements] == ["literal", "optional", "choice", "literal"]
    choice = p.elements[2]
    assert choice.options[0] == ("living room", ("living", "room"))


def test_punctuation_and_case_in_patterns_are_ignored():
    cmds = Commands().add("What's the WEATHER, in {city:text}?")
    assert cmds.match("whats the weather in oslo").slots == {"city": "oslo"}


@pytest.mark.parametrize(
    "pattern, message",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("set {value:numbr} degrees", "unknown type"),
        ("set {:number} degrees", "no name"),
        ("set {2value:number}", "identifier"),
        ("set {class}", "identifier"),
        ("turn {state:on|} light", "empty choice"),
        ("turn {state:on|on} light", "twice"),
        ("say {a:text} {b:text}", "two text slots"),
        ("move {x} and {x}", "more than once"),
        ("set {value:number degrees", "unmatched"),
        ("set value} degrees", "unmatched"),
        ("turn on [the lights", "unmatched"),
        ("turn on []", "empty optional"),
        ("[only optional words]", "not optional"),
    ],
)
def test_invalid_patterns_raise_value_error(pattern, message):
    with pytest.raises(ValueError, match=message):
        Commands().add(pattern)


def test_non_string_pattern_is_a_type_error():
    with pytest.raises(TypeError):
        Commands().add(42)  # type: ignore[arg-type]


def test_unicode_slot_names_are_allowed():
    cmds = Commands().add("allume la {pièce}", lambda pièce: pièce.upper())
    assert cmds.run("allume la cuisine") == "CUISINE"


def test_specificity_orders_fixed_words_over_open_slots():
    fixed = parse_pattern("turn on the kitchen light").specificity
    open_slot = parse_pattern("turn on the {device:text}").specificity
    assert fixed > open_slot
