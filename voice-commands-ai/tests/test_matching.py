"""The fuzzy matcher against realistic messy transcripts.

Every behaviour is tested both ways: plant the thing (a filler, a misheard word,
a reordered word) and check it is found and reported, then give clean input and
check nothing is reported that was not there.
"""

import time

import pytest

from voice_commands_ai import Commands, Match
from voice_commands_ai._text import fold


def assistant() -> Commands:
    """A realistic smart-home / assistant command set."""
    c = Commands()
    c.add("set temperature to {value:number} degrees", name="thermostat")
    c.add("turn {state:on|off} the {device:text}", name="switch")
    c.add("play {song:text}", name="play")
    c.add("play {song:text} by {artist:text}", name="play_by")
    c.add("set a timer for {minutes:number} minutes", name="timer")
    c.add("what's the weather in {city:text}", name="weather")
    c.add("call {contact:text}", name="call")
    c.add("volume {direction:up|down}", name="volume")
    c.add("set volume to {level:number}", name="set_volume")
    c.add("remind me to {task:text} at {time:text}", name="remind")
    c.add("add {item:text} to my shopping list", name="shopping")
    c.add("stop the music", name="stop")
    c.add("lock the {door:front|back} door", name="lock")
    return c


CMDS = assistant()

MESSY = [
    # filler words and hesitations
    ("um could you set the temperature to twenty one degrees please", "thermostat", {"value": 21}),
    ("uh set temperature to minus three degrees", "thermostat", {"value": -3}),
    ("please set the temperature to a hundred and five degrees", "thermostat", {"value": 105}),
    ("hey could you turn off the kitchen lights", "switch", {"state": "off", "device": "kitchen lights"}),
    ("um set a timer for twenty five minutes please", "timer", {"minutes": 25}),
    ("could you set a timer for fifteen minutes", "timer", {"minutes": 15}),
    ("please call doctor smith", "call", {"contact": "doctor smith"}),
    ("remind me to buy um milk at noon", "remind", {"task": "buy milk", "time": "noon"}),
    ("could you add bread and butter to my shoping list please", "shopping", {"item": "bread and butter"}),
    ("can you play just like heaven", "play", {"song": "just like heaven"}),
    ("stop the music please", "stop", {}),
    # one misheard word
    ("set the tempreture to 19 degrees", "thermostat", {"value": 19}),
    ("sat temperature to eighteen degrees", "thermostat", {"value": 18}),
    ("turn of the lights", "switch", {"state": "off", "device": "lights"}),
    ("set a time for ten minutes", "timer", {"minutes": 10}),
    ("what's the whether in london", "weather", {"city": "london"}),
    ("um lock the frunt door", "lock", {"door": "front"}),
    # spoken numbers
    ("set temperature to twenty two point five degrees", "thermostat", {"value": 22.5}),
    ("set the volume to seven", "set_volume", {"level": 7}),
    ("set volume to 11", "set_volume", {"level": 11}),
    ("set timer for 3 minutes", "timer", {"minutes": 3}),
    # word order
    ("turn the bedroom lamp on please", "switch", {"state": "on", "device": "bedroom lamp"}),
    ("turn the volume down", "volume", {"direction": "down"}),
    # dropped or extra words
    ("set temperature to 21", "thermostat", {"value": 21}),
    ("set the temperature to twenty one degrees because it's cold in here", "thermostat", {"value": 21}),
    ("add almond milk to the shopping list", "shopping", {"item": "almond milk"}),
    ("turn on the garage light for me", "switch", {"state": "on", "device": "garage light"}),
    # case
    ("Turn ON the TV", "switch", {"state": "on", "device": "TV"}),
    ("whats the weather in New York", "weather", {"city": "New York"}),
    # plain
    ("set a timer for five minutes", "timer", {"minutes": 5}),
    ("what's the weather in paris", "weather", {"city": "paris"}),
    ("call mom", "call", {"contact": "mom"}),
    ("volume up", "volume", {"direction": "up"}),
    ("remind me to call mom at five pm", "remind", {"task": "call mom", "time": "five pm"}),
    ("add eggs to my shopping list", "shopping", {"item": "eggs"}),
    ("play bohemian rhapsody", "play", {"song": "bohemian rhapsody"}),
    ("play hello by adele", "play_by", {"song": "hello", "artist": "adele"}),
    ("play despacito please", "play", {"song": "despacito"}),
    ("stop the music", "stop", {}),
    ("lock the front door", "lock", {"door": "front"}),
    ("lock the back door please", "lock", {"door": "back"}),
]

UNRELATED = [
    "what time is it", "tell me a joke", "how tall is mount everest", "pay the bills",
    "i love you", "good morning", "the weather is nice today", "turn around", "play", "set",
    "lights", "call", "set the alarm for seven", "what's the temperature", "turn up the heat",
    "set the temperature", "open the window", "how are you doing today", "thank you",
    "cancel that", "what is the capital of france", "i need to go to the store",
    "can you hear me", "lock", "send a message to john", "increase the brightness",
    "the front door is open", "stop", "what's up", "order a pizza", "set an alarm for seven am",
    "turn it up", "show me the news",
]


@pytest.mark.parametrize("text, name, slots", MESSY)
def test_messy_transcripts_map_to_the_right_command(text, name, slots):
    found = CMDS.match(text)
    assert found is not None, CMDS.rank(text)[:3]
    assert found.name == name
    assert found.slots == slots
    assert CMDS.min_confidence <= found.confidence <= 1.0


@pytest.mark.parametrize("text", UNRELATED)
def test_unrelated_text_matches_nothing(text):
    assert CMDS.match(text) is None


def test_confidence_floor_refuses_the_closest_poor_match():
    text = "pay the bills"  # "pay" is one letter away from "play"
    ranked = CMDS.rank(text)
    assert ranked[0].name == "play"  # there *is* a closest candidate...
    assert 0 < ranked[0].confidence < CMDS.min_confidence
    assert CMDS.match(text) is None  # ...and it is not returned


def test_the_floor_is_configurable():
    text = "set temperature to 21"  # the unit was not said
    default = assistant().match(text)
    assert default is not None and default.missing == ["degrees"]
    strict = Commands(min_confidence=0.9).add("set temperature to {value:number} degrees")
    assert strict.match(text) is None
    assert strict.match("set temperature to 21 degrees") is not None


def test_poor_matches_have_low_confidence():
    for text in UNRELATED:
        ranked = CMDS.rank(text)
        assert ranked and ranked[0].confidence < CMDS.min_confidence


# ------------------------------------------------------------ specificity


def test_more_specific_pattern_wins_and_other_is_an_alternative():
    found = CMDS.match("play hello by adele")
    assert found.name == "play_by"
    assert found.slots == {"song": "hello", "artist": "adele"}
    assert [a.name for a in found.alternatives] == ["play"]
    assert found.alternatives[0].slots == {"song": "hello by adele"}
    assert found.alternatives[0].confidence >= found.confidence  # it tied or beat it on score
    assert "more specific" in found.chosen_because
    assert "Also matched: 'play'" in found.summary()


def test_fixed_words_beat_a_text_slot():
    cmds = Commands()
    cmds.add("turn on the {device:text}", name="any_device")
    cmds.add("turn on the kitchen light", name="kitchen_light")
    found = cmds.match("turn on the kitchen light")
    assert found.name == "kitchen_light"
    assert [a.name for a in found.alternatives] == ["any_device"]
    other = cmds.match("turn on the porch light")
    assert other.name == "any_device"
    assert other.slots == {"device": "porch light"}


def test_single_match_has_no_alternatives():
    found = CMDS.match("lock the front door")
    assert found.alternatives == []
    assert found.chosen_because == ""
    assert "Also matched" not in found.summary()


def test_contradicting_word_is_not_a_match():
    only_on = Commands().add("turn on the kitchen light", name="on")
    assert only_on.match("turn on the kitchen light") is not None
    ranked = only_on.rank("turn off the kitchen light")
    assert ranked[0].replaced == [{"expected": "on", "heard": "off"}]
    assert only_on.match("turn off the kitchen light") is None


# ------------------------------------------------------------ fuzzy evidence


def test_misheard_word_is_reported_and_clean_input_reports_none():
    messy = CMDS.match("set the tempreture to 19 degrees")
    assert messy.corrections == [
        {"heard": "tempreture", "as": "temperature", "how": "edit distance 2"}
    ]
    assert "tempreture" in messy.summary()
    clean = CMDS.match("set temperature to 19 degrees")
    assert clean.corrections == []
    assert clean.confidence == pytest.approx(1.0)
    assert clean.confidence > messy.confidence


def test_filler_words_are_ignored_and_listed():
    messy = CMDS.match("um could you please set the temperature to twenty degrees thank you")
    assert messy.name == "thermostat" and messy.slots == {"value": 20}
    assert messy.ignored == ["um", "could you please", "thank you"]
    assert messy.extra == ["the"]
    clean = CMDS.match("set temperature to twenty degrees")
    assert clean.ignored == [] and clean.extra == []


def test_reordered_word_is_found_and_flagged():
    moved = CMDS.match("turn the kitchen light off")
    assert moved.slots == {"state": "off", "device": "kitchen light"}
    assert moved.reordered == ["{state}"]
    in_order = CMDS.match("turn off the kitchen light")
    assert in_order.reordered == []
    assert in_order.confidence > moved.confidence


def test_choice_heard_with_a_doubled_letter_dropped():
    found = CMDS.match("turn of the lights")
    assert found.slots["state"] == "off"
    assert found.slot_words["state"] == "of"
    assert found.corrections == [{"heard": "of", "as": "off", "how": "doubled letter"}]


def test_sound_alike_word():
    found = CMDS.match("what's the whether in london")
    assert found.corrections[0]["how"] == "sounds alike"


def test_plural_is_close_to_exact():
    cmds = Commands().add("switch on the light")
    assert cmds.match("switch on the lights").corrections[0]["how"] == "plural"
    assert cmds.match("switch on the lights").confidence > 0.9


def test_split_and_joined_words():
    cmds = Commands()
    cmds.add("open the database", name="db")
    cmds.add("set up the {device}", name="setup")
    split = cmds.match("open the data base")
    assert split.name == "db"
    assert split.corrections[0]["how"] == "split into two words"
    joined = cmds.match("setup the printer")
    assert joined.name == "setup" and joined.slots == {"device": "printer"}
    assert joined.corrections[0] == {"heard": "setup", "as": "set up", "how": "run together as one word"}


def test_short_words_need_to_be_exact():
    cmds = Commands().add("turn {state:on|off} the fan")
    # "in" is one letter from "on" but short words get no typo allowance
    assert cmds.match("turn in the fan") is None
    assert cmds.match("turn on the fan").slots == {"state": "on"}


def test_optional_words_cost_nothing_when_absent():
    cmds = Commands().add("turn on [the] lights")
    with_the = cmds.match("turn on the lights")
    without = cmds.match("turn on lights")
    assert with_the.confidence == pytest.approx(1.0)
    assert without.confidence == pytest.approx(1.0)
    assert without.missing == []
    required = Commands().add("turn on the lights").match("turn on lights")
    assert required.missing == ["the"] and required.confidence < 1.0


# ------------------------------------------------------------ slots


def test_number_slots_are_typed():
    assert CMDS.match("set temperature to twenty one degrees").slots["value"] == 21
    assert isinstance(CMDS.match("set temperature to twenty one degrees").slots["value"], int)
    assert CMDS.match("set temperature to 21.0 degrees").slots["value"] == 21.0
    assert isinstance(CMDS.match("set temperature to 21.0 degrees").slots["value"], float)
    assert CMDS.match("set temperature to two point five degrees").slots["value"] == 2.5
    assert CMDS.match("set temperature to minus three degrees").slots["value"] == -3
    assert CMDS.match("set temperature to a hundred and five degrees").slots["value"] == 105
    assert CMDS.match("set temperature to two and a half degrees").slots["value"] == 2.5


def test_text_slot_keeps_original_spelling_and_drops_hesitations():
    found = CMDS.match("remind me to email AC/DC's manager, um, tomorrow at 9:30")
    assert found.slots == {"task": "email AC/DC's manager tomorrow", "time": "9:30"}


def test_text_slot_trims_politeness_but_keeps_real_words():
    assert CMDS.match("play despacito please").slots == {"song": "despacito"}
    assert CMDS.match("play just like heaven").slots == {"song": "just like heaven"}
    assert CMDS.match("play thank you next").slots == {"song": "thank you next"}
    # text that is only politeness is not a song
    assert CMDS.match("play please") is None


def test_framing_is_trimmed_from_a_text_slot_that_opens_the_pattern():
    cmds = Commands().add("{task:text} at {time:text}", name="at")
    assert cmds.match("could you water the plants at six").slots == {
        "task": "water the plants", "time": "six",
    }
    # inside a message it is content and stays
    msg = Commands().add("send a message saying {message:text}")
    assert msg.match("send a message saying can you pick me up").slots == {
        "message": "can you pick me up"
    }


def test_word_slot_takes_exactly_one_word():
    cmds = Commands().add("turn on the {device}")
    found = cmds.match("turn on the heater please")
    assert found.slots == {"device": "heater"}
    assert found.ignored == ["please"]


def test_multi_word_choice_returns_the_option_as_written():
    cmds = Commands().add("open the {room:Living Room|kitchen} door")
    found = cmds.match("open the living room door")
    assert found.slots == {"room": "Living Room"}
    assert cmds.match("open the livin room door").slots == {"room": "Living Room"}


# ------------------------------------------------------------ unicode, case, empty


def test_unicode_and_mixed_case():
    cmds = Commands()
    cmds.add("schalte das licht in der {raum} ein", name="licht")
    cmds.add("allume la lumière du {pièce:salon|garage}", name="lumiere")
    cmds.add("play {song:text}", name="play")
    german = cmds.match("Schalte das LICHT in der KÜCHE ein")
    assert german.name == "licht" and german.slots == {"raum": "KÜCHE"}
    french = cmds.match("ALLUME LA LUMIERE DU Salon")  # accents dropped by the transcriber
    assert french.name == "lumiere" and french.slots == {"pièce": "salon"}
    song = cmds.match("play Café del Mar ☕ por favor")
    assert song.slots == {"song": "Café del Mar ☕ por favor"}
    fullwidth = cmds.match("ｐｌａｙ Ｄｅｓｐａｃｉｔｏ")
    assert fullwidth.name == "play"


def test_accents_fold_only_where_transcribers_drop_them():
    cmds = Commands().add("fly to istanbul", name="fly")
    assert cmds.match("Fly to İSTANBUL").confidence == pytest.approx(1.0)
    # marks that are part of the spelling in other scripts are kept, so words
    # still match themselves exactly and differ from their unmarked forms
    hindi = Commands().add("किताब खोलो", name="open_book")
    assert hindi.match("किताब खोलो").confidence == pytest.approx(1.0)
    japanese = Commands().add("がっこう {x}", name="school")
    assert japanese.match("がっこう えき").slots == {"x": "えき"}
    assert fold("がっこう") != fold("かっこう")  # the dakuten is part of the word
    assert fold("Küche") == fold("KUCHE")  # a Latin accent is not


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "?!...", "☕💡", "um uh", "please"])
def test_empty_or_wordless_text_returns_none(text):
    assert CMDS.match(text) is None


def test_documented_limit_a_text_slot_accepts_anything():
    # the README warns about this; keep the warning honest
    found = Commands().add("call {contact:text}").match("call of duty is a great game")
    assert found is not None and found.slots == {"contact": "of duty is a great game"}


def test_no_commands_registered_returns_none():
    assert Commands().match("turn on the lights") is None
    assert Commands().rank("turn on the lights") == []


def test_match_requires_a_string():
    with pytest.raises(TypeError):
        CMDS.match(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CMDS.match(b"turn on the lights")  # type: ignore[arg-type]


def test_custom_fillers():
    plain = Commands().add("lock the door")
    assert plain.match("yo lock the door").extra == ["yo"]
    custom = Commands(fillers=["yo", "if you would"]).add("lock the door")
    found = custom.match("yo lock the door if you would")
    assert found.ignored == ["yo", "if you would"] and found.extra == []
    assert found.confidence == pytest.approx(1.0)


def test_rank_lists_every_command_best_first():
    ranked = CMDS.rank("set a timer for five minutes")
    assert len(ranked) == len(CMDS)
    assert ranked[0].name == "timer"
    scores = [m.confidence for m in ranked[1:]]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(m, Match) for m in ranked)


def test_matching_is_deterministic():
    first = CMDS.match("um could you set the tempreture to twenty one degrees please").to_dict()
    for _ in range(3):
        again = CMDS.match("um could you set the tempreture to twenty one degrees please").to_dict()
        assert again == first


def test_matching_is_fast_enough_for_a_voice_loop():
    cmds = assistant()
    for n in range(40):
        cmds.add("open application number %d please" % n, name="app%d" % n)
    start = time.perf_counter()
    for text, _, _ in MESSY:
        cmds.match(text)
    per_utterance = (time.perf_counter() - start) / len(MESSY)
    assert per_utterance < 0.25  # generous; typically a few milliseconds
