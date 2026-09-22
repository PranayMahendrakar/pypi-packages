"""The awkward inputs: empty, tiny, mixed, digit-only, punctuation-only."""
import pytest

import multilingual_text as mlt
from multilingual_text import UNDETERMINED, detect, normalize, script_of, transliterate


def test_empty_string_is_undetermined_not_an_exception():
    result = detect("")
    assert result.language == UNDETERMINED
    assert result.confidence == 0.0
    assert result.reliable is False
    assert result.alternatives == []
    assert result.characters == 0
    assert result.script == "Unknown"
    assert "not enough to go on" in result.summary()


@pytest.mark.parametrize("text", ["12345", "0", "٤٢", "!!! ??? ...", "   ", "\t\n", "---"])
def test_text_without_letters_is_undetermined(text):
    result = detect(text)
    assert result.language == UNDETERMINED
    assert result.confidence == 0.0
    assert result.reliable is False


def test_digits_and_punctuation_report_the_common_script():
    assert detect("12345").script == "Common"
    assert detect("!!!").script == "Common"
    assert script_of("42 + 7 = 49") == "Common"


@pytest.mark.parametrize("char", ["A", "z", "я", "你", "あ", "ह", "ก", "א", "ع", "ا"])
def test_a_single_character_never_raises(char):
    result = detect(char)
    assert result.characters == 1
    assert result.reliable is False
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.language, str)


def test_a_single_character_still_reports_its_script():
    assert detect("你").script == "Han"
    assert detect("ก").script == "Thai"
    assert detect("я").script == "Cyrillic"


def test_short_text_is_never_marked_reliable():
    for text in ["Hola", "Oui", "Da", "Hi", "Bonjour", "नम"]:
        result = detect(text)
        assert result.reliable is False, "%r claimed to be reliable" % text
        assert result.confidence < 0.8


def test_a_full_sentence_is_marked_reliable():
    result = detect("This is a long enough sentence in English to be sure about")
    assert result.reliable is True
    assert result.confidence >= 0.6


def test_mixed_script_reports_the_dominant_script_and_flags_the_mixture():
    result = detect("Hello world, это смешанный текст с двумя алфавитами")
    assert result.mixed is True
    assert result.script == "Cyrillic"
    assert set(result.scripts) == {"Latin", "Cyrillic"}
    assert sum(result.scripts.values()) == pytest.approx(1.0, abs=0.01)
    assert "mixed scripts" in result.summary()


def test_single_script_text_is_not_flagged_as_mixed():
    assert detect("The quick brown fox jumps over the lazy dog").mixed is False


def test_japanese_kanji_plus_kana_is_not_called_mixed():
    result = detect("私は日本語を勉強していますがとても楽しいです")
    assert result.language == "ja"
    assert result.mixed is False


def test_japanese_with_an_english_word_is_still_japanese():
    """The dominant script is decided on families, the same way `mixed` is.

    Regression: kanji, hiragana and katakana were counted separately, so each
    lost to the single Latin run and the sentence came back as English written
    in Latin -- while `.mixed` was grouping those same three as one family.
    """
    result = detect("日本語とEnglishの混在テキストです")
    assert result.language == "ja"
    assert result.script in ("Han", "Hiragana", "Katakana")
    assert result.mixed is True
    assert result.scripts["Latin"] < 0.5


def test_scripts_are_ordered_largest_first():
    result = detect("Hello мир")
    shares = list(result.scripts.values())
    assert shares == sorted(shares, reverse=True)


def test_text_in_an_unsupported_script_is_undetermined_but_names_the_script():
    result = detect("ጤና ይስጥልኝ እንደምን አለህ")  # Amharic, not a supported language
    assert result.language == UNDETERMINED
    assert result.script == "Ethiopic"
    assert result.characters > 0


def test_normalisation_is_idempotent_on_the_defaults():
    messy = "  “Crème” — brûlée…  ٤٢​  "
    once = normalize(messy)
    assert normalize(once) == once


def test_transliteration_never_raises_on_an_unsupported_script():
    for text in ["مرحبا بالعالم", "日本語", "안녕하세요", "สวัสดี", "வணக்கம்", "שלום"]:
        out = transliterate(text)
        assert out == text
        assert out.changed is False
        assert out.supported is False
        assert out.untouched
        assert "not covered" in out.note


def test_transliteration_of_an_empty_string():
    out = transliterate("")
    assert out == ""
    assert out.note == "empty text"
    assert out.supported is True


def test_a_partly_supported_string_converts_what_it_can():
    out = transliterate("Привет 日本")
    assert out.startswith("Privet ")
    assert out.endswith("日本")
    assert out.converted == ("Cyrillic",)
    assert out.untouched == ("Han",)
    assert out.supported is False
    assert "Cyrillic romanised" in out.note and "Han not covered" in out.note


def test_detect_of_whitespace_only_input_matches_empty_behaviour():
    assert detect("\n\n\t  ").language == UNDETERMINED


def test_very_long_input_stays_fast_and_correct():
    text = "The quick brown fox jumps over the lazy dog. " * 500
    result = detect(text)
    assert result.language == "en"
    assert result.reliable is True


def test_public_api_is_importable_and_complete():
    for name in mlt.__all__:
        assert hasattr(mlt, name), name
    assert mlt.__version__ == "0.1.0"


def test_every_supported_language_has_a_name_and_every_name_a_language():
    """The advertised 30 codes and the printable names must not drift apart."""
    required = set(
        "en es fr de it pt nl ru uk ar he hi mr bn ta te gu pa kn ml "
        "zh ja ko th el tr vi id pl sv".split()
    )
    assert required == set(mlt.SUPPORTED_LANGUAGES)
    for code in mlt.SUPPORTED_LANGUAGES:
        assert mlt.LANGUAGE_NAMES[code].strip()
        assert mlt.language_name(code) == mlt.LANGUAGE_NAMES[code]
    assert set(mlt.LANGUAGE_NAMES) == required | {UNDETERMINED}
    assert mlt.language_name("zz") == "zz"


def test_transliterate_targets_are_exactly_what_the_argument_accepts():
    assert mlt.TARGETS == ("latin", "ascii")
    for target in mlt.TARGETS:
        assert str(transliterate("мир", to=target)) == "mir"
    with pytest.raises(ValueError) as caught:
        transliterate("мир", to="cyrillic")
    assert "latin" in str(caught.value) and "ascii" in str(caught.value)
