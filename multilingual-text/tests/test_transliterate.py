"""transliterate(): the three covered scripts, and honesty about the rest."""
import json
import unicodedata

import pytest

from multilingual_text import SUPPORTED_SCRIPTS, Transliteration, transliterate


@pytest.mark.parametrize(
    "source,expected",
    [
        ("नमस्ते", "namaste"),
        ("नमस्ते दुनिया", "namaste duniyā"),
        ("हिन्दी", "hindī"),
        ("भाषा", "bhāṣā"),
        ("मराठी", "marāṭhī"),
        ("भारत", "bhārata"),
        ("कमल", "kamala"),
        ("क्या", "kyā"),
        # nukta on ja is the z sound: angrezi, not angre-r-i.
        ("अंग्रेज़ी", "aṃgrezī"),
        ("ॐ", "oṃ"),
    ],
)
def test_devanagari_follows_iast(source, expected):
    assert str(transliterate(source)) == expected


def test_devanagari_virama_suppresses_the_inherent_vowel():
    assert str(transliterate("सत्य")) == "satya"
    assert str(transliterate("सत")) == "sata"


def test_output_is_nfc_composed():
    """Romanised text must come back composed, not letter-plus-combining-mark.

    Regression: the IAST and Greek tables were saved half-decomposed, so
    ``transliterate("Αθήνα")`` returned
    "Athēna" instead of "Athina" -- equal on screen, unequal to
    ``==``, which is the one thing a caller relies on.
    """
    for source in ["भाषा", "Αθήνα",
                   "ॐ", "हिन्दी",
                   "Κόσμος"]:
        out = str(transliterate(source))
        assert out == unicodedata.normalize("NFC", out), repr(out)


def test_devanagari_digits_and_danda():
    assert str(transliterate("२०२६।")) == "2026."


@pytest.mark.parametrize(
    "source,expected",
    [
        ("привет", "privet"),
        ("Привет, мир!", "Privet, mir!"),
        ("Москва", "Moskva"),
        ("ШКОЛА", "SHKOLA"),
        ("щука", "shchuka"),
        ("Ёлка", "Yolka"),
        ("Съезд", "Sezd"),
        ("чай", "chay"),
    ],
)
def test_cyrillic_russian(source, expected):
    assert str(transliterate(source)) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Київ", "Kyiv"),
        ("Україна", "Ukraina"),
        ("привіт світ", "pryvit svit"),
        ("Євген", "Yevhen"),
        ("Яблуко", "Yabluko"),
    ],
)
def test_cyrillic_switches_to_ukrainian_when_it_sees_ukrainian_letters(source, expected):
    assert str(transliterate(source)) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Ελλάδα", "Ellada"),
        ("Αθήνα", "Athina"),
        ("ΑΘΗΝΑ", "ATHINA"),
        ("φιλοσοφία", "filosofia"),
        ("Ψυχή", "Psychi"),
        ("κόσμος", "kosmos"),
    ],
)
# Greek romanisation follows ISO 843, the standard the README names. It covers
# MODERN Greek, so beta is "v" and phi is "f". These expectations previously held
# the classical scheme, which is what let the code and the documentation disagree.
def test_greek(source, expected):
    assert str(transliterate(source)) == expected


def test_greek_final_sigma_is_an_s():
    assert str(transliterate("κόσμος")).endswith("s")


def test_ascii_target_folds_the_diacritics():
    assert str(transliterate("भाषा", to="ascii")) == "bhasa"
    assert str(transliterate("Αθήνα", to="ascii")) == "Athina"
    assert str(transliterate("नमस्ते दुनिया", to="ascii")) == "namaste duniya"


def test_ascii_target_output_is_actually_ascii():
    for text in ["नमस्ते दुनिया", "Привет, мир!", "Ελλάδα", "hello"]:
        assert str(transliterate(text, to="ascii")).isascii(), text


def test_latin_target_is_the_default():
    assert transliterate("भाषा").target == "latin"
    assert str(transliterate("भाषा")) == str(transliterate("भाषा", to="latin"))


def test_result_is_a_string_subclass():
    out = transliterate("Привет")
    assert isinstance(out, Transliteration)
    assert isinstance(out, str)
    assert out == "Privet"
    assert out.upper() == "PRIVET"
    assert "-".join([out, out]) == "Privet-Privet"
    assert "%s!" % out == "Privet!"


def test_result_reports_what_it_did():
    out = transliterate("नमस्ते")
    assert out.supported is True
    assert out.changed is True
    assert out.converted == ("Devanagari",)
    assert out.untouched == ()
    assert out.scripts == ("Devanagari",)
    assert "Devanagari romanised" in out.note
    assert out.summary().startswith("multilingual-text: transliterate to latin")


def test_already_latin_text_is_unchanged_and_says_so():
    out = transliterate("hello world")
    assert out == "hello world"
    assert out.changed is False
    assert out.supported is True
    assert "already Latin" in out.note


def test_unsupported_script_comes_back_untouched_with_a_reason():
    out = transliterate("مرحبا بالعالم")
    assert out == "مرحبا بالعالم"
    assert out.changed is False
    assert out.supported is False
    assert out.untouched == ("Arabic",)
    assert "Arabic not covered" in out.note


@pytest.mark.parametrize(
    "text,script",
    [
        ("日本語", "Han"),
        ("こんにちは", "Hiragana"),
        ("안녕하세요", "Hangul"),
        ("สวัสดี", "Thai"),
        ("שלום", "Hebrew"),
        ("வணக்கம்", "Tamil"),
        ("ಕನ್ನಡ", "Kannada"),
    ],
)
def test_every_uncovered_script_is_named_not_mangled(text, script):
    out = transliterate(text)
    assert out == text
    assert out.untouched == (script,)
    assert out.supported is False


def test_mixed_supported_and_unsupported():
    out = transliterate("Привет 日本 नमस्ते")
    assert out.converted == ("Cyrillic", "Devanagari")
    assert out.untouched == ("Han",)
    assert "日本" in out
    assert out.startswith("Privet")
    assert out.endswith("namaste")


def test_to_dict_is_json_safe():
    payload = transliterate("Привет 日本").to_dict()
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["supported"] is False
    assert restored["converted"] == ["Cyrillic"]
    assert restored["untouched"] == ["Han"]


def test_bad_target_is_a_clear_value_error():
    with pytest.raises(ValueError) as excinfo:
        transliterate("x", to="cyrillic")
    assert "latin" in str(excinfo.value)


def test_transliterate_rejects_non_strings():
    with pytest.raises(TypeError):
        transliterate(None)


def test_supported_scripts_are_advertised():
    assert SUPPORTED_SCRIPTS == ("Devanagari", "Cyrillic", "Greek")


def test_punctuation_and_digits_survive():
    out = transliterate("Привет, мир! 42 (да)")
    assert out == "Privet, mir! 42 (da)"


def test_transliteration_is_deterministic():
    assert transliterate("नमस्ते") == transliterate("नमस्ते")


def test_greek_punctuation_is_romanised():
    """Regression: both _GREEK punctuation entries used to be unreachable.

    U+0387 ANO TELEIA and U+037E GREEK QUESTION MARK are canonical singletons,
    so fold_marks() rewrote them into a middle dot and an ASCII semicolon
    before the lookup ran, and neither entry could ever fire.
    """
    assert str(transliterate("ναι· οχι")) == "nai; ochi"
    assert str(transliterate("τι κανεις;")) == "ti kaneis?"


def test_greek_punctuation_is_romanised_in_ascii_mode_too():
    out = transliterate("ναι· οχι", to="ascii")
    assert str(out) == "nai; ochi"
    assert out.converted == ("Greek",)


def test_an_ordinary_ascii_semicolon_in_greek_text_is_left_alone():
    """Only the two Greek-block marks convert; a typed ';' is Common."""
    assert str(transliterate("ναι; οχι")) == "nai; ochi"



def test_serbian_and_macedonian_letters_are_romanised():
    """Regression: these letters were missing from the table, so a Serbian name came
    back half romanised - and the result still said supported=True, which is worse than
    refusing outright because the caller had no way to tell the output was incomplete."""
    from multilingual_text import transliterate

    for source, expected in [
        ("Ђорђе Љубић", "Djordje Ljubic"),
        ("Ѓорѓи Ќулавков", "Gorgi Kulavkov"),
    ]:
        result = transliterate(source)
        assert str(result) == expected
        assert str(result).isascii(), "nothing may be left behind in the original script"
        assert result.supported is True


def test_greek_follows_the_standard_the_readme_names():
    from multilingual_text import transliterate

    # ISO 843 romanises modern Greek: beta is "v", phi is "f", eta is "i"
    assert str(transliterate("Βιβλιοθήκη φως")) == "Vivliothiki fos"
    assert str(transliterate("Αθήνα")) == "Athina"


def test_other_cyrillic_languages_are_unaffected():
    from multilingual_text import transliterate

    assert str(transliterate("Москва")) == "Moskva"
    assert str(transliterate("Київ")) == "Kyiv"
    assert str(transliterate("Пловдив")) == "Plovdiv"


def test_a_script_with_no_table_is_reported_as_unsupported():
    from multilingual_text import transliterate

    result = transliterate("北京 Beijing")
    assert result.supported is False
    assert "北京" in str(result), "unconvertible text is returned as it was"
