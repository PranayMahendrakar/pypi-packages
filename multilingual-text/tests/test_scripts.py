"""script_of(), is_rtl() and the script machinery underneath them."""
import pytest

import multilingual_text as mlt
from multilingual_text import char_script_of, is_rtl, script_names, script_of


@pytest.mark.parametrize(
    "text,script",
    [
        ("hello", "Latin"),
        ("Grüße", "Latin"),
        ("Tiếng Việt", "Latin"),
        ("мир", "Cyrillic"),
        ("Ελλάδα", "Greek"),
        ("مرحبا", "Arabic"),
        ("שלום", "Hebrew"),
        ("नमस्ते", "Devanagari"),
        ("বাংলা", "Bengali"),
        ("ਪੰਜਾਬੀ", "Gurmukhi"),
        ("ગુજરાતી", "Gujarati"),
        ("தமிழ்", "Tamil"),
        ("తెలుగు", "Telugu"),
        ("ಕನ್ನಡ", "Kannada"),
        ("മലയാളം", "Malayalam"),
        ("ไทย", "Thai"),
        ("한국어", "Hangul"),
        ("中文", "Han"),
        ("ひらがな", "Hiragana"),
        ("カタカナ", "Katakana"),
    ],
)
def test_script_of(text, script):
    assert script_of(text) == script


def test_script_of_empty_and_symbol_only():
    assert script_of("") == "Unknown"
    assert script_of("42 + 7 = 49") == "Common"
    assert script_of("!!!???") == "Common"
    assert script_of("   ") == "Common"


def test_script_of_uses_the_majority():
    assert script_of("hello hello мир") == "Latin"
    assert script_of("привет привет hi") == "Cyrillic"


def test_script_of_rejects_non_strings():
    with pytest.raises(TypeError):
        script_of(7)


@pytest.mark.parametrize(
    "char,script",
    [("A", "Latin"), ("é", "Latin"), ("я", "Cyrillic"), ("α", "Greek"),
     ("ا", "Arabic"), ("א", "Hebrew"), ("क", "Devanagari"), ("中", "Han"),
     ("あ", "Hiragana"), ("ア", "Katakana"), ("한", "Hangul"), ("ก", "Thai"),
     ("1", "Common"), (" ", "Common"), ("!", "Common")],
)
def test_char_script_of(char, script):
    assert char_script_of(char) == script


def test_combining_marks_belong_to_their_script():
    assert char_script_of("ि") == "Devanagari"   # Devanagari vowel sign i
    assert char_script_of("ְ") == "Hebrew"        # Hebrew point sheva


def test_script_names_lists_what_can_be_reported():
    names = script_names()
    assert names == sorted(names)
    for expected in ("Latin", "Cyrillic", "Devanagari", "Han", "Thai", "Hangul"):
        assert expected in names


@pytest.mark.parametrize(
    "text,expected",
    [
        ("مرحبا بالعالم", True),
        ("שלום עולם", True),
        ("hello", False),
        ("", False),
        ("12345", False),
        ("مرحبا hi", True),
        ("Hello مرحبا", False),
        ("नमस्ते", False),
    ],
)
def test_is_rtl(text, expected):
    assert is_rtl(text) is expected


def test_is_rtl_rejects_non_strings():
    with pytest.raises(TypeError):
        is_rtl(None)


def test_rtl_scripts_are_advertised():
    assert "Arabic" in mlt.RTL_SCRIPTS
    assert "Hebrew" in mlt.RTL_SCRIPTS
    assert "Latin" not in mlt.RTL_SCRIPTS


def test_supported_languages_are_advertised():
    assert len(mlt.SUPPORTED_LANGUAGES) == 30
    assert "en" in mlt.SUPPORTED_LANGUAGES
    assert "und" not in mlt.SUPPORTED_LANGUAGES
    for code in mlt.SUPPORTED_LANGUAGES:
        assert mlt.language_name(code) != code, code
