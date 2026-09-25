"""Language codes and names, in any script."""

from __future__ import annotations

import pytest

from offline_stt_router import language_name, normalize_language
from offline_stt_router._languages import (
    WHISPER_LANGUAGES,
    all_whisper_codes,
    vosk_language_from_name,
    whisper_tier,
)


@pytest.mark.parametrize(
    "given, code",
    [
        ("en", "en"), ("EN-us", "en"), ("English", "en"), ("  english ", "en"), ("eng", "en"),
        ("hi", "hi"), ("Hindi", "hi"), ("हिन्दी", "hi"), ("हिंदी", "hi"), ("hi-IN", "hi"),
        ("pt_BR", "pt"), ("Português", "pt"), ("Español", "es"), ("Français", "fr"),
        ("zh-Hant-TW", "zh"), ("中文", "zh"), ("Mandarin", "zh"), ("粵語", "yue"), ("zh-HK", "yue"),
        ("日本語", "ja"), ("한국어", "ko"), ("Русский", "ru"), ("العربية", "ar"), ("Ελληνικά", "el"),
        ("jv", "jw"), ("nb", "no"), ("Filipino", "tl"), ("Haitian Creole", "ht"),
        ("Esperanto", "eo"), ("tlh", "tlh"),
        (None, None), ("", None), ("auto", None), ("AUTO", None),
    ],
)
def test_normalize_language(given, code):
    assert normalize_language(given) == code


def test_unknown_language_text_raises():
    with pytest.raises(ValueError, match="unknown language"):
        normalize_language("not a language at all")
    with pytest.raises(TypeError):
        normalize_language(42)


def test_language_names():
    assert language_name("hi") == "Hindi"
    assert language_name("ht") == "Haitian Creole"
    assert language_name(None) == "any language (auto-detect)"
    assert language_name("tlh") == "tlh"


def test_tiers():
    assert whisper_tier("en") == "strong"
    assert whisper_tier("uk") == "moderate"
    assert whisper_tier("hi") == "limited"
    assert whisper_tier("eo") == "unsupported"
    assert whisper_tier(None) == "moderate"


def test_whisper_language_list():
    assert len(WHISPER_LANGUAGES) == 100
    assert len(all_whisper_codes(v3=False)) == 99 and "yue" not in all_whisper_codes(v3=False)


@pytest.mark.parametrize(
    "folder, code",
    [
        ("vosk-model-small-en-us-0.15", "en"), ("vosk-model-en-in-0.5", "en"), ("vosk-model-cn-0.22", "zh"),
        ("vosk-model-small-hi-0.22", "hi"), ("vosk-model-small-vn-0.4", "vi"), ("vosk-model-kz-0.15", "kk"),
        ("vosk-model-el-gr-0.7", "el"), ("vosk-model-small-eo-0.42", "eo"),
        ("my-model", None), ("vosk-model-", None), ("vosk-model-small-xx-0.1", None),
    ],
)
def test_vosk_folder_names(folder, code):
    assert vosk_language_from_name(folder) == code
