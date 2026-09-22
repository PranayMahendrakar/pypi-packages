"""Language detection over all thirty supported languages."""
import pytest

import multilingual_text as mlt
from multilingual_text import Detection, detect, detect_batch

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog and then runs back home",
    "es": "El rápido zorro marrón salta sobre el perro perezoso y luego vuelve a casa",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux et rentre chez lui",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund und geht dann nach Hause",
    "it": "La volpe marrone veloce salta sopra il cane pigro e poi torna a casa",
    "pt": "A raposa marrom rápida salta sobre o cão preguiçoso e depois volta para casa",
    "nl": "De snelle bruine vos springt over de luie hond en gaat dan naar huis",
    "pl": "Szybki brązowy lis przeskakuje nad leniwym psem i wraca do domu",
    "sv": "Den snabba bruna räven hoppar över den lata hunden och går sedan hem",
    "tr": "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar ve sonra eve gider",
    "vi": "Con cáo nâu nhanh nhẹn nhảy qua con chó lười biếng rồi đi về nhà",
    "id": "Rubah cokelat yang cepat melompati anjing yang malas itu lalu pulang",
    "ru": "Быстрая коричневая лиса прыгает через ленивую собаку и возвращается домой",
    "uk": "Швидка руда лисиця стрибає через лінивого пса і повертається додому",
    "el": "Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί",
    "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول ثم يعود إلى المنزل",
    "he": "השועל החום המהיר קופץ מעל הכלב העצלן ואז חוזר הביתה",
    "hi": "यह एक बहुत अच्छा दिन है और मैं बहुत खुश हूँ",
    "mr": "हा एक खूप चांगला दिवस आहे आणि मी खूप आनंदी आहे",
    "bn": "এটি একটি খুব ভাল দিন এবং আমি খুব খুশি",
    "ta": "இது ஒரு மிகவும் நல்ல நாள் மற்றும் நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்",
    "te": "ఇది చాలా మంచి రోజు మరియు నేను చాలా సంతోషంగా ఉన్నాను",
    "gu": "આ એક ખૂબ સારો દિવસ છે અને હું ખૂબ ખુશ છું",
    "pa": "ਇਹ ਇੱਕ ਬਹੁਤ ਵਧੀਆ ਦਿਨ ਹੈ ਅਤੇ ਮੈਂ ਬਹੁਤ ਖੁਸ਼ ਹਾਂ",
    "kn": "ಇದು ತುಂಬಾ ಒಳ್ಳೆಯ ದಿನ ಮತ್ತು ನಾನು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ",
    "ml": "ഇത് വളരെ നല്ല ഒരു ദിവസമാണ് ഞാൻ വളരെ സന്തോഷവാനാണ്",
    "zh": "我们今天去北京看朋友，因为天气很好",
    "ja": "私は日本語を勉強していますがとても楽しいです",
    "ko": "저는 한국어를 공부하고 있습니다 그리고 매우 재미있습니다",
    "th": "วันนี้อากาศดีมากและเราไปเที่ยวกับเพื่อน",
}

EXPECTED_SCRIPT = {
    "en": "Latin", "ru": "Cyrillic", "el": "Greek", "ar": "Arabic", "he": "Hebrew",
    "hi": "Devanagari", "mr": "Devanagari", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "gu": "Gujarati", "pa": "Gurmukhi", "kn": "Kannada",
    "ml": "Malayalam", "zh": "Han", "ja": "Hiragana", "ko": "Hangul", "th": "Thai",
}


@pytest.mark.parametrize("code", sorted(SENTENCES))
def test_every_supported_language_is_detected(code):
    result = detect(SENTENCES[code])
    assert result.language == code, result.summary()
    assert result.reliable, result.summary()
    assert 0.0 < result.confidence <= 1.0


def test_every_advertised_language_has_a_sentence():
    assert set(SENTENCES) == set(mlt.SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("code,script", sorted(EXPECTED_SCRIPT.items()))
def test_script_is_reported(code, script):
    assert detect(SENTENCES[code]).script == script


def test_top_controls_how_many_candidates_come_back():
    one = detect(SENTENCES["es"], top=1)
    three = detect(SENTENCES["es"], top=3)
    assert one.alternatives == []
    assert len(three.alternatives) == 2
    assert len(three.ranked) == 3
    assert three.ranked[0] == (three.language, three.confidence)
    assert [code for code, _ in three.alternatives] != [three.language]


def test_alternatives_are_ordered_by_confidence():
    result = detect(SENTENCES["pt"], top=4)
    confidences = [conf for _, conf in result.ranked]
    assert confidences == sorted(confidences, reverse=True)


def test_top_beyond_the_candidate_list_is_not_an_error():
    result = detect(SENTENCES["th"], top=10)
    assert result.language == "th"
    assert result.alternatives == []


def test_russian_and_ukrainian_are_told_apart():
    assert detect("Я живу в Києві і дуже люблю це місто").language == "uk"
    assert detect("Я живу в Москве и очень люблю этот город").language == "ru"


def test_hindi_and_marathi_are_told_apart():
    assert detect("मैं यहाँ रहता हूँ और यह शहर बहुत अच्छा है").language == "hi"
    assert detect("मी इथे राहतो आणि हे शहर खूप चांगले आहे").language == "mr"


def test_chinese_and_japanese_are_told_apart():
    assert detect("我们今天去北京看朋友").language == "zh"
    assert detect("今日は友達と東京に行きます").language == "ja"


def test_name_and_language_agree():
    result = detect(SENTENCES["de"])
    assert result.name == "German"
    assert mlt.language_name(result.language) == "German"


def test_summary_is_plain_ascii_even_for_non_latin_text():
    for code in ("ar", "hi", "zh", "th", "ru"):
        assert detect(SENTENCES[code], top=2).summary().isascii()


def test_summary_mentions_the_code_and_the_package():
    text = detect(SENTENCES["fr"]).summary()
    assert text.startswith("multilingual-text:")
    assert "fr (French)" in text


def test_to_dict_is_json_safe():
    import json

    payload = detect(SENTENCES["ja"], top=3).to_dict()
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["language"] == "ja"
    assert restored["script"] == "Hiragana"
    assert restored["reliable"] is True
    assert isinstance(restored["alternatives"], list)


def test_detection_is_a_frozen_dataclass():
    result = detect("hello world this is english text")
    with pytest.raises(Exception):
        result.language = "fr"


def test_detect_is_deterministic():
    first = detect(SENTENCES["sv"], top=3)
    second = detect(SENTENCES["sv"], top=3)
    assert first == second


def test_detect_batch_keeps_order():
    results = detect_batch([SENTENCES["en"], SENTENCES["ru"], SENTENCES["th"]])
    assert [r.language for r in results] == ["en", "ru", "th"]
    assert all(isinstance(r, Detection) for r in results)


def test_detect_batch_handles_an_empty_list():
    assert detect_batch([]) == []


def test_detect_batch_accepts_any_iterable():
    results = detect_batch(iter([SENTENCES["en"], ""]))
    assert [r.language for r in results] == ["en", "und"]


def test_detect_batch_rejects_a_bare_string():
    with pytest.raises(TypeError) as excinfo:
        detect_batch("not a list")
    assert "detect()" in str(excinfo.value)


def test_detect_batch_rejects_a_non_string_item():
    with pytest.raises(TypeError) as excinfo:
        detect_batch(["fine", 42])
    assert "item 1" in str(excinfo.value)


def test_detect_rejects_non_strings():
    with pytest.raises(TypeError):
        detect(None)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3"])
def test_detect_rejects_a_bad_top(bad):
    with pytest.raises(ValueError):
        detect("hello", top=bad)


def test_rtl_flag_on_the_detection():
    assert detect(SENTENCES["ar"]).rtl is True
    assert detect(SENTENCES["he"]).rtl is True
    assert detect(SENTENCES["en"]).rtl is False


def test_inverted_marks_raise_confidence_in_spanish():
    """Regression: DISTINCTIVE['es'] holds the inverted marks, unused.

    detect() builds its `letters` string from characters whose Unicode
    category starts with L or M.  Both inverted marks are category Po, so they
    were filtered out before the scorer saw them and the two spellings scored
    byte-identically.
    """
    plain = detect("Como estas")
    marked = detect("¿Como estas?")
    assert marked.language == plain.language == "es"
    assert marked.confidence > plain.confidence


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/a/b?c=1",
        "http://example.com",
        "www.example.com",
        "WWW.Example.COM",
        "hola@ejemplo.es",
    ],
)
def test_a_bare_address_is_not_a_language(text):
    """Regression: 'com' and 'a' are both Portuguese function words.

    A URL used to come back as Portuguese at 0.57.  It carries no natural
    language, so it gets the same answer digits and punctuation already get.
    """
    result = detect(text)
    assert result.language == "und", (text, result.language, result.confidence)
    assert result.confidence == 0.0
    assert result.reliable is False


def test_an_address_inside_a_sentence_does_not_stop_detection():
    result = detect(
        "Visita https://ejemplo.com para mas informacion sobre el producto"
    )
    assert result.language == "es"
    assert result.reliable is True


def test_abbreviations_are_not_mistaken_for_addresses():
    """Only unambiguous address shapes go; "e.g." and "i.e." must survive."""
    result = detect(
        "e.g. this is an ordinary English sentence with abbreviations i.e. these"
    )
    assert result.language == "en"
    assert result.reliable is True

