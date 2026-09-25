"""Language codes, names and how well each speech engine covers them.

Whisper's language list is fixed by its tokenizer, so it is written out here
rather than imported: importing ``whisper`` would pull in PyTorch just to read
a dictionary.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet, Optional, Tuple

#: Every language a multilingual Whisper checkpoint can be told to transcribe,
#: in the order of Whisper's tokenizer. ``yue`` exists from large-v3 onwards.
WHISPER_LANGUAGES: Dict[str, str] = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

#: Languages only large-v3 and later know (older checkpoints have 99).
V3_ONLY_LANGUAGES: FrozenSet[str] = frozenset({"yue"})

#: Languages outside Whisper's list that Vosk has models for.
EXTRA_LANGUAGES: Dict[str, str] = {"eo": "esperanto", "ky": "kyrgyz"}

#: Whisper's training data, by hours, is dominated by a handful of languages.
#: This grouping follows the per-language hours in the Whisper paper and is a
#: heuristic, not a measurement of any particular recording.
STRONG_LANGUAGES: FrozenSet[str] = frozenset(
    "en zh de es ru fr pt ko ja tr pl it sv nl ca fi id".split()
)
MODERATE_LANGUAGES: FrozenSet[str] = frozenset(
    "ar uk vi he el da ms hu ro no th cs".split()
)

_ALIASES: Dict[str, str] = {
    # other codes people use
    "jv": "jw", "nb": "no", "fil": "tl", "iw": "he", "in": "id", "ji": "yi",
    "cmn": "zh", "zho": "zh", "chi": "zh", "eng": "en", "hin": "hi",
    "spa": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de",
    "jpn": "ja", "kor": "ko", "rus": "ru", "ara": "ar", "por": "pt",
    "ita": "it", "nld": "nl", "dut": "nl", "tur": "tr", "pol": "pl",
    "ukr": "uk", "vie": "vi", "tha": "th", "ben": "bn", "tam": "ta",
    "tel": "te", "urd": "ur", "mar": "mr", "guj": "gu", "kan": "kn",
    "mal": "ml", "pan": "pa", "fas": "fa", "per": "fa", "heb": "he",
    "ell": "el", "gre": "el", "swe": "sv", "fin": "fi", "dan": "da",
    "nor": "no", "ces": "cs", "cze": "cs", "hun": "hu", "ron": "ro",
    "rum": "ro", "ind": "id", "msa": "ms", "may": "ms",
    # English names Whisper also accepts
    "burmese": "my", "valencian": "ca", "flemish": "nl", "haitian": "ht",
    "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa", "moldavian": "ro",
    "moldovan": "ro", "sinhalese": "si", "castilian": "es", "mandarin": "zh",
    "farsi": "fa", "filipino": "tl", "bangla": "bn",
    # names in the language itself
    "espanol": "es", "francais": "fr", "deutsch": "de", "italiano": "it",
    "portugues": "pt", "nederlands": "nl", "polski": "pl", "turkce": "tr",
    "svenska": "sv", "suomi": "fi", "dansk": "da", "norsk": "no",
    "cestina": "cs", "magyar": "hu", "romana": "ro", "catala": "ca",
    "bahasa indonesia": "id", "bahasa melayu": "ms", "tieng viet": "vi",
    "русский": "ru", "українська": "uk", "ελληνικά": "el", "עברית": "he",
    "العربية": "ar", "فارسی": "fa", "اردو": "ur", "हिन्दी": "hi",
    "हिंदी": "hi", "मराठी": "mr", "বাংলা": "bn", "தமிழ்": "ta",
    "తెలుగు": "te", "ಕನ್ನಡ": "kn", "മലയാളം": "ml", "ગુજરાતી": "gu",
    "ਪੰਜਾਬੀ": "pa", "नेपाली": "ne", "ไทย": "th", "中文": "zh",
    "汉语": "zh", "漢語": "zh", "普通话": "zh", "國語": "zh", "日本語": "ja",
    "한국어": "ko", "粵語": "yue", "粤语": "yue", "廣東話": "yue",
    "广东话": "yue",
}

_AUTO_WORDS = frozenset({"", "auto", "detect", "any", "none", "multi", "multilingual"})

_NAME_TO_CODE: Dict[str, str] = {}
for _code, _name in list(WHISPER_LANGUAGES.items()) + list(EXTRA_LANGUAGES.items()):
    _NAME_TO_CODE[_name] = _code

_CODE_SHAPE = re.compile(r"^[a-z]{2,3}$")

#: Vosk names its models with its own short codes; these are the ones that
#: differ from ISO 639-1 or carry a region.
VOSK_CODES: Dict[str, str] = {
    "en-us": "en", "en-in": "en", "en-gb": "en", "cn": "zh", "vn": "vi",
    "kz": "kk", "el-gr": "el", "tl-ph": "tl", "ar-tn": "ar", "fr-pguyot": "fr",
    "sv-rhasspy": "sv", "de-tuda": "de", "es-mx": "es", "pt-br": "pt",
    "ua": "uk", "jp": "ja", "kr": "ko", "cz": "cs", "gr": "el", "ir": "fa",
}

#: Languages Vosk publishes at least one model for (as ISO codes). Used only to
#: say whether suggesting Vosk makes sense; nothing is fetched.
VOSK_LANGUAGES: FrozenSet[str] = frozenset(
    "en zh ru fr de es pt el tr vi it nl ca ar fa tl uk kk sv ja eo hi cs pl "
    "uz ko br gu tg te ky".split()
)


def _fold(text: str) -> str:
    """Lower-case and strip accents from Latin text, leave other scripts alone."""
    lowered = unicodedata.normalize("NFC", text.strip().lower())
    decomposed = unicodedata.normalize("NFKD", lowered)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Stripping marks from Devanagari, Tamil or Thai destroys the word, so the
    # accent-free form is only used when the text was Latin to begin with.
    if all(ord(ch) < 0x250 for ch in folded):
        return re.sub(r"\s+", " ", folded)
    return re.sub(r"\s+", " ", lowered)


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Turn ``"en-US"``, ``"English"``, ``"हिन्दी"`` or ``None`` into a code.

    Returns ``None`` for auto-detection (``None``, ``""`` or ``"auto"``). An
    unrecognised two- or three-letter code is passed through lower-cased, so a
    custom engine can declare a language this table has never heard of. Any
    other unrecognised text raises ``ValueError``.
    """
    if language is None:
        return None
    if not isinstance(language, str):
        raise TypeError(
            "language must be a string such as 'en' or 'Hindi', or None to "
            "auto-detect; got {0}".format(type(language).__name__)
        )
    folded = _fold(language)
    if folded in _AUTO_WORDS:
        return None
    if folded in WHISPER_LANGUAGES or folded in EXTRA_LANGUAGES:
        return folded
    if folded in _ALIASES:
        return _ALIASES[folded]
    if folded in _NAME_TO_CODE:
        return _NAME_TO_CODE[folded]
    dashed = folded.replace("_", "-")
    if dashed in ("zh-yue", "zh-hk"):
        return "yue"
    # regional tags: en-US, pt_BR, zh-Hant-TW
    head = re.split(r"[-_ ]", folded, maxsplit=1)[0]
    if head != folded:
        if head in WHISPER_LANGUAGES or head in EXTRA_LANGUAGES:
            return head
        if head in _ALIASES:
            return _ALIASES[head]
    if _CODE_SHAPE.match(folded):
        return folded
    raise ValueError(
        "unknown language {0!r}; pass a code such as 'en' or 'hi', a name such "
        "as 'Hindi', or None to let the engine detect it".format(language)
    )


def language_name(code: Optional[str]) -> str:
    """``"hi"`` -> ``"Hindi"``; ``None`` -> ``"any language (auto-detect)"``."""
    if code is None:
        return "any language (auto-detect)"
    name = WHISPER_LANGUAGES.get(code) or EXTRA_LANGUAGES.get(code)
    if name is None:
        return code
    return " ".join(part.capitalize() for part in name.split(" "))


def whisper_tier(code: Optional[str]) -> str:
    """How well Whisper's training data covers a language.

    ``"strong"``, ``"moderate"``, ``"limited"`` or ``"unsupported"``. Auto-detect
    (``None``) is treated as ``"moderate"`` because the language is unknown.
    """
    if code is None:
        return "moderate"
    if code not in WHISPER_LANGUAGES:
        return "unsupported"
    if code in STRONG_LANGUAGES:
        return "strong"
    if code in MODERATE_LANGUAGES:
        return "moderate"
    return "limited"


def vosk_language_from_name(dirname: str) -> Optional[str]:
    """Read the language out of a Vosk model folder name.

    ``vosk-model-small-en-us-0.15`` -> ``"en"``; ``vosk-model-cn-0.22`` -> ``"zh"``.
    Returns ``None`` when the name does not say.
    """
    name = dirname.lower()
    if not name.startswith("vosk-model-"):
        # "my-model" must not be read as Burmese ("my"): only Vosk's own naming counts
        return None
    name = name[len("vosk-model-"):]
    name = re.sub(r"^(small|spk)-", "", name)
    parts = [p for p in re.split(r"[-_]", name) if p]
    if not parts:
        return None
    if len(parts) >= 2:
        pair = parts[0] + "-" + parts[1]
        if pair in VOSK_CODES:
            return VOSK_CODES[pair]
    head = parts[0]
    if head in VOSK_CODES:
        return VOSK_CODES[head]
    if head in WHISPER_LANGUAGES or head in EXTRA_LANGUAGES:
        return head
    return None


def all_whisper_codes(v3: bool = True) -> Tuple[str, ...]:
    """Every Whisper language code; ``v3=False`` leaves out the v3-only ones."""
    if v3:
        return tuple(WHISPER_LANGUAGES)
    return tuple(code for code in WHISPER_LANGUAGES if code not in V3_ONLY_LANGUAGES)
