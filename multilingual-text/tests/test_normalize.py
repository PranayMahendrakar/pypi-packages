"""normalize(): every option, every documented example, and idempotence."""
import itertools
import unicodedata

import pytest

from multilingual_text import CASE_MODES, FORMS, normalize

MESSY = (
    "  “Crème” — Brûlée… "
    "٤٢ and २२​, co-operate  "
)

# The option matrix below is run against every one of these, not just MESSY.
# Varying only the options and always normalising the same string is how the
# idempotence bug shipped: MESSY contains none of the seven characters that
# broke the guarantee, so all 200 cases walked one path through the tables.
#
# Six of the seven are Latin letters whose accent sits on a base Unicode
# refuses to decompose, so peeling the accent off uncovers a character the
# hand-written table still has to map.  The seventh is Greek punctuation whose
# canonical form is itself foldable.
UNDECOMPOSABLE_WITH_ACCENT = "Ǣǣ Ǽǽ Ǿǿ Ǽsir Øre"
GREEK_WITH_PUNCTUATION = "ναι· οχι· ίσως;"

IDEMPOTENCE_CORPUS = [
    MESSY,
    UNDECOMPOSABLE_WITH_ACCENT,
    GREEK_WITH_PUNCTUATION,
    "Crème Brûlée — “naïve” café… ٤٢",
    "नमस्ते दुनिया, مرحبا, Привет мир",
]

#: The six letters whose accent hides an undecomposable base, and what
#: ``diacritics=True`` has to make of them in a single pass.
UNDECOMPOSABLE_ACCENTED = {
    "Ǣ": "AE", "ǣ": "ae",   # AE / ae with macron
    "Ǽ": "AE", "ǽ": "ae",   # AE / ae with acute
    "Ǿ": "O", "ǿ": "o",     # O / o with stroke and acute
}


def test_empty_string_stays_empty():
    assert normalize("") == ""


def test_defaults_fold_quotes_dashes_and_whitespace():
    assert normalize("  “Smart” quotes — and hard spaces  ") == (
        '"Smart" quotes - and hard spaces'
    )


def test_form_composes_a_decomposed_accent():
    decomposed = "e\u0301cole"          # e + COMBINING ACUTE ACCENT
    assert len(decomposed) == 6
    assert normalize(decomposed) == "\u00e9cole"
    assert len(normalize(decomposed)) == 5


def test_form_none_leaves_the_encoding_alone():
    decomposed = "e\u0301cole"
    assert normalize(decomposed, form=None) == decomposed


def test_form_nfd_decomposes():
    assert normalize("\u00e9cole", form="NFD") == "e\u0301cole"


def test_form_nfkc_folds_compatibility_characters():
    assert normalize("ＡＢ", form="NFKC") == "AB"
    assert normalize("ﬁne", form="NFKC") == "fine"


def test_bad_form_is_a_clear_value_error():
    with pytest.raises(ValueError) as excinfo:
        normalize("x", form="NFX")
    assert "NFC" in str(excinfo.value)


def test_bad_case_is_a_clear_value_error():
    with pytest.raises(ValueError) as excinfo:
        normalize("x", case="sentence")
    assert "fold" in str(excinfo.value)


def test_normalize_rejects_non_strings():
    with pytest.raises(TypeError):
        normalize(42)


@pytest.mark.parametrize(
    "case,expected",
    [("lower", "straße"), ("upper", "STRASSE"), ("fold", "strasse"),
     ("title", "Straße")],
)
def test_case_modes(case, expected):
    assert normalize("Straße", case=case) == expected


def test_case_none_leaves_the_case_alone():
    assert normalize("MiXeD Case") == "MiXeD Case"


def test_whitespace_collapses_runs_and_exotic_spaces():
    text = "a  b\t\tc  \n\n\n  d"
    assert normalize(text) == "a b c\n\nd"


def test_whitespace_removes_zero_width_characters():
    text = "A\u200bB\u00adC\ufeffD"   # zero width space, soft hyphen, BOM
    assert normalize(text) == "ABCD"


def test_whitespace_turns_a_no_break_space_into_an_ordinary_one():
    assert normalize("a\u00a0b") == "a b"


def test_whitespace_false_keeps_the_layout():
    assert normalize("  a  b  ", whitespace=False) == "  a  b  "


def test_punctuation_removal():
    assert normalize("Hello, world!", punctuation=True) == "Hello world"


def test_punctuation_removal_joins_hyphenated_words_as_documented():
    assert normalize("co-operate", punctuation=True) == "cooperate"


def test_punctuation_is_kept_by_default():
    assert normalize("Hello, world!") == "Hello, world!"


def test_digits_converts_arabic_indic_and_devanagari():
    assert normalize("٤٢ और २२", digits=True) == (
        "42 और 22"
    )


def test_digits_converts_every_unicode_decimal():
    assert normalize("৪২ ௧௨", digits=True) == "42 12"


def test_digits_leaves_ascii_alone_and_is_off_by_default():
    assert normalize("42", digits=True) == "42"
    assert normalize("٤٢") == "٤٢"


def test_quotes_folds_smart_punctuation():
    assert normalize("“a” – b…") == '"a" - b...'
    assert normalize("‘x’ «y»") == "'x' \"y\""


def test_quotes_false_keeps_the_typography():
    assert normalize("“a”", quotes=False) == "“a”"


def test_diacritics_strips_latin_accents():
    assert normalize("Crème Brûlée", diacritics=True) == "Creme Brulee"


def test_diacritics_handles_letters_unicode_cannot_decompose():
    assert normalize("ø ł đ æ œ", diacritics=True) == (
        "o l d ae oe"
    )


def test_diacritics_leaves_devanagari_vowel_signs_alone():
    word = "नमस्ते"  # namaste
    assert normalize(word, diacritics=True) == word


def test_diacritics_leaves_arabic_vowel_points_alone():
    pointed = "مَرْحَبًا"
    assert normalize(pointed, diacritics=True) == pointed


def test_diacritics_strips_greek_and_cyrillic_accents():
    assert normalize("Άθήνα", diacritics=True) == (
        "Αθηνα"
    )


def test_diacritics_is_off_by_default():
    assert normalize("café") == "café"


def test_idempotent_on_the_defaults():
    once = normalize(MESSY)
    assert normalize(once) == once


@pytest.mark.parametrize("text", IDEMPOTENCE_CORPUS)
@pytest.mark.parametrize(
    "form,case,punctuation,digits,diacritics",
    list(
        itertools.product(
            list(FORMS) + [None], list(CASE_MODES) + [None], [False, True],
            [False, True], [False, True],
        )
    ),
)
def test_idempotent_for_every_option_combination(
    text, form, case, punctuation, digits, diacritics
):
    kwargs = dict(
        form=form, case=case, punctuation=punctuation, digits=digits,
        diacritics=diacritics,
    )
    once = normalize(text, **kwargs)
    assert normalize(once, **kwargs) == once, (text, kwargs)


@pytest.mark.parametrize(
    "char,expected", sorted(UNDECOMPOSABLE_ACCENTED.items())
)
def test_accent_over_an_undecomposable_base_resolves_in_one_pass(char, expected):
    """Regression: the accent strip has to run *before* the manual table.

    U+01FE LATIN CAPITAL LETTER O WITH STROKE AND ACUTE decomposes to U+00D8
    plus an acute, so the O-with-stroke only appears once the accent has been
    peeled away.  With the table applied first it ran too early to see it: the
    first call returned an O-with-stroke and a second call turned it into "O".
    """
    once = normalize(char, diacritics=True)
    assert once == expected
    assert normalize(once, diacritics=True) == once


def test_greek_ano_teleia_is_folded_on_the_first_pass():
    """Regression: the same failure, with no options passed at all.

    U+0387 GREEK ANO TELEIA is canonically U+00B7 MIDDLE DOT, which the quote
    fold maps to ".".  The fold used to run only *before* the form pass, so the
    first call left the ano teleia standing and the second folded it.
    """
    once = normalize("ναι· οχι")
    assert once == "ναι. οχι"
    assert normalize(once) == once


def test_the_two_spellings_of_aesir_give_one_dedup_key():
    """The documented use case: a cache key must not depend on the spelling."""
    accented = normalize("Ǽsir", case="fold", diacritics=True)
    plain = normalize("Æsir", case="fold", diacritics=True)
    assert accented == plain == "aesir"


@pytest.mark.parametrize(
    "char", sorted(UNDECOMPOSABLE_ACCENTED) + ["·"]
)
def test_every_codepoint_that_broke_idempotence_is_stable_everywhere(char):
    """All seven, against all 800 combinations of all seven options."""
    for form, case, punctuation, digits, diacritics, quotes, whitespace in (
        itertools.product(
            list(FORMS) + [None], list(CASE_MODES) + [None],
            [False, True], [False, True], [False, True], [False, True],
            [False, True],
        )
    ):
        kwargs = dict(
            form=form, case=case, punctuation=punctuation, digits=digits,
            diacritics=diacritics, quotes=quotes, whitespace=whitespace,
        )
        once = normalize(char, **kwargs)
        assert normalize(once, **kwargs) == once, (char, kwargs)


# Characters whose compatibility decomposition uncovers something the earlier
# stages would otherwise never see: a caron that only appears once NFKC has
# split the digraph, a case distinction that only appears once the squared
# glyph has become "hPa", a ligature that only becomes two letters under NFKC.
NASTY = "Straße ẞ İstanbul i̇ ﬁsh Ⅷ ⅷ ㍱ ǰ ǅ ǆ Ǆ Ǳ ǲ ＡＢ ﾟﾞ ① ㎙"


@pytest.mark.parametrize(
    "form,case,diacritics",
    list(
        itertools.product(
            list(FORMS) + [None], list(CASE_MODES) + [None], [False, True]
        )
    ),
)
def test_idempotent_on_characters_that_compatibility_forms_expand(
    form, case, diacritics
):
    """A second call must not find accents the first call did not.

    Regression: with ``form`` applied after ``diacritics``, NFKC turned
    ``ǅ`` into "D" plus a z-with-caron *after* the accent stripper had
    already run, so the first call returned "dž"-ish text and the second
    stripped it -- two different answers for the same input.
    """
    kwargs = dict(form=form, case=case, diacritics=diacritics)
    once = normalize(NASTY, **kwargs)
    assert normalize(once, **kwargs) == once, kwargs


def test_compatibility_expansion_is_case_folded_and_stripped_in_one_pass():
    assert normalize("ǅ", form="NFKC", case="lower", diacritics=True) == "dz"
    assert normalize("㍱", form="NFKC", case="lower") == "hpa"
    assert normalize("Ⅷ", form="NFKC", case="lower") == "viii"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ascii",
        "Crème Brûlée",
        "नमस्ते दुनिया",
        "Привет, мир!",
        "こんにちは世界",
        "مرحبا بالعالم",
        "  spaces everywhere  ",
    ],
)
def test_normalize_normalize_equals_normalize(text):
    assert normalize(normalize(text)) == normalize(text)


def test_output_is_always_in_the_requested_form():
    out = normalize(MESSY, form="NFD")
    assert unicodedata.is_normalized("NFD", out)
    out = normalize(MESSY, form="NFC")
    assert unicodedata.is_normalized("NFC", out)


def test_unicode_text_survives_the_defaults():
    for text in ["你好世界", "مرحبا",
                 "สวัสดี"]:
        assert normalize(text) == text
