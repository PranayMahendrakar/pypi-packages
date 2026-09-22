"""Number, date, quantity and name extraction, and the text primitives under it."""
import pytest

from hallucination_check._facts import collect_keys, extract_facts, mask_numbers, number_tokens
from hallucination_check._text import (
    coverage,
    normalize,
    split_units,
    stem,
    tokenize,
    truncate,
    vectorize,
)


def kinds(text):
    return [(f.kind, f.text) for f in extract_facts(text)]


def keys_of(text):
    out = set()
    for fact in extract_facts(text):
        out.update(fact.keys)
    return out


def test_plain_numbers_and_thousands_separators():
    assert "num:1200" in keys_of("The trial enrolled 1,200 patients.")
    assert "num:1200" in keys_of("The trial enrolled 1200 patients.")


def test_percentages_and_currency_scales():
    assert {"num:12", "pct:12"} <= keys_of("Revenue rose 12% last year.")
    assert "num:12" in keys_of("Revenue rose 12 percent last year.")
    assert "num:4300000000" in keys_of("Revenue reached $4.3 billion.")
    assert "num:4300000000" in keys_of("Revenue reached 4.3bn.")


def test_numbers_written_in_words():
    assert "num:23" in keys_of("Twenty-three sites closed.")
    assert "num:105" in keys_of("The fleet grew to one hundred and five aircraft.")
    assert "num:2000000" in keys_of("Two million people attended.")


def test_a_bare_one_is_not_treated_as_a_quantity():
    assert not [f for f in extract_facts("One of the sites closed.") if f.kind == "number"]
    assert "num:1" in {k for f in extract_facts("One of the sites closed.", for_source=True) for k in f.keys}


def test_ordinals_are_not_flaggable_but_still_back_a_source_up():
    assert not [f for f in extract_facts("The third site closed.") if f.kind == "number"]
    source = extract_facts("The third site closed.", for_source=True)
    assert any(f.kind == "ordinal" for f in source)


def test_dates_in_every_common_shape():
    assert "date:2024-03-12" in keys_of("The report landed on 12 March 2024.")
    assert "date:2024-03-12" in keys_of("The report landed on March 12, 2024.")
    assert "date:2024-03-12" in keys_of("The report landed on 2024-03-12.")
    assert "date:2022-11" in keys_of("NASA launched Artemis in November 2022.")
    slashes = keys_of("The report landed on 03/12/2024.")
    assert "date:2024-12-03" in slashes and "date:2024-03-12" in slashes


def test_a_date_matches_when_only_its_parts_are_in_the_source():
    fact = [f for f in extract_facts("It happened on 12 March 2024.") if f.kind == "date"][0]
    assert not fact.is_in({"num:2024"})
    assert fact.is_in(collect_keys("March 2024, on the 12th day."))


def test_names_and_acronyms():
    assert ("name", "NASA") in kinds("NASA launched Artemis in November 2022.")
    assert ("name", "Artemis") in kinds("NASA launched Artemis in November 2022.")
    assert ("name", "Paris") in kinds("The tower is in Paris.")


def test_a_sentence_initial_common_word_is_not_a_name():
    assert not [f for f in extract_facts("Cats sleep a lot.") if f.kind == "name"]
    assert [f for f in extract_facts("Eiffel Tower is tall.") if f.kind == "name"]


def test_number_tokens_bridge_words_and_digits():
    assert "23" in number_tokens("twenty-three sites")
    assert "23" in number_tokens("23 sites")


def test_collect_keys_covers_tokens_and_stems():
    keys = collect_keys("The sites closed in 2019.")
    assert "name:sites" in keys and "name:site" in keys and "num:2019" in keys


def test_mask_numbers_keeps_the_sentence_shape():
    assert mask_numbers("The trial enrolled 4,200 patients.") == "The trial enrolled # patients."


def test_tokenizer_handles_unicode_and_cjk():
    assert tokenize("Café naïve") == ["café", "naïve"]
    assert tokenize("北京") == ["北", "京"]
    assert tokenize("It cost 1,200 euros") == ["it", "cost", "1,200", "euros"]


def test_stemming_is_light():
    assert stem("sites") == "site"
    assert stem("studies") == "study"
    assert stem("business") == "business"
    assert stem("reported") == "report"


def test_normalize_and_truncate():
    assert normalize("  a\n b  ") == "a b"
    assert truncate("x" * 200).endswith("...")
    assert len(truncate("x" * 200)) == 100
    assert truncate("short") == "short"


def test_coverage_is_a_share_of_the_claim_vocabulary():
    claim = vectorize("cats sleep")
    assert coverage(claim, vectorize("cats sleep a lot")) == 1.0
    assert coverage(claim, vectorize("dogs run")) == 0.0
    assert coverage(vectorize("the"), vectorize("the")) == 1.0


def test_split_units_rejects_an_unknown_granularity():
    with pytest.raises(ValueError, match="granularity must be"):
        split_units("text", "token")


def test_sentence_splitting_keeps_decimals_and_initials_together():
    assert split_units("The rate was 3.5 percent in May.", "sentence") == [
        "The rate was 3.5 percent in May."
    ]
    assert len(split_units("J. R. Smith wrote it. He was right.", "sentence")) == 2


def test_line_breaks_end_a_unit():
    assert split_units("- one item\n- two item", "sentence") == ["- one item", "- two item"]


def test_cjk_sentence_after_a_full_stop_starts_a_new_unit():
    assert len(split_units("Une hausse de 12 %. 李雷 a confirmé.", "sentence")) == 2
