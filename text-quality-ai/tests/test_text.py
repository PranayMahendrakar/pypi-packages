

def test_es_plurals_that_keep_their_syllable():
    """Regression: the silent-"es" rule fired on "-les", "-ches" and "-shes", which are
    pronounced. Every such plural lost a syllable, so Flesch and Flesch-Kincaid reported
    text as easier than it reads."""
    from text_quality_ai._text import syllables

    # "-es" is pronounced: after a sibilant, and after consonant + l where "-le" carries it
    for word in ("bottles", "candles", "tables", "apples", "watches", "dishes",
                 "wishes", "churches", "boxes", "pages", "faces", "buses"):
        assert syllables(word) == 2, f"{word} should be two syllables"

    # "-es" really is silent here, and must stay that way
    for word in ("names", "makes", "lives", "holes", "rules", "times"):
        assert syllables(word) == 1, f"{word} should be one syllable"


def test_syllable_counter_still_right_on_the_ordinary_cases():
    from text_quality_ai._text import syllables

    for word, expected in [("cat", 1), ("running", 2), ("beautiful", 3), ("area", 3),
                           ("science", 2), ("idea", 3), ("hoped", 1), ("wanted", 2),
                           ("the", 1), ("strengths", 1)]:
        assert syllables(word) == expected, f"{word} should be {expected}"
