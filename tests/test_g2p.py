"""Tests for grapheme-to-phoneme (g2p) methods."""

from unicode_rbnf import RbnfEngine

from speech_to_phrase.g2p import LexiconDatabase, split_words


def test_split_words() -> None:
    lexicon = LexiconDatabase()
    engine = RbnfEngine.for_language("en")

    # Initialisms
    assert split_words("HVAC", lexicon, engine) == ["H", "V", "A", "C"]
    assert split_words("H.V.A.C.", lexicon, engine) == ["H", "V", "A", "C"]

    # Word + number
    assert split_words("PM2.5", lexicon, engine) == [
        "P",
        "M",
        ("two", "2.5"),
        ("point", None),
        ("five", None),
    ]

    # Dashes and underscores
    assert split_words("test_name-1", lexicon, engine) == ["test", "name", ("one", "1")]


def test_decimal_numbers() -> None:
    assert split_words("PM2.5", LexiconDatabase(), RbnfEngine.for_language("fr")) == [
        "P",
        "M",
        ("deux", "2.5"),
        ("virgule", None),
        ("cinq", None),
    ]
