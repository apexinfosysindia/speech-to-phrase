import io
import re
import shlex
import tempfile
from pathlib import Path

import pytest
from hassil import Intents

from speech_to_phrase.const import WordCasing
from speech_to_phrase.g2p import LexiconDatabase
from speech_to_phrase.hassil_fst import SPACE, G2PInfo, intents_to_fst

from . import SETTINGS

INTENTS_YAML = """
language: en
intents:
  GetTime:
    data:
      - sentences:
          - "what time is it"
          - "what is the time"
  TurnOn:
    data:
      - sentences:
          - "turn on [the] {name}"

  GetState:
    data:
      - sentences:
          - "what is the {name}'s state"

  Todo:
    data:
      - sentences:
          - "add {item} to todo"

  SetBrightness:
    data:
      - sentences:
          - "set brightness to {brightness} percent"

  ShoppingList:
    data:
      - sentences:
          - "add {food} to shopping list"

  SetColorWithFeatures:
    data:
      - sentences:
          - "set {name_with_features} to {color}"
        requires_context:
          domain: light
          light_supports_color: true

  SetBrightnessWithFeatures:
    data:
      - sentences:
          - "set {name_with_features} to {brightness} percent"
        requires_context:
          domain: light
          light_supports_brightness: true

  SetFanSpeedWithFeatures:
    data:
      - sentences:
          - "set {name_with_features} to {speed} percent"
        requires_context:
          domain: fan
          fan_supports_speed: true

  PauseMediaWithFeatures:
    data:
      - sentences:
          - "pause {name_with_features}"
        requires_context:
          domain: media_player
          media_player_supports_pause: true

  SetMediaVolumeWithFeatures:
    data:
      - sentences:
          - "set {name_with_features} to {volume} percent"
        requires_context:
          domain: media_player
          media_player_supports_volume_set: true

  NextMediaWithFeatures:
    data:
      - sentences:
          - "next track on {name_with_features}"
        requires_context:
          domain: media_player
          media_player_supports_next_track: true

  IsSmokeInArea:
    data:
      - sentences:
          - "(is there smoke;in the {area})"
lists:
  name:
    values:
      - tv
      - light
  area:
    values:
      - kitchen
      - living room
  item:
    wildcard: true
  brightness:
    range:
      from: 20
      to: 22
  food:
    values:
      - A1 Steak Sauce
      - NASA Moon Cake[s]
      - 0 A.D. DVD
  name_with_features:
    values:
      - in: On/Off Light
        context:
          domain: light
          light_supports_color: false
          light_supports_brightness: false
      - in: RGB Light
        context:
          domain: light
          light_supports_color: true
          light_supports_brightness: false
      - in: Brightness Light
        context:
          domain: light
          light_supports_color: false
          light_supports_brightness: true
      - in: Non-Speedy Fan
        context:
          domain: fan
          fan_supports_speed: false
      - in: Speedy Fan
        context:
          domain: fan
          fan_supports_speed: true
      - in: Dumb Speaker
        context:
          domain: media_player
          media_player_supports_pause: false
          media_player_supports_volume_set: false
          media_player_supports_next_track: false
      - in: Smart Speaker
        context:
          domain: media_player
          media_player_supports_pause: true
          media_player_supports_volume_set: true
          media_player_supports_next_track: true
  color:
    values:
      - red
      - green
      - blue
  speed:
    range:
      from: 20
      to: 22
  volume:
    range:
      from: 20
      to: 22
"""


def test_text_only() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"GetTime"})
    assert fst.words == {SPACE, "what", "time", "is", "it", "the"}

    assert set(tuple(t) for t in fst.to_tokens()) == {
        ("what", SPACE, "time", SPACE, "is", SPACE, "it"),
        ("what", SPACE, "is", SPACE, "the", SPACE, "time"),
    }
    assert set(fst.to_strings(False)) == {"what time is it", "what is the time"}

    fst_without_spaces = fst.remove_spaces()
    assert fst_without_spaces.words == {"what", "time", "is", "it", "the"}
    assert set(tuple(t) for t in fst_without_spaces.to_tokens()) == {
        ("what", "time", "is", "it"),
        ("what", "is", "the", "time"),
    }
    assert set(fst_without_spaces.to_strings(True)) == {
        "what time is it",
        "what is the time",
    }


def test_lists() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"TurnOn"})
    assert set(fst.to_strings(False)) == {
        "turn on tv",
        "turn on light",
        "turn on the tv",
        "turn on the light",
    }

    fst = intents_to_fst(intents, include_intents={"GetState"}).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "what is the tv's state",
        "what is the light's state",
    }

    fst = intents_to_fst(
        intents, number_language="en", include_intents={"SetBrightness"}
    ).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "set brightness to twenty percent",
        "set brightness to twenty one percent",
        "set brightness to twenty two percent",
    }


def test_prune() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"Todo"})

    # Wildcard branch is dead
    assert not fst.to_strings(False)
    assert not fst.to_tokens(only_connected=True)

    # Branch is still in FST
    assert fst.to_tokens(only_connected=False) == [["add", SPACE, "{item}"]]

    # Branch is pruned
    fst.prune()
    assert not fst.to_tokens(only_connected=False)


def test_g2p() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    lexicon = LexiconDatabase()
    lexicon.add("NASA", [["nah", "suh"]])

    fst = intents_to_fst(
        intents,
        include_intents={"ShoppingList"},
        number_language="en",
        g2p_info=G2PInfo(lexicon, WordCasing.get_function(WordCasing.LOWER)),
    ).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "add a one steak sauce to shopping list",
        "add nasa moon cake to shopping list",
        "add nasa moon cakes to shopping list",
        "add zero a d d v d to shopping list",
    }


def test_features() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    # light color
    fst = intents_to_fst(intents, include_intents={"SetColorWithFeatures"})
    assert set(fst.to_strings(False)) == {
        "set RGB Light to red",
        "set RGB Light to green",
        "set RGB Light to blue",
    }

    # light brightness
    fst = intents_to_fst(
        intents, include_intents={"SetBrightnessWithFeatures"}, number_language="en"
    )
    assert set(fst.to_strings(False)) == {
        "set Brightness Light to twenty percent",
        "set Brightness Light to twenty one percent",
        "set Brightness Light to twenty two percent",
    }

    # fan speed
    fst = intents_to_fst(
        intents, include_intents={"SetFanSpeedWithFeatures"}, number_language="en"
    )
    assert set(fst.to_strings(False)) == {
        "set Speedy Fan to twenty percent",
        "set Speedy Fan to twenty one percent",
        "set Speedy Fan to twenty two percent",
    }

    # media player pause
    fst = intents_to_fst(intents, include_intents={"PauseMediaWithFeatures"})
    assert set(fst.to_strings(False)) == {"pause Smart Speaker"}

    # media player volume
    fst = intents_to_fst(
        intents, include_intents={"SetMediaVolumeWithFeatures"}, number_language="en"
    )
    assert set(fst.to_strings(False)) == {
        "set Smart Speaker to twenty percent",
        "set Smart Speaker to twenty one percent",
        "set Smart Speaker to twenty two percent",
    }

    # media player next
    fst = intents_to_fst(intents, include_intents={"NextMediaWithFeatures"})
    assert set(fst.to_strings(False)) == {"next track on Smart Speaker"}


def test_permutations() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"IsSmokeInArea"})
    assert set(fst.to_strings(False)) == {
        "is there smoke in the kitchen",
        "is there smoke in the living room",
        "in the kitchen is there smoke",
        "in the living room is there smoke",
    }


@pytest.mark.asyncio
async def test_list_value_probabilities() -> None:
    """Test that normalizing intent level probabilities boost the probabilities
    of intents with fewer sentences."""
    intents_str = """
language: en
intents:
  GetTime:
    data:
      - sentences:
          - "what time is it"
  SetColor:
    data:
      - sentences:
          - "set color to {color}"

lists:
  color:
    values:
      - red
      - green
      - blue
    """

    with io.StringIO(intents_str) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, number_language="en").remove_spaces()
    tools = SETTINGS.tools

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        text_fst_path = temp_dir / "test.fst.txt"
        fst_path = temp_dir / "test.fst"
        ngram_fst_path = temp_dir / "test_ngram.fst"
        words_path = temp_dir / "words.txt"

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file, open(
            words_path, "w", encoding="utf-8"
        ) as words_file:
            fst.write(text_fst_file, words_file)

        await tools.async_run(
            "fstcompile",
            [
                shlex.quote(f"--isymbols={words_path}"),
                shlex.quote(f"--osymbols={words_path}"),
                "--keep_isymbols=true",
                "--keep_osymbols=true",
                shlex.quote(str(text_fst_path)),
                shlex.quote(str(fst_path)),
            ],
        )
        await tools.async_run_pipeline(
            ["ngramcount", "--order=3", shlex.quote(str(fst_path)), "-"],
            [
                "ngrammake",
                "--method=kneser_ney",
                "-",
                shlex.quote(str(ngram_fst_path)),
            ],
        )

        test_sentences = [
            "what time is it",
            "set color to red",
            "set color to green",
            "set color to blue",
        ]

        perplexities = {}
        for sentence in test_sentences:
            test_sentences_path = temp_dir / "test_sentences.txt"
            test_sentences_path.write_text(sentence, encoding="utf-8")
            test_archive_path = temp_dir / "test_sentences.far"

            await tools.async_run(
                "farcompilestrings",
                [
                    "--entry_type=file",
                    shlex.quote(f"--symbols={words_path}"),
                    shlex.quote(str(test_sentences_path)),
                    shlex.quote(str(test_archive_path)),
                ],
            )

            output = (
                await tools.async_run(
                    "ngramperplexity",
                    [
                        shlex.quote(str(ngram_fst_path)),
                        shlex.quote(str(test_archive_path)),
                    ],
                )
            ).decode("utf-8")
            perplexity_match = re.search(r"perplexity = ([0-9]+\.[0-9]+)", output)
            assert perplexity_match is not None
            perplexities[sentence] = float(perplexity_match.group(1))

        time_perplexity = perplexities["what time is it"]
        color_perlexities = {
            perplexities[s] for s in test_sentences if s.startswith("set color")
        }

        # set color sentences should all have equal perplexity
        assert len(color_perlexities) == 1
        color_perplexity = next(iter(color_perlexities))

        # Normalizing probabilities will reduce the ratio from about 1.9 to
        # about 1.39.
        #
        # This means that "what time is it" will not be considered so much rarer
        # than "set color to..." just because there are many colors.
        #
        # The perplexities are not identical though because all "set color"
        # sentences still share common subsequences. Even with no smoothing,
        # they will still not be identical because order=3.
        perplexity_ratio = time_perplexity / color_perplexity
        assert perplexity_ratio < 1.5, perplexity_ratio
