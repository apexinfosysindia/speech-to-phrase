"""Validate sentence YAML files."""

from typing import Any, Dict, Optional

import pytest
import voluptuous as vol
from hassil import (
    Alternative,
    Expression,
    Group,
    ListReference,
    RuleReference,
    Sentence,
    parse_sentence,
)
from voluptuous.humanize import validate_with_humanized_errors

from speech_to_phrase.util import yaml

from . import SETTINGS, TEST_LANGUAGES, TESTS_DIR

PROGRAM_DIR = TESTS_DIR.parent
MODULE_DIR = PROGRAM_DIR / "speech_to_phrase"
SENTENCES_DIR = MODULE_DIR / "sentences"


def _visit_expression(
    e: Expression,
    visitor,
    visitor_arg: Any,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
):
    result = visitor(e, visitor_arg)
    if isinstance(e, Group):
        grp: Group = e
        for item in grp.items:
            _visit_expression(item, visitor, result, expansion_rules=expansion_rules)
    elif isinstance(e, RuleReference) and expansion_rules:
        rule_ref: RuleReference = e
        rule = expansion_rules.get(rule_ref.rule_name)
        if rule is not None:
            _visit_expression(
                rule.expression, visitor, visitor_arg, expansion_rules=expansion_rules
            )


def no_rule_references(sentence: str):
    """Validator that doesn't allow for <rule> references in a sentence template."""

    def visitor(e: Expression, arg: Any):
        if isinstance(e, RuleReference):
            rule_ref: RuleReference = e
            raise vol.Invalid(
                f"Rule references not allow in expansion rules ({{{rule_ref.rule_name}}})"
            )

    _visit_expression(parse_sentence(sentence).expression, visitor, None)
    return sentence


def not_optional(sentence: str):
    """Validator that ensures a sentence is not completely optional."""

    top_expression = parse_sentence(sentence).expression
    if isinstance(top_expression, Alternative):
        alt: Alternative = top_expression
        if alt.is_optional:
            raise vol.Invalid("Expansion rule must have some required text")

    return sentence


def no_alternative_list_references(sentence: str):
    """Validator that doesn't allow for {list} references in (an|alternative) or [an optional]."""

    def visitor(e: Expression, arg: Any):
        if isinstance(e, Alternative):
            return True

        in_alternative: bool = arg

        if isinstance(e, ListReference) and in_alternative:
            list_ref: ListReference = e
            raise vol.Invalid(
                f"List references not allow in alternatives (a|b) or optionals [c] ({{{list_ref.list_name}}})"
            )

        return in_alternative

    _visit_expression(parse_sentence(sentence).expression, visitor, False)
    return sentence


def get_slots(sentence: Sentence, expansion_rules: Dict[str, Sentence]) -> set[str]:
    """Get the used list/slot names in a sentence."""
    slot_names: set[str] = set()

    def visitor(e: Expression, arg: Any):
        if isinstance(e, ListReference):
            list_ref: ListReference = e
            slot_names.add(list_ref.slot_name)

    _visit_expression(
        sentence.expression, visitor, None, expansion_rules=expansion_rules
    )
    return slot_names


SHARED_LISTS_SCHEMA = vol.Schema(
    {
        # list name
        str: vol.Any(
            {
                vol.Required("range"): {
                    vol.Required("from"): int,
                    vol.Required("to"): int,
                    vol.Optional("step"): int,
                }
            },
            {
                vol.Required("multi_range"): [
                    {
                        vol.Required("from"): int,
                        vol.Required("to"): int,
                        vol.Optional("step"): int,
                    }
                ]
            },
        )
    }
)

SENTENCES_SCHEMA = vol.Schema(
    {
        vol.Required("language"): str,
        vol.Optional("lists"): {
            # list name
            str: [str],
        },
        vol.Optional("expansion_rules"): {
            # rule name
            str: vol.All(str, no_rule_references)
        },
        vol.Optional("wildcards"): [str],
        vol.Optional("transformations"): {
            # transform name
            str: [{vol.Required("outputs"): [str], vol.Optional("match"): str}]
        },
        vol.Optional("transformed_lists"): {
            # list name
            str: {"source": str, "transformations": [str]}
        },
        vol.Required("data"): [
            vol.Any(
                {
                    vol.Required("sentences"): [
                        vol.All(str, no_alternative_list_references)
                    ],
                    vol.Optional("domains"): [str],
                    vol.Optional("light_supports_color"): bool,
                    vol.Optional("light_supports_brightness"): bool,
                    vol.Optional("fan_supports_speed"): bool,
                    vol.Optional("cover_supports_position"): bool,
                    vol.Optional("media_player_supports_pause"): bool,
                    vol.Optional("media_player_supports_volume_set"): bool,
                    vol.Optional("media_player_supports_next_track"): bool,
                },
                vol.All(str, no_alternative_list_references),
            ),
        ],
    }
)

FIXTURES_SCHEMA = vol.Schema(
    {
        vol.Required("language"): str,
        "fixtures": {
            vol.Required("entities"): [
                {
                    vol.Required("name"): vol.Any(str, [str]),
                    vol.Required("domain"): str,
                    vol.Optional("light_supports_color"): bool,
                    vol.Optional("light_supports_brightness"): bool,
                    vol.Optional("fan_supports_speed"): bool,
                    vol.Optional("cover_supports_position"): bool,
                    vol.Optional("media_player_supports_pause"): bool,
                    vol.Optional("media_player_supports_volume_set"): bool,
                    vol.Optional("media_player_supports_next_track"): bool,
                }
            ],
            vol.Optional("floors"): [{vol.Required("name"): vol.Any(str, [str])}],
            vol.Optional("areas"): [{vol.Required("name"): vol.Any(str, [str])}],
        },
    }
)

TEST_SENTENCES_SCHEMA = vol.Schema(
    {
        vol.Required("language"): str,
        vol.Required("sentences"): [str],
        vol.Optional("failing_transcriptions"): [str],
    }
)


def test_validate_shared_lists() -> None:
    """Validate shared lists."""
    with open(SETTINGS.shared_lists_path, "r", encoding="utf-8") as shared_lists_file:
        shared_lists_dict = yaml.load(shared_lists_file)
        validate_with_humanized_errors(shared_lists_dict, SHARED_LISTS_SCHEMA)


@pytest.mark.parametrize("language", TEST_LANGUAGES)
def test_validate_sentences(language: str) -> None:
    """Test that sentence YAML files match expected schema."""
    lang_sentences_path = SENTENCES_DIR / f"{language}.yaml"
    assert (
        lang_sentences_path.exists()
    ), f"Missing sentences file for language '{language}' at '{lang_sentences_path}'"

    with open(lang_sentences_path, "r", encoding="utf-8") as lang_sentences_file:
        lang_sentences_dict = yaml.load(lang_sentences_file)
        validate_with_humanized_errors(lang_sentences_dict, SENTENCES_SCHEMA)

    assert lang_sentences_dict["language"] == language

    # Check list transformations
    tr_names = set(lang_sentences_dict.get("transformations", {}).keys())

    name_lists: set[str] = {"name"}
    for tr_list_name, tr_list_info in lang_sentences_dict.get(
        "transformed_lists", {}
    ).items():
        list_source = tr_list_info["source"]
        assert list_source in ("name", "area", "floor"), (
            "Transformed list source must be name/area/floor: "
            f"file={lang_sentences_path}"
        )

        if list_source == "name":
            name_lists.add(tr_list_name)

        trs = tr_list_info["transformations"]
        unknown_trs = set(trs) - tr_names
        assert not unknown_trs, (
            f"Undefined list transformation: {unknown_trs} ,"
            f"file={lang_sentences_path}"
        )

    expansion_rules = {
        rule_name: parse_sentence(rule_body)
        for rule_name, rule_body in lang_sentences_dict.get(
            "expansion_rules", {}
        ).items()
    }

    # Check that {name} is used appropriately
    for sentence_info in lang_sentences_dict["data"]:
        if isinstance(sentence_info, str):
            # Sentence template
            sentence = parse_sentence(sentence_info)
            assert name_lists.isdisjoint(get_slots(sentence, expansion_rules)), (
                "Sentence templates with {name} must be in a block with domains: "
                f"sentence={sentence_info}, "
                f"file={lang_sentences_path}"
            )
            continue

        assert sentence_info["domains"], (
            "At least one domain is required in a sentence block: "
            f"block={sentence_info}, "
            f"file={lang_sentences_path}"
        )

        for sentence_text in sentence_info["sentences"]:
            sentence = parse_sentence(sentence_text)
            assert not name_lists.isdisjoint(get_slots(sentence, expansion_rules)), (
                "Sentences in a block must contain {name}: "
                f"sentence={sentence_text}, "
                f"file={lang_sentences_path}"
            )


@pytest.mark.parametrize("language", TEST_LANGUAGES)
def test_validate_fixtures(language: str) -> None:
    """Test that fixture YAML files match expected schema."""
    lang_test_fixtures_path = TESTS_DIR / "fixtures" / f"{language}.yaml"
    assert (
        lang_test_fixtures_path.exists()
    ), f"Missing test fixtures file for language '{language}' at '{lang_test_fixtures_path}'"

    with open(
        lang_test_fixtures_path, "r", encoding="utf-8"
    ) as lang_test_fixtures_file:
        lang_test_fixtures_dict = yaml.load(lang_test_fixtures_file)
        validate_with_humanized_errors(lang_test_fixtures_dict, FIXTURES_SCHEMA)

    assert lang_test_fixtures_dict.get("language") == language


@pytest.mark.parametrize("language", TEST_LANGUAGES)
def test_validate_test_sentences(language: str) -> None:
    """Test that test sentences YAML files match expected schema."""
    lang_test_sentences_path = TESTS_DIR / "sentences" / f"{language}.yaml"
    assert (
        lang_test_sentences_path.exists()
    ), f"Missing test sentences file for language '{language}' at '{lang_test_sentences_path}'"

    with open(
        lang_test_sentences_path, "r", encoding="utf-8"
    ) as lang_test_sentences_file:
        lang_test_sentences_dict = yaml.load(lang_test_sentences_file)
        validate_with_humanized_errors(lang_test_sentences_dict, TEST_SENTENCES_SCHEMA)

    assert lang_test_sentences_dict.get("language") == language
