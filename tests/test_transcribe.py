"""Test transcribing and recognition for each language."""

import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
import regex as re
from hassil import Intents, recognize_best
from home_assistant_intents import get_intents
from pysilero_vad import SileroVoiceActivityDetector

from speech_to_phrase import MODELS, Model, Things, train, transcribe
from speech_to_phrase.audio import wav_audio_stream
from speech_to_phrase.util import get_language_family, yaml

from . import SETTINGS, TEST_LANGUAGES, TESTS_DIR

CONTEXT_AREA = "<context area>"
INTENT_CONTEXT = {"area": CONTEXT_AREA}


@dataclass
class Resources:
    """Resources for a language."""

    language: str
    intents: Intents
    test_things: Things
    model: Model
    wav_dir: Path
    vad: SileroVoiceActivityDetector
    failing_transcriptions: set[str]


@pytest_asyncio.fixture(name="lang_resources", params=TEST_LANGUAGES, scope="session")
async def lang_resources_fixture(request) -> Resources:
    """Load language resources and train STP model."""
    language = request.param
    lang_intents_dict = get_intents(language)
    if not lang_intents_dict:
        lang_intents_dict = get_intents(get_language_family(language))

    assert lang_intents_dict, f"No intents for language: {language}"

    with open(
        TESTS_DIR / "fixtures" / f"{language}.yaml", "r", encoding="utf-8"
    ) as fixtures_file:
        fixtures_dict = yaml.load(fixtures_file)

    assert fixtures_dict["language"] == language
    test_things = Things.from_dict(fixtures_dict["fixtures"])

    lang_lists = lang_intents_dict.setdefault("lists", {})
    lang_lists.update(test_things.to_lists_dict())

    with open(
        TESTS_DIR / "sentences" / f"{language}.yaml", "r", encoding="utf-8"
    ) as test_file:
        test_sentences_dict = yaml.load(test_file)

    # Train STP model
    model = MODELS[language]
    model_train_dir = SETTINGS.model_train_dir(model.id)
    if model_train_dir.exists():
        shutil.rmtree(model_train_dir)

    await train(model, SETTINGS, test_things)

    return Resources(
        language=language,
        intents=Intents.from_dict(lang_intents_dict),
        test_things=test_things,
        model=model,
        wav_dir=TESTS_DIR / "wav" / language,
        vad=SileroVoiceActivityDetector(),
        failing_transcriptions=set(
            test_sentences_dict.get("failing_transcriptions", [])
        ),
    )


async def do_transcribe_recognize(
    lang_resources: Resources, wav_path: Path, generated: bool
) -> None:
    """Test transcribing expected sentences."""
    expected_text = wav_path.stem
    if expected_text in lang_resources.failing_transcriptions:
        warnings.warn(UserWarning(f"Skipping {wav_path} because it's expected to fail"))
        return

    actual_text = await transcribe(
        lang_resources.model,
        SETTINGS,
        wav_audio_stream(wav_path, lang_resources.vad),
    )

    if wav_path.name.startswith("oov_"):
        # Out of vocabulary should produce empty transcript
        assert not actual_text, f"Expected no transcript for OOV: {wav_path}"
        return

    assert actual_text, f"Got empty transcript for: {wav_path}"

    if actual_text != expected_text:
        # Check that the result would be the same in Home Assistant
        error_info = (
            f"expected_text='{expected_text}', "
            f"actual_text='{actual_text}', "
            f"file={wav_path}"
        )
        actual_result = recognize_best(
            actual_text,
            lang_resources.intents,
            intent_context=INTENT_CONTEXT,
            best_slot_name="name",
        )
        assert actual_result is not None, f"Transcript does not match: {error_info}"

        # Remove context area
        if ("area" in actual_result.entities) and (
            actual_result.entities["area"].value == CONTEXT_AREA
        ):
            actual_result.entities.pop("area")

        expected_result = recognize_best(
            expected_text,
            lang_resources.intents,
            intent_context=INTENT_CONTEXT,
            best_slot_name="name",
        )
        assert expected_result is not None, f"Transcript does not match: {error_info}"

        # Remove context area
        if ("area" in expected_result.entities) and (
            expected_result.entities["area"].value == CONTEXT_AREA
        ):
            expected_result.entities.pop("area")

        assert (
            actual_result.intent.name == expected_result.intent.name
        ), f"Recognized intents do not match between Speech-to-Phrase and Home Assistant: {error_info}"

        # Only check entity names, not values. This is because some values are
        # normalized by the Home Assistant intents, such as "rideau" ->
        # "rideaux" in French. In the futurue, these normalizations should be
        # codified here so values can be checked too.
        assert (
            actual_result.entities.keys() == expected_result.entities.keys()
        ), f"Recognized entities do not match between Speech-to-Phrase and Home Assistant: {error_info}"


def gen_test(language: str, wav_path: Path, generated: bool) -> None:
    """Generate a test function for each test WAV per language."""

    @pytest.mark.parametrize("lang_resources", [language], indirect=True)
    @pytest.mark.asyncio
    async def test_func(lang_resources: Resources) -> None:
        await do_transcribe_recognize(lang_resources, wav_path, generated)

    text = wav_path.stem
    text_sanitized = text.lower()
    text_sanitized = re.sub(r"(?:\s+)|(?:[-]+)", "_", text_sanitized)
    text_sanitized = re.sub(r"[^\p{L}0-9_]", "", text_sanitized)

    if generated:
        gen = "gen_"
        # test_func = pytest.mark.generated(test_func)
    else:
        gen = ""

    test_func.__name__ = f"test_transcribe_{gen}{text_sanitized}"
    setattr(sys.modules[__name__], test_func.__name__, test_func)


def gen_tests() -> None:
    """Generate test functions for all languages."""
    for lang_dir in (TESTS_DIR / "wav").iterdir():
        if not lang_dir.is_dir():
            continue

        language = lang_dir.name
        if language not in TEST_LANGUAGES:
            continue

        for wav_path in lang_dir.glob("*.wav"):
            gen_test(language, wav_path, generated=False)

        for wav_path in (lang_dir / "generated").glob("*.wav"):
            gen_test(language, wav_path, generated=True)


gen_tests()
