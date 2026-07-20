"""Test for model utilities."""

from speech_to_phrase.models import get_models_for_languages


def test_get_models_for_languages() -> None:
    for language in ("en", "en_US", "en_GB"):
        models = get_models_for_languages({language})
        assert len(models) == 1
        assert models[0].id == "en_US-rhasspy"

    for language in ("fr", "fr_FR", "fr_CA"):
        models = get_models_for_languages({language})
        assert len(models) == 1
        assert models[0].id == "fr_FR-rhasspy"

    models = get_models_for_languages({"en", "fr_FR"})
    assert len(models) == 2
    assert {m.id for m in models} == {"en_US-rhasspy", "fr_FR-rhasspy"}
