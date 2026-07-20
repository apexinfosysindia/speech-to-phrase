"""Tests."""

from pathlib import Path

from speech_to_phrase import Language, Settings

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent

LOCAL_DIR = ROOT_DIR / "local"
SETTINGS = Settings(
    models_dir=LOCAL_DIR / "models",
    train_dir=TESTS_DIR / "train",
    tools_dir=LOCAL_DIR,
    hass_token="",
    hass_websocket_uri="",
    retrain_on_connect=False,
    custom_sentences_dirs=[TESTS_DIR / "custom_sentences"],
)

TEST_LANGUAGES = [lang.value for lang in Language]
