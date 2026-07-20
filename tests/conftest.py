import pytest

from . import TEST_LANGUAGES


def pytest_addoption(parser):
    parser.addoption(
        "--language",
        action="append",
        help="Run tests for a specific language(s) only.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    languages = config.getoption("--language")
    if languages:
        TEST_LANGUAGES.clear()
        TEST_LANGUAGES.extend(languages)
