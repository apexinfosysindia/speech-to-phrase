"""Fast but limited speech-to-text."""

import importlib

from .const import Language, Settings, WordCasing
from .hass_api import Things
from .models import MODELS, Model
from .train import train
from .transcribe import transcribe

__version__ = importlib.metadata.version("speech-to-phrase")
__all__ = [
    "Language",
    "train",
    "transcribe",
    "MODELS",
    "Model",
    "WordCasing",
    "Things",
    "Settings",
    "__version__",
]
