"""Model transcription."""

import argparse
import asyncio
import logging
import wave
from collections.abc import AsyncIterable
from pathlib import Path

from .const import Settings, TranscribingError
from .models import MODELS, Model, ModelType
from .transcribe_coqui_stt import transcribe_coqui_stt
from .transcribe_kaldi import transcribe_kaldi

_LOGGER = logging.getLogger(__name__)


async def transcribe(
    model: Model, settings: Settings, audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe text from an audio stream."""
    if model.type == ModelType.KALDI:
        return await transcribe_kaldi(model, settings, audio_stream)

    if model.type == ModelType.COQUI_STT:
        return await transcribe_coqui_stt(model, settings, audio_stream)

    raise TranscribingError(f"Unexpected model type for {model.id}: {model.type}")


# -----------------------------------------------------------------------------


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, nargs="+", help="Path to WAV file")
    parser.add_argument(
        "--model", required=True, help="Id of speech model (e.g., en_US-rhasspy)"
    )
    parser.add_argument(
        "--train-dir", required=True, help="Directory with trained model files"
    )
    parser.add_argument(
        "--tools-dir", required=True, help="Directory with kaldi, openfst, etc."
    )
    parser.add_argument(
        "--models-dir", required=True, help="Directory with speech models"
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    model = next(iter(m for m in MODELS.values() if m.id == args.model), None)
    assert model is not None, f"Unknown model id: {args.model}"

    settings = Settings(
        models_dir=Path(args.models_dir),
        train_dir=Path(args.train_dir),
        tools_dir=Path(args.tools_dir),
        custom_sentences_dirs=[],
        hass_token="",
        hass_websocket_uri="",
        retrain_on_connect=False,
    )

    for wav_path in args.wav:
        with wave.open(wav_path, "rb") as wav_file:

            async def audio_stream(f: wave.Wave_read) -> AsyncIterable[bytes]:
                while True:
                    chunk = f.readframes(args.chunk_size)
                    if not chunk:
                        break

                    yield chunk

            text = await transcribe(model, settings, audio_stream(wav_file))
            _LOGGER.info("%s: %s", wav_path, text)


if __name__ == "__main__":
    asyncio.run(main())
