"""Wyoming server."""

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Optional

from wyoming.server import AsyncServer

from . import __version__
from .const import Settings, State
from .event_handler import SpeechToPhraseEventHandler
from .apex_api import ApexOSInfo, get_apex_info
from .models import DEFAULT_MODEL, Model, get_models_for_languages
from .train import train

_LOGGER = logging.getLogger()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--train-dir", required=True, help="Directory to write trained model files"
    )
    parser.add_argument(
        "--tools-dir", required=True, help="Directory with kaldi, openfst, etc."
    )
    parser.add_argument(
        "--models-dir", required=True, help="Directory with speech models"
    )
    parser.add_argument(
        "--custom-sentences-dir",
        action="append",
        help="Directory with custom sentence directories for each language",
    )
    # ApexOS core API
    # NOTE: --hass-token/--hass-websocket-uri are kept as deprecated legacy
    # aliases for compatibility with existing wrappers and run scripts.
    parser.add_argument(
        "--apex-token",
        "--hass-token",
        dest="apex_token",
        required=True,
        help="Long-lived access token for the core API",
    )
    parser.add_argument(
        "--apex-websocket-uri",
        "--hass-websocket-uri",
        dest="apex_websocket_uri",
        default="ws://apexos.local:1702/api/websocket",
        help="URI of the core websocket API",
    )
    # Training
    parser.add_argument(
        "--retrain-on-start",
        action="store_true",
        help="Automatically retrain when starting",
    )
    parser.add_argument(
        "--retrain-seconds",
        type=float,
        help="Number of seconds to wait before automatically retraining",
    )
    parser.add_argument(
        "--retrain-on-connect",
        action="store_true",
        help="Automatically retrain on every client connection",
    )
    # Audio
    parser.add_argument("--volume-multiplier", type=float, default=1.0)
    #
    parser.add_argument(
        "--log-format",
        default="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
        help="Format for log messages",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    state = State(
        settings=Settings(
            models_dir=Path(args.models_dir),
            train_dir=Path(args.train_dir),
            tools_dir=Path(args.tools_dir),
            custom_sentences_dirs=args.custom_sentences_dir or [],
            apex_token=args.apex_token,
            apex_websocket_uri=args.apex_websocket_uri,
            retrain_on_connect=args.retrain_on_connect,
            volume_multiplier=args.volume_multiplier,
        )
    )

    if args.retrain_on_start:
        await _retrain_once(state, force_retrain=True)

    # Retrain on an interval
    retrain_task: Optional[asyncio.Task] = None
    if (args.retrain_seconds is not None) and (args.retrain_seconds > 0):
        retrain_task = asyncio.create_task(_retrain_loop(state, args.retrain_seconds))

    # Run server
    wyoming_server = AsyncServer.from_uri(args.uri)

    _LOGGER.info("Ready")

    try:
        await wyoming_server.run(partial(SpeechToPhraseEventHandler, state))
    except KeyboardInterrupt:
        pass
    finally:
        if retrain_task is not None:
            retrain_task.cancel()
            await retrain_task


async def _retrain_loop(state: State, wait_seconds: float) -> None:
    """Wait and retrain on a loop."""
    while True:
        await asyncio.sleep(wait_seconds)
        await _retrain_once(state)


async def _retrain_once(state: State, force_retrain: bool = False) -> None:
    """Retrain all models that match the core system language or a pipeline language."""
    settings = state.settings
    _LOGGER.debug(
        "Getting exposed things from the core API (%s)", settings.apex_websocket_uri
    )
    apex_info = await get_apex_info(
        token=settings.apex_token, uri=settings.apex_websocket_uri
    )
    _LOGGER.debug("Core system language: %s", apex_info.system_language)
    if apex_info.pipeline_languages:
        _LOGGER.debug("Core pipeline language(s): %s", apex_info.pipeline_languages)

    settings.default_language = apex_info.system_language
    things = apex_info.things
    _LOGGER.debug(
        "Got %s entities, %s area(s), %s floor(s), %s extra sentence(s)",
        len(things.entities),
        len(things.areas),
        len(things.floors),
        len(things.extra_sentences),
    )

    languages_to_train = list(apex_info.pipeline_languages) + [
        apex_info.system_language
    ]
    models_to_train = get_models_for_languages(languages_to_train)
    if not models_to_train:
        # Fall back to English model
        models_to_train = [DEFAULT_MODEL]

    async with state.model_train_tasks_lock:
        for model in models_to_train:
            if model.id in state.model_train_tasks:
                # Already training
                continue

            train_task = asyncio.create_task(
                _train_model(model, settings, apex_info, force_retrain=force_retrain)
            )
            state.model_train_tasks[model.id] = train_task
            train_task.add_done_callback(
                partial(
                    lambda _task, model_id: state.model_train_tasks.pop(model_id, None),
                    model.id,
                )
            )


async def _train_model(
    model: Model,
    settings: Settings,
    apex_info: ApexOSInfo,
    force_retrain: bool = False,
) -> None:
    try:
        await train(model, settings, apex_info.things, force_retrain=force_retrain)
    except Exception:
        _LOGGER.exception("Unexpected error while training %s", model.id)
        raise


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
