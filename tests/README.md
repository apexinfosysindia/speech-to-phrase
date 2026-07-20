# Speech-to-phrase tests

To run the tests, ensure you have the `dev` requirements installed (see `pyproject.toml`) or just `pip install -e .[dev]`
You will also require a copy of the [Rhasspy speech tools](https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main/tools) for your CPU architecture extracted to `../local` (so `../local/kaldi`, `../local/openfst`, etc. should exist). If you need to rebuild the tools, see the [build scripts](https://github.com/rhasspy/rhasspy-speech/tree/master/script).

Test sentences for each language are in a `test_{language}.py` file, with the corresponding WAV files in `wav/{language}/` directory. Each test sentence is also checked against the official [Home Assistant intents](https://github.com/home-assistant/intents/) to ensure it's supported. WAV files starting with `oov_` are "out of vocabulary" tests that should produce an empty text response (some are also fun to listen to 🙂).

Models are automatically downloaded to `../local/{model_id}` and training data is written to `train/{model_id}`.
