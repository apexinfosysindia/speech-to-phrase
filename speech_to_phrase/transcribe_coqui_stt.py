"""Uses Coqui STT and OpenFST for decoding.

See:
- https://github.com/coqui-ai/STT
- https://arxiv.org/pdf/2206.14589
"""

import asyncio
import asyncio.subprocess
import itertools
import logging
import math
import shlex
import struct
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Dict, List, Optional, Union

from .const import BLANK, EPS, SPACE, Settings
from .hassil_fst import decode_meta
from .models import Model
from .speech_tools import SpeechTools

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PRUNE_THRESHOLD = 10
_DEFAULT_SENTENCE_PROB_THRESHOLD = 20


async def transcribe_coqui_stt(
    model: Model, settings: Settings, audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe text from an audio stream using Kaldi."""
    model_dir = (settings.models_dir / model.id).absolute()
    train_dir = (settings.train_dir / model.id).absolute()

    exe_path = settings.tools.tools_dir / "stt_onlyprobs"
    proc = await asyncio.create_subprocess_exec(
        str(exe_path),
        str(model_dir / "model.tflite"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    async for chunk in audio_stream:
        # Write chunk size (4 bytes), then chunk
        proc.stdin.write(struct.pack("I", len(chunk)))
        proc.stdin.write(chunk)
        await proc.stdin.drain()

    # Zero-length chunk signals end
    proc.stdin.write(struct.pack("I", 0))
    await proc.stdin.drain()

    line = (await proc.stdout.readline()).decode().strip()
    probs: List[List[float]] = []
    while line:
        probs.append([float(p) for p in line.split()])
        line = (await proc.stdout.readline()).decode().strip()

    # Clean up
    proc.terminate()
    await proc.wait()

    return await _decode_probs(
        probs,
        train_dir,
        settings.tools,
        sentence_prob_threshold=model.sentence_prob_threshold,
    )


async def _decode_probs(
    probs: List[List[float]],
    train_dir: Union[str, Path],
    tools: SpeechTools,
    prune_threshold: Optional[float] = None,
    sentence_prob_threshold: Optional[float] = None,
) -> str:
    if not probs:
        # Nothing to decode
        return ""

    if prune_threshold is None:
        prune_threshold = _DEFAULT_PRUNE_THRESHOLD

    if sentence_prob_threshold is None:
        sentence_prob_threshold = _DEFAULT_SENTENCE_PROB_THRESHOLD

    train_dir = Path(train_dir)

    tokens_txt = train_dir / "tokens_with_blank.txt"
    output_txt = train_dir / "output.txt"
    char2idx: Dict[str, int] = {}
    with open(tokens_txt, "r", encoding="utf-8") as words_file:
        for line in words_file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            label = parts[0]
            if label == EPS:
                continue

            char2idx[label] = int(parts[1])

    blank_id = char2idx[BLANK]
    idx2char = {i: c for c, i in char2idx.items()}

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logits_txt = temp_dir / "logits.fst.txt"
        with open(logits_txt, "w", encoding="utf-8") as logits_file:
            current = 0

            # Add space to the end and make it the most probable
            space_prob = 0.99
            nonspace_prob = ((1 - space_prob) / (len(probs[0]) - 1)) + 1e-9
            space_probs = [
                space_prob if c == SPACE else nonspace_prob for c in char2idx
            ]

            for current_probs in itertools.chain(probs, [space_probs]):
                for i, prob in enumerate(current_probs, start=1):
                    log_prob = -math.log(prob + 1e-9)
                    if (i == blank_id) or (i >= len(idx2char)):
                        c = BLANK
                    else:
                        c = idx2char[i]

                    print(current, current + 1, c, log_prob, file=logits_file)

                current += 1

            print(current, file=logits_file)

        # tokens -> chars -> words -> sentences
        tokens_txt = train_dir / "tokens_with_blank.txt"
        token2sen_fst = train_dir / "token2sen.fst"
        stdout = await tools.async_run_pipeline(
            [
                "fstcompile",
                shlex.quote(f"--isymbols={tokens_txt}"),
                shlex.quote(f"--osymbols={tokens_txt}"),
                "--acceptor",
                shlex.quote(str(logits_txt)),
            ],
            ["fstdeterminize"],
            ["fstminimize"],
            ["fstpush", "--push_weights"],
            ["fstarcsort", "--sort_type=olabel"],
            ["fstprune", f"--weight={prune_threshold}"],  # prune logits
            ["fstcompose", "-", shlex.quote(str(token2sen_fst))],
            ["fstshortestpath"],
            ["fstproject", "--project_type=output"],
            ["fstrmepsilon"],
            ["fsttopsort"],
            [
                "fstprint",
                shlex.quote(f"--isymbols={output_txt}"),
                shlex.quote(f"--osymbols={output_txt}"),
            ],
            # ["awk", "{print $4}"],  # output label
        )

    words: List[str] = []
    sentence_prob = 0.0
    for line in stdout.decode(encoding="utf-8").splitlines():
        line_parts = line.strip().split()
        if len(line_parts) < 4:
            continue

        word = line_parts[3]
        words.append(word)

        if len(line_parts) > 4:
            word_prob = float(line_parts[4])
            sentence_prob += word_prob

    norm_sentence_prob = sentence_prob / max(1, len(words))
    if norm_sentence_prob > sentence_prob_threshold:
        # Out of vocabulary
        return ""

    text = " ".join(words)
    return decode_meta(text)
