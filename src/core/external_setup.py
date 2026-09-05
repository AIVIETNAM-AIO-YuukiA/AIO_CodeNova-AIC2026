"""Auto-provision the external repos and weights the pipeline needs.

TransNetV2 and gipformer live outside the Python dependency tree (own repo,
own venv, weights that need converting). Each helper here is idempotent: it
checks for the artifact and only does work when it is missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess

from core.errors import CodeNovaError

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = REPO_ROOT / "external"

TRANSNETV2_REPO = "https://github.com/soCzech/TransNetV2.git"
TRANSNETV2_DIR = EXTERNAL_DIR / "TransNetV2"
TRANSNETV2_PYTORCH_DIR = TRANSNETV2_DIR / "inference-pytorch"
TRANSNETV2_WEIGHTS = TRANSNETV2_PYTORCH_DIR / "transnetv2-pytorch-weights.pth"
TRANSNETV2_TF_WEIGHTS = TRANSNETV2_DIR / "inference" / "transnetv2-weights"

GIPFORMER_REPO = "https://github.com/ggroup-ai-lab/gipformer.git"
GIPFORMER_DIR = EXTERNAL_DIR / "gipformer"
GIPFORMER_VENV_PYTHON = (
    GIPFORMER_DIR
    / ".venv"
    / ("Scripts" if os.name == "nt" else "bin")
    / ("python.exe" if os.name == "nt" else "python")
)
GIPFORMER_JSON_SCRIPT = GIPFORMER_DIR / "infer_json.py"

# Upstream ships infer_onnx.py, which prints human-readable text. The pipeline
# needs machine-readable output for a batch of segments, so this wrapper reuses
# upstream's model/recognizer helpers verbatim and only changes the output
# format. Written here so a fresh clone is immediately usable.
_INFER_JSON_SOURCE = '''#!/usr/bin/env python3
"""Gipformer ONNX inference emitting one JSON object per audio file.

Same model and sherpa-onnx setup as the upstream ``infer_onnx.py`` (whose
``download_model``/``read_audio``/``create_recognizer`` this reuses verbatim);
only the output format differs — stdout carries one JSON line per file so the
CodeNova pipeline can parse results instead of scraping human-readable text.

Usage:
    python infer_json.py --audio seg1.wav seg2.wav --quantize fp32 --num-threads 4
"""

import argparse
import json
import sys

from infer_onnx import create_recognizer, download_model, read_audio


def main():
    parser = argparse.ArgumentParser(description="Gipformer ONNX inference (JSON output)")
    parser.add_argument("--audio", type=str, nargs="+", required=True)
    parser.add_argument("--quantize", type=str, choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Segments decoded per sherpa-onnx call (caps peak memory)",
    )
    parser.add_argument(
        "--decoding-method",
        type=str,
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    args = parser.parse_args()

    try:
        model_paths = download_model(args.quantize)
        recognizer = create_recognizer(
            model_paths,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
        )
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1

    # Decode in small groups: sherpa-onnx parallelises across streams, but its
    # padded batch tensor is sized by the longest segment, so decoding many
    # 30s segments at once tries to allocate GBs and dies in onnxruntime.
    for start in range(0, len(args.audio), args.batch_size):
        batch = args.audio[start : start + args.batch_size]
        streams = []
        for audio_path in batch:
            try:
                samples, sample_rate = read_audio(audio_path)
            except Exception as exc:
                print(json.dumps({"error": f"read {audio_path}: {exc}"}), flush=True)
                return 1
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            streams.append(stream)

        try:
            recognizer.decode_streams(streams)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return 1

        for audio_path, stream in zip(batch, streams):
            print(
                json.dumps({"audio_path": audio_path, "text": stream.result.text.strip()}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# The conversion script only reads a SavedModel and writes a state_dict, so it
# runs on modern TF/torch despite upstream's 2019-era pins. Kept in a throwaway
# venv so those loose pins never touch the project's own cu128 torch.
_CONVERT_PACKAGES = ("tensorflow==2.16.*", "torch", "numpy")
_CONVERT_VENV_PYTHON = "python.exe" if os.name == "nt" else "python"


def ensure_transnetv2(weights_path: Path | None = None) -> Path:
    """Clone TransNetV2 and convert its weights to PyTorch if not already done."""
    target = weights_path or TRANSNETV2_WEIGHTS
    if target.exists():
        return target

    _require_command("git", "Install git to auto-download TransNetV2.")
    _require_command("uv", "Install uv to build the weight-conversion environment.")

    if not TRANSNETV2_DIR.exists():
        LOGGER.info("Cloning TransNetV2 into %s", TRANSNETV2_DIR)
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", TRANSNETV2_REPO, str(TRANSNETV2_DIR)])

    if not TRANSNETV2_TF_WEIGHTS.exists():
        raise CodeNovaError(
            f"TransNetV2 TensorFlow weights missing at {TRANSNETV2_TF_WEIGHTS}. "
            f"Delete {TRANSNETV2_DIR} and let it re-clone."
        )

    LOGGER.info("Converting TransNetV2 weights to PyTorch (one-time, a few minutes)")
    venv_dir = TRANSNETV2_DIR / ".convert-venv"
    python_bin = venv_dir / ("Scripts" if os.name == "nt" else "bin") / _CONVERT_VENV_PYTHON
    try:
        _run(["uv", "venv", "--python", "3.11", str(venv_dir)])
        _run(["uv", "pip", "install", "--python", str(python_bin), *_CONVERT_PACKAGES])
        _run(
            [str(python_bin), "convert_weights.py", "--tf_weights", str(TRANSNETV2_TF_WEIGHTS)],
            cwd=TRANSNETV2_PYTORCH_DIR,
        )
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)

    if not target.exists():
        raise CodeNovaError(f"Weight conversion finished but {target} is missing.")
    LOGGER.info("TransNetV2 weights ready at %s", target)
    return target


def ensure_gipformer() -> Path:
    """Clone gipformer and build its isolated venv if not already done.

    Returns the venv's Python, which ``modules/asr/gipformer.py`` shells out to.
    The ONNX model itself is pulled from Hugging Face on first transcription.
    """
    if GIPFORMER_VENV_PYTHON.exists() and GIPFORMER_JSON_SCRIPT.exists():
        return GIPFORMER_VENV_PYTHON

    _require_command("git", "Install git to auto-download gipformer.")
    _require_command("uv", "Install uv to build the gipformer environment.")

    if not GIPFORMER_DIR.exists():
        LOGGER.info("Cloning gipformer into %s", GIPFORMER_DIR)
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", GIPFORMER_REPO, str(GIPFORMER_DIR)])

    if not GIPFORMER_JSON_SCRIPT.exists():
        GIPFORMER_JSON_SCRIPT.write_text(_INFER_JSON_SOURCE, encoding="utf-8")

    if not GIPFORMER_VENV_PYTHON.exists():
        LOGGER.info("Building gipformer venv (one-time)")
        _run(["uv", "sync"], cwd=GIPFORMER_DIR)

    if not GIPFORMER_VENV_PYTHON.exists():
        raise CodeNovaError(f"gipformer setup finished but {GIPFORMER_VENV_PYTHON} is missing.")
    LOGGER.info("gipformer ready at %s", GIPFORMER_DIR)
    return GIPFORMER_VENV_PYTHON


def _require_command(name: str, hint: str) -> None:
    if shutil.which(name) is None:
        raise CodeNovaError(f"'{name}' not found on PATH. {hint}")


def _run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CodeNovaError(
            f"Command failed ({' '.join(command[:2])}): {result.stderr.strip()[:500]}"
        )
