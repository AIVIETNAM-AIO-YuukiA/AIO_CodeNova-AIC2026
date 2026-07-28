"""GPU detection for choosing the agent's Docker serving engine.

The agent model is fixed at Qwen3.5-4B (4-bit); only the ENGINE differs by
hardware, since GB10 and other NVIDIA GPUs need different servers:

- DGX Spark / GB10: vLLM, AWQ 4-bit (Marlin kernel — GB10's SM121 has no
  CUTLASS support yet, so CUTLASS MoE/scaled-mm kernels are disabled via
  ``VLLM_DISABLED_KERNELS``).
- Any other NVIDIA GPU: llama.cpp server — reads GGUF, the portable choice.

Both are started via ``make agent-up`` (docker-compose profiles
``vllm-agent`` / ``llamacpp-agent``) and serve OpenAI-compatible ``/v1`` on
port 8884 — the Python side never loads model weights itself.
"""

from __future__ import annotations

import logging
import os
import subprocess

LOGGER = logging.getLogger(__name__)

_VLLM_DEFAULT_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
_LLAMACPP_DEFAULT_MODEL = "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"


def gpu_name() -> str | None:
    """Return the first NVIDIA GPU's name via nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return first_line or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def is_dgx_spark() -> bool:
    """True if the first NVIDIA GPU is a DGX Spark (GB10)."""
    name = gpu_name()
    return bool(name and "GB10" in name)


def default_agent_model() -> str:
    """Return the model name the running Docker engine serves.

    ``AGENT_LOCAL_ENGINE_MODEL`` overrides; otherwise the engine picked by
    hardware determines the default (vLLM AWQ on GB10, llama.cpp GGUF
    elsewhere) — mirroring which docker-compose profile ``make agent-up``
    starts on this machine.
    """
    override = os.environ.get("AGENT_LOCAL_ENGINE_MODEL")
    if override:
        return override
    return _VLLM_DEFAULT_MODEL if is_dgx_spark() else _LLAMACPP_DEFAULT_MODEL
