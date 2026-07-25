"""GPU detection for the agent's local-model fallback chain.

create_agent() (react.py) tries Gemini first (cloud, no local GPU needed);
when GEMINI_API_KEY isn't set, it falls back to a self-hosted engine chosen
by hardware, since GB10 and other NVIDIA GPUs need different engines:

- DGX Spark / GB10: Atlas (avarok/atlas-gb10) — a from-scratch Rust/CUDA
  engine hand-tuned for GB10's SM121 kernels, ~3x vLLM's decode throughput
  on this hardware per Atlas's own benchmarks. It reads HF checkpoints in
  its own quant formats (NVFP4/FP8/BF16), NOT GGUF or AWQ.
- Any other NVIDIA GPU: llama.cpp server — reads GGUF, the portable choice
  when Atlas's GB10-only kernels don't apply.

Within either engine, model size (9B vs 4B) is picked by free VRAM, since
both a 9B and 4B GGUF/NVFP4 checkpoint exist for this project's models.
"""

from __future__ import annotations

import logging
import os
import subprocess

LOGGER = logging.getLogger(__name__)

_DEFAULT_VRAM_THRESHOLD_GB = 12.0


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


def free_vram_gb() -> float | None:
    """Return free VRAM in GB on the first NVIDIA GPU, or None if unavailable.

    GB10 shares unified LPDDR5x memory with the system instead of discrete
    VRAM, so nvidia-smi reports "Memory-Usage: Not Supported" / "[N/A]" for
    memory.free there (confirmed on this machine) — fall back to
    /proc/meminfo's MemAvailable in that case, since that pool IS the GPU
    memory on this hardware.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if first_line and first_line.replace(".", "", 1).isdigit():
                return float(first_line) / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if is_dgx_spark():
        return _meminfo_available_gb()
    return None


def _meminfo_available_gb() -> float | None:
    """Return MemAvailable from /proc/meminfo in GB, or None if unreadable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        pass
    return None


def has_enough_vram_for_9b() -> bool:
    """True if free VRAM meets AGENT_VRAM_THRESHOLD_GB (default 12GB) for the 9B model.

    Below the threshold, callers should use the smaller 4B model instead.
    Unknown/no GPU counts as "not enough" — the local-engine fallback chain
    only runs when a GPU was already detected, so this should rarely fire.
    """
    threshold = float(os.environ.get("AGENT_VRAM_THRESHOLD_GB", _DEFAULT_VRAM_THRESHOLD_GB))
    free = free_vram_gb()
    if free is None:
        LOGGER.warning("Could not read free VRAM; assuming not enough for the 9B model.")
        return False
    return free >= threshold
