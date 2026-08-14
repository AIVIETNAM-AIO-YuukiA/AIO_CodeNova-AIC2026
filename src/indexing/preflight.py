"""Read-only preflight inventory for an offline indexing run."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import uuid

from config.settings import Experiment
from core.errors import CodeNovaError
from core.errors import EmbeddingError
from modules.embedding import resolve_embedding_model
from video.discovery import VIDEO_EXTENSIONS


class PreflightError(CodeNovaError):
    """Raised when an indexing plan cannot be created or verified."""


def build_preflight_plan(experiment: Experiment, input_dir: Path, *, approved: bool) -> dict:
    if not input_dir.is_dir():
        raise PreflightError(f"Input directory does not exist: {input_dir}")
    entries: list[tuple[str, int, int]] = []
    extensions: dict[str, int] = {}
    for path in sorted(input_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in VIDEO_EXTENSIONS:
            continue
        stat = path.stat()
        entries.append((str(path.relative_to(input_dir)), stat.st_size, stat.st_mtime_ns))
        extensions[suffix] = extensions.get(suffix, 0) + 1
    if not entries:
        raise PreflightError(f"No supported videos found under {input_dir}")
    device = _device_info(experiment.config.device)
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()
    models = [
        _model_info(model, device["resolved"]) for model in experiment.config.embedding_models
    ]
    return {
        "plan_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}",
        "experiment": experiment.name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "APPROVED" if approved else "PENDING_APPROVAL",
        "config_hash": experiment.config.config_hash(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(experiment.config).items()
        },
        "device": device,
        "dataset": {
            "input_dir": str(input_dir.resolve()),
            "video_count": len(entries),
            "total_size_bytes": sum(size for _, size, _ in entries),
            "extensions": extensions,
            "fingerprint": sha256(payload).hexdigest(),
        },
        "models": models,
    }


def write_preflight_plan(experiment: Experiment, plan: dict) -> Path:
    directory = experiment.run_dir / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{plan['plan_id']}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".plan.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def verify_preflight_plan(experiment: Experiment, input_dir: Path, path: Path) -> dict:
    """Reject an unapproved or stale plan before discovery starts."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "APPROVED":
        raise PreflightError(f"Preflight plan is not approved: {path}")
    current = build_preflight_plan(experiment, input_dir, approved=True)
    if plan.get("config_hash") != current["config_hash"]:
        raise PreflightError("Pipeline config changed after preflight approval")
    expected_dataset = plan.get("dataset") or {}
    if expected_dataset.get("fingerprint") != current["dataset"]["fingerprint"]:
        raise PreflightError("Dataset changed after preflight approval")
    return plan


def _device_info(requested: str) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {
            "requested": requested,
            "resolved": "cpu",
            "cuda_available": False,
            "reason": "torch is not installed",
        }
    available = torch.cuda.is_available()
    if requested == "auto":
        resolved = "cuda:0" if available else "cpu"
    elif requested.startswith("cuda") and not available:
        raise PreflightError(f"CUDA requested ({requested}) but unavailable")
    else:
        resolved = requested
    info: dict[str, object] = {
        "requested": requested,
        "resolved": resolved,
        "cuda_available": available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if available else 0,
    }
    if resolved.startswith("cuda"):
        index = int(resolved.split(":", 1)[1]) if ":" in resolved else 0
        properties = torch.cuda.get_device_properties(index)
        info.update({"gpu_name": properties.name, "total_memory_bytes": properties.total_memory})
        try:
            free, _ = torch.cuda.mem_get_info(index)
            info["free_memory_bytes"] = free
        except Exception:
            info["free_memory_bytes"] = None
    return info


def _model_info(model_name: str, resolved_device: object) -> dict[str, object]:
    try:
        spec = resolve_embedding_model(model_name)
    except EmbeddingError as exc:
        raise PreflightError(str(exc)) from exc
    return {
        **spec.to_dict(),
        "name": spec.requested_name,
        "status": "CONFIGURED",
        "resolved_device": resolved_device,
    }
