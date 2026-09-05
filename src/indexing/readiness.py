"""Atomic readiness report persistence."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile

from config.settings import Experiment
from indexing.validation import ExperimentValidationReport


def write_readiness(
    experiment: Experiment, report: ExperimentValidationReport, *, config_hash: str | None = None
) -> Path:
    destination = experiment.run_dir / "readiness.json"
    payload = {
        **report.to_dict(),
        "generated_at": datetime.now(UTC).isoformat(),
        "config_hash": config_hash or experiment.config.config_hash(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=".readiness.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    try:
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_readiness(experiment: Experiment) -> dict[str, object]:
    path = experiment.run_dir / "readiness.json"
    if not path.exists():
        raise RuntimeError(f"Experiment has no readiness report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Readiness payload must be a JSON object")
    return payload
