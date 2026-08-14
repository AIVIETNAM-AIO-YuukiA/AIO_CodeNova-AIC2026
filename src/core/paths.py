"""Utilities for deterministic experiment-owned frame paths."""

from dataclasses import dataclass
from pathlib import Path

from config.settings import Experiment
from core.errors import FramePathError

_PROJECT_ROOT: Path | None = None


def set_project_root(root: Path) -> None:
    """Set the global project root for resolving relative paths."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = root.resolve()


def resolve_frame_path(frame_path: str | Path) -> Path:
    """Resolve a frame path that may be relative to the project root.

    Args:
        frame_path: Absolute or relative path to a frame.

    Returns:
        Resolved absolute Path object. If the file doesn't exist at the
        resolved location, returns the original path (so standard
        FileNotFoundError can be raised downstream).
    """
    # Normalize backslashes (Windows) to forward slashes (Linux/POSIX)
    # otherwise Path() on Linux treats the whole string as a single filename.
    normalized_path = str(frame_path).replace("\\", "/")
    p = Path(normalized_path)
    if p.is_absolute() and p.exists():
        return p
    if _PROJECT_ROOT:
        resolved = (_PROJECT_ROOT / p).resolve()
        if resolved.exists():
            return resolved
    return p


@dataclass(frozen=True)
class FramePathResolution:
    raw_path: str | None
    resolved_path: Path | None
    reason: str | None
    canonical: bool = False

    @property
    def valid(self) -> bool:
        return self.reason is None and self.resolved_path is not None


def resolve_experiment_frame_path(
    experiment: Experiment,
    frame_path: str | Path | None,
) -> FramePathResolution:
    """Resolve a frame path relative to its experiment, never the process CWD.

    Absolute paths inside the run remain readable for legacy manifests but are
    marked non-canonical so the quality gate can require migration.
    """
    if frame_path is None or not str(frame_path).strip():
        return FramePathResolution(None, None, "FRAME_PATH_MISSING")
    raw = str(frame_path).replace("\\", "/")
    path = Path(raw)
    run_root = experiment.run_dir.resolve()
    resolved = path.resolve() if path.is_absolute() else (run_root / path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError:
        return FramePathResolution(raw, resolved, "FRAME_PATH_OUTSIDE_EXPERIMENT")
    if not resolved.is_file():
        return FramePathResolution(raw, resolved, "FRAME_FILE_MISSING")
    return FramePathResolution(raw, resolved, None, canonical=not path.is_absolute())


def require_experiment_frame_path(
    experiment: Experiment,
    frame_path: str | Path | None,
) -> Path:
    """Return one usable frame file or raise a typed actionable error."""
    resolution = resolve_experiment_frame_path(experiment, frame_path)
    if not resolution.valid:
        raise FramePathError(
            f"{resolution.reason}: raw={resolution.raw_path!r} resolved={resolution.resolved_path}"
        )
    assert resolution.resolved_path is not None
    return resolution.resolved_path


def canonical_frame_path(experiment: Experiment, absolute_path: str | Path) -> str:
    """Return a POSIX path relative to ``experiment.run_dir``."""
    resolved = Path(absolute_path).resolve()
    try:
        relative = resolved.relative_to(experiment.run_dir.resolve())
    except ValueError as exc:
        raise FramePathError(f"Frame file is outside experiment run_dir: {resolved}") from exc
    return relative.as_posix()
