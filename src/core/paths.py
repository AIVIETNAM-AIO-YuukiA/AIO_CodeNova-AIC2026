"""Utility for resolving paths globally."""

from pathlib import Path

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
