"""Logging setup for command-line pipeline runs."""

from __future__ import annotations

from pathlib import Path
import logging


def configure_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure console, pipeline, and error logs.

    Calling this function multiple times resets existing handlers so tests and
    repeated CLI invocations do not duplicate log lines.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)

    pipeline_file = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    pipeline_file.setLevel(logging.DEBUG)
    pipeline_file.setFormatter(formatter)

    error_file = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(pipeline_file)
    root.addHandler(error_file)
