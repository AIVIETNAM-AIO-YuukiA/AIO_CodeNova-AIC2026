"""Logging setup for command-line pipeline runs."""

from __future__ import annotations

from logging.handlers import RotatingFileHandler
from datetime import UTC, datetime
from pathlib import Path
import logging
import re
import uuid

# Cap each log file so a long captioning/OCR run can't fill the disk: those
# stages emit one line per HTTP request, which reached 380MB over a 60k-frame
# pass and took the whole pipeline down with ENOSPC.
_MAX_LOG_BYTES = 32 * 1024 * 1024
_LOG_BACKUPS = 2

# Third-party loggers that emit one DEBUG/INFO line per network call. Their
# detail belongs on the console when debugging, not in a multi-hour log file.
_CHATTY_LOGGERS = ("httpx", "httpcore", "urllib3", "elastic_transport", "elasticsearch")


def configure_logging(
    log_dir: Path,
    verbose: bool = False,
    *,
    command: str = "pipeline",
    experiment: str = "unknown",
) -> str:
    """Configure console, pipeline, and error logs.

    Calling this function multiple times resets existing handlers so tests and
    repeated CLI invocations do not duplicate log lines.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    safe_command = re.sub(r"[^a-zA-Z0-9_-]+", "-", command).strip("-") or "pipeline"
    execution_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{safe_command}_{uuid.uuid4().hex[:8]}"

    class ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.execution_id = execution_id
            record.experiment = experiment
            return True

    context_filter = ContextFilter()
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s %(levelname)s execution=%(execution_id)s "
            "experiment=%(experiment)s %(name)s %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    console.addFilter(context_filter)

    pipeline_file = RotatingFileHandler(
        log_dir / "pipeline.log",
        encoding="utf-8",
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_LOG_BACKUPS,
    )
    pipeline_file.setLevel(logging.DEBUG if verbose else logging.INFO)
    pipeline_file.setFormatter(formatter)
    pipeline_file.addFilter(context_filter)

    execution_dir = log_dir / "executions"
    execution_dir.mkdir(parents=True, exist_ok=True)
    execution_file = RotatingFileHandler(
        execution_dir / f"{execution_id}.log",
        encoding="utf-8",
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_LOG_BACKUPS,
    )
    execution_file.setLevel(logging.DEBUG if verbose else logging.INFO)
    execution_file.setFormatter(formatter)
    execution_file.addFilter(context_filter)

    error_file = RotatingFileHandler(
        log_dir / "errors.log",
        encoding="utf-8",
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_LOG_BACKUPS,
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    error_file.addFilter(context_filter)

    root.addHandler(console)
    root.addHandler(pipeline_file)
    root.addHandler(execution_file)
    root.addHandler(error_file)

    for name in _CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.WARNING)
    return execution_id


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Use ``get_logger(__name__)`` instead of ``logging.getLogger(__name__)`` so
    every module obtains its logger the same way; output formatting and handlers
    are owned by :func:`configure_logging`.
    """
    return logging.getLogger(name)
