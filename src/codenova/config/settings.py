"""Application settings and experiment naming."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import re

from codenova.core.errors import ExperimentNameError

_EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,119}$")


def slugify(value: str) -> str:
    """Return a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "unnamed"


def validate_experiment_name(name: str) -> str:
    """Validate and return an experiment name.

    Experiment names are intentionally strict because they become directory names,
    run IDs, and comparison keys.
    """
    if not _EXPERIMENT_NAME_RE.fullmatch(name):
        raise ExperimentNameError(
            "Experiment names must be 3-120 characters, lowercase, start with a "
            "letter or number, and contain only letters, numbers, hyphens, or "
            "underscores."
        )
    if "/" in name or "\\" in name or " " in name:
        raise ExperimentNameError("Experiment names cannot contain spaces or path separators.")
    return name


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration that defines a retrieval experiment."""

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    pipeline: str = "retrieval"
    clip_model: str = "clip-vit-b-32"
    frame_sampling: str = "shot-percentile"
    index_backend: str = "qdrant"
    keyframe_percentiles: tuple[float, ...] = (0.15, 0.5, 0.85)
    top_k: int = 20
    device: str = "auto"

    def normalized_payload(self) -> dict[str, str | int]:
        """Return stable config values used for hashing and run metadata."""
        return {
            "pipeline": slugify(self.pipeline),
            "clip_model": slugify(self.clip_model),
            "frame_sampling": slugify(self.frame_sampling),
            "index_backend": slugify(self.index_backend),
            "keyframe_percentiles": ",".join(str(p) for p in self.keyframe_percentiles),
            "top_k": self.top_k,
            "device": slugify(self.device),
        }

    def config_hash(self) -> str:
        """Return a short deterministic hash for the experiment configuration."""
        payload = json.dumps(self.normalized_payload(), sort_keys=True).encode("utf-8")
        return sha256(payload).hexdigest()[:8]

    def default_experiment_name(self, now: datetime | None = None) -> str:
        """Build a valid deterministic experiment name for this config and date."""
        current = now or datetime.now(UTC)
        parts = [
            current.strftime("%Y%m%d"),
            slugify(self.pipeline),
            slugify(self.clip_model),
            slugify(self.frame_sampling),
            slugify(self.index_backend),
            self.config_hash(),
        ]
        return validate_experiment_name("_".join(parts))


@dataclass(frozen=True)
class Experiment:
    """A concrete run directory and its metadata."""

    name: str
    run_dir: Path
    config: PipelineConfig
    alias: str | None = None
    created_at: str = ""

    @classmethod
    def create(
        cls,
        config: PipelineConfig,
        name: str | None = None,
        alias: str | None = None,
        resume: bool = False,
    ) -> "Experiment":
        """Create or resume an experiment directory."""
        experiment_name = validate_experiment_name(name or config.default_experiment_name())
        run_dir = config.runs_dir / experiment_name
        if run_dir.exists() and not resume:
            raise ExperimentNameError(
                f"Experiment '{experiment_name}' already exists. Use --resume or choose another name."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC).isoformat()
        experiment = cls(
            name=experiment_name,
            run_dir=run_dir,
            config=config,
            alias=alias,
            created_at=created_at,
        )
        experiment.write_metadata()
        return experiment

    @classmethod
    def open(cls, config: PipelineConfig, name: str) -> "Experiment":
        """Open an existing experiment directory without rewriting metadata."""
        experiment_name = validate_experiment_name(name)
        run_dir = config.runs_dir / experiment_name
        if not run_dir.exists():
            raise ExperimentNameError(f"Experiment '{experiment_name}' does not exist.")
        return cls(name=experiment_name, run_dir=run_dir, config=config)

    def write_metadata(self) -> None:
        """Persist experiment metadata to ``config.json``."""
        payload = {
            "name": self.name,
            "alias": self.alias,
            "created_at": self.created_at,
            "config_hash": self.config.config_hash(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.config).items()
            },
        }
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
