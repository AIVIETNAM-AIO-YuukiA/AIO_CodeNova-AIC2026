"""Application settings and experiment naming."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import re
import tempfile

from core.errors import ExperimentConfigError, ExperimentNameError

_EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,119}$")
EXPERIMENT_METADATA_SCHEMA_VERSION = 1
ARTIFACT_CONFIG_FIELDS = (
    "data_dir",
    "pipeline",
    "embedding_models",
    "frame_sampling",
    "index_backend",
    "keyframe_percentiles",
)
RUNTIME_CONFIG_FIELDS = ("runs_dir", "top_k", "device")


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
    embedding_models: tuple[str, ...] = ("jina-clip-v2",)
    frame_sampling: str = "shot-percentile"
    index_backend: str = "qdrant"
    keyframe_percentiles: tuple[float, ...] = (0.15, 0.5, 0.85)
    top_k: int = 20
    device: str = "auto"

    def artifact_payload(self) -> dict[str, object]:
        """Return typed, JSON-compatible values that define offline artifacts."""
        return {
            "data_dir": str(self.data_dir),
            "pipeline": self.pipeline,
            "embedding_models": list(self.embedding_models),
            "frame_sampling": self.frame_sampling,
            "index_backend": self.index_backend,
            "keyframe_percentiles": list(self.keyframe_percentiles),
        }

    def runtime_payload(self) -> dict[str, object]:
        """Return controls that may be overridden when an experiment is opened."""
        return {
            "runs_dir": str(self.runs_dir),
            "top_k": self.top_k,
            "device": self.device,
        }

    def normalized_payload(self) -> dict[str, object]:
        """Return artifact-defining values used for experiment hashing.

        ``device`` and ``top_k`` are runtime controls: changing either must not
        make persisted embeddings look like a different artifact definition.
        """
        return self.artifact_payload()

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
            "-".join(slugify(m) for m in self.embedding_models),
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
        artifact_overrides: dict[str, object] | None = None,
    ) -> "Experiment":
        """Create or resume an experiment directory."""
        experiment_name = validate_experiment_name(name or config.default_experiment_name())
        run_dir = config.runs_dir / experiment_name
        if run_dir.exists():
            if resume:
                return cls.open(
                    config=config,
                    name=experiment_name,
                    artifact_overrides=artifact_overrides,
                )
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
    def open(
        cls,
        config: PipelineConfig,
        name: str,
        *,
        artifact_overrides: dict[str, object] | None = None,
    ) -> "Experiment":
        """Open an experiment using its persisted artifact-defining config.

        ``runs_dir`` locates the run, while ``device`` and ``top_k`` are runtime
        controls. Every other field is restored from ``config.json`` so online
        commands cannot silently reinterpret offline artifacts.
        """
        experiment_name = validate_experiment_name(name)
        run_dir = config.runs_dir / experiment_name
        if not run_dir.exists():
            raise ExperimentNameError(f"Experiment '{experiment_name}' does not exist.")
        metadata_path = run_dir / "config.json"
        if not metadata_path.is_file():
            raise ExperimentConfigError(f"Experiment metadata is missing: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise TypeError("metadata root must be an object")
            schema_version = metadata.get("schema_version", 0)
            if schema_version not in (0, EXPERIMENT_METADATA_SCHEMA_VERSION):
                raise ValueError(f"unsupported schema_version={schema_version!r}")
            persisted = (
                metadata["artifact_config"]
                if schema_version == EXPERIMENT_METADATA_SCHEMA_VERSION
                else metadata["config"]
            )
            if metadata.get("name", experiment_name) != experiment_name:
                raise ValueError(
                    f"metadata name={metadata.get('name')!r} does not match {experiment_name!r}"
                )
            persisted_config = _config_from_persisted(persisted, runtime=config)
            if schema_version == EXPERIMENT_METADATA_SCHEMA_VERSION:
                expected_hash = persisted_config.config_hash()
                if metadata.get("config_hash") != expected_hash:
                    raise ValueError(
                        f"config_hash mismatch persisted={metadata.get('config_hash')!r} "
                        f"computed={expected_hash!r}"
                    )
            _validate_artifact_overrides(persisted_config, artifact_overrides or {})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExperimentConfigError(
                f"Experiment metadata is invalid: {metadata_path}: {exc}"
            ) from exc
        return cls(
            name=experiment_name,
            run_dir=run_dir,
            config=persisted_config,
            alias=metadata.get("alias"),
            created_at=str(metadata.get("created_at", "")),
        )

    def write_metadata(self) -> None:
        """Persist experiment metadata to ``config.json``."""
        payload = {
            "schema_version": EXPERIMENT_METADATA_SCHEMA_VERSION,
            "name": self.name,
            "alias": self.alias,
            "created_at": self.created_at,
            "config_hash": self.config.config_hash(),
            "artifact_config": self.config.artifact_payload(),
            "runtime_defaults": self.config.runtime_payload(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.config).items()
            },
        }
        path = self.run_dir / "config.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        try:
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _config_from_persisted(payload: object, *, runtime: PipelineConfig) -> PipelineConfig:
    if not isinstance(payload, dict):
        raise TypeError("artifact config must be an object")
    models = payload["embedding_models"]
    percentiles = payload["keyframe_percentiles"]
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("embedding_models must be a non-empty array")
    if not isinstance(percentiles, (list, tuple)) or not percentiles:
        raise ValueError("keyframe_percentiles must be a non-empty array")
    parsed_percentiles = tuple(float(value) for value in percentiles)
    if any(value < 0.0 or value > 1.0 for value in parsed_percentiles):
        raise ValueError("keyframe_percentiles must be within [0, 1]")
    return PipelineConfig(
        data_dir=Path(str(payload["data_dir"])),
        runs_dir=runtime.runs_dir,
        pipeline=str(payload["pipeline"]),
        embedding_models=tuple(str(value) for value in models),
        frame_sampling=str(payload["frame_sampling"]),
        index_backend=str(payload["index_backend"]),
        keyframe_percentiles=parsed_percentiles,
        top_k=runtime.top_k,
        device=runtime.device,
    )


def _validate_artifact_overrides(persisted: PipelineConfig, overrides: dict[str, object]) -> None:
    unknown = set(overrides) - set(ARTIFACT_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown artifact override fields: {sorted(unknown)}")
    for field_name, requested in overrides.items():
        actual = getattr(persisted, field_name)
        if field_name == "data_dir":
            requested = Path(str(requested))
        elif field_name in {"embedding_models", "keyframe_percentiles"}:
            requested = tuple(requested) if isinstance(requested, (list, tuple)) else requested
        if requested != actual:
            raise ExperimentConfigError(
                f"Artifact config mismatch for {field_name}: "
                f"persisted={actual!r}, requested={requested!r}"
            )
