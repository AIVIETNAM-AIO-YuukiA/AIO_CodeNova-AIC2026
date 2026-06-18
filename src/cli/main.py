"""Command-line interface for the retrieval pipeline."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

from dotenv import load_dotenv

from config.settings import Experiment, PipelineConfig, validate_experiment_name
from core.errors import CodeNovaError
from core.logging import configure_logging, get_logger
from indexing.build_index import build_index
from indexing.embeddings import embed_frames
from indexing.frames import extract_frames
from indexing.ingest import ingest_videos
from indexing.shots import detect_shots
from retrieval import build_retriever
from ui.server import serve_ui

LOGGER = get_logger(__name__)


def build_parser() -> ArgumentParser:
    """Build the CLI parser."""
    parser = ArgumentParser(prog="codenova", description="Video retrieval pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug console logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    name_parser = subparsers.add_parser("name-experiment", help="Generate a valid experiment name")
    add_config_args(name_parser)

    ingest_parser = subparsers.add_parser("ingest", help="Discover videos and initialize a run")
    add_config_args(ingest_parser)
    ingest_parser.add_argument(
        "--input", required=True, type=Path, help="Directory containing videos"
    )
    ingest_parser.add_argument("--experiment-name", help="Explicit experiment name")
    ingest_parser.add_argument("--alias", help="Human-readable alias")
    ingest_parser.add_argument("--resume", action="store_true", help="Resume an existing run")
    ingest_parser.add_argument(
        "--force", action="store_true", help="Re-record completed discovery items"
    )

    shots_parser = subparsers.add_parser("detect-shots", help="Detect shots with TransNetV2")
    add_run_args(shots_parser)
    shots_parser.add_argument("--transnetv2-weights", required=True, type=Path)
    shots_parser.add_argument("--transnetv2-module-dir", type=Path)
    shots_parser.add_argument("--force", action="store_true")

    frames_parser = subparsers.add_parser("extract-frames", help="Extract shot keyframes")
    add_run_args(frames_parser)
    frames_parser.add_argument("--force", action="store_true")

    embed_parser = subparsers.add_parser("embed-frames", help="Embed keyframes with CLIP")
    add_run_args(embed_parser)
    embed_parser.add_argument("--batch-size", default=32, type=int)
    embed_parser.add_argument("--force", action="store_true")

    index_parser = subparsers.add_parser("build-index", help="Build the FAISS vector index")
    add_run_args(index_parser)
    index_parser.add_argument("--force", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search videos by text query")
    add_run_args(search_parser)
    search_parser.add_argument("query")

    ui_parser = subparsers.add_parser("serve-ui", help="Serve the local retrieval UI")
    add_run_args(ui_parser)
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", default=7860, type=int)

    validate_parser = subparsers.add_parser("validate-experiment-name", help="Validate a run name")
    validate_parser.add_argument("name", help="Experiment name to validate")

    return parser


def add_config_args(parser: ArgumentParser) -> None:
    """Attach shared pipeline config arguments."""
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path)
    parser.add_argument("--clip-model", default="clip-vit-b-32")
    parser.add_argument("--frame-sampling", default="shot-midpoint")
    parser.add_argument("--index-backend", default="qdrant")
    parser.add_argument("--keyframes-per-shot", default=1, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--device", default="auto")


def add_run_args(parser: ArgumentParser) -> None:
    """Attach shared existing-run arguments."""
    add_config_args(parser)
    parser.add_argument("--experiment-name", required=True, help="Experiment name")


def config_from_args(args: Namespace) -> PipelineConfig:
    """Create a pipeline config from parsed CLI args."""
    return PipelineConfig(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        clip_model=args.clip_model,
        frame_sampling=args.frame_sampling,
        index_backend=args.index_backend,
        keyframes_per_shot=args.keyframes_per_shot,
        top_k=args.top_k,
        device=args.device,
    )


def handle_name_experiment(args: Namespace) -> int:
    """Print the generated experiment name."""
    config = config_from_args(args)
    print(config.default_experiment_name())
    return 0


def handle_validate_experiment_name(args: Namespace) -> int:
    """Validate and print a normalized experiment name."""
    print(validate_experiment_name(args.name))
    return 0


def handle_ingest(args: Namespace) -> int:
    """Run video discovery for an experiment."""
    config = config_from_args(args)
    experiment = Experiment.create(
        config=config,
        name=args.experiment_name,
        alias=args.alias,
        resume=args.resume,
    )
    configure_logging(experiment.run_dir / "logs", verbose=args.verbose)
    LOGGER.info("Using experiment=%s run_dir=%s", experiment.name, experiment.run_dir)
    count = ingest_videos(experiment=experiment, input_dir=args.input, force=args.force)
    print(
        json.dumps(
            {"experiment": experiment.name, "recorded": count, "run_dir": str(experiment.run_dir)}
        )
    )
    return 0


def load_experiment(args: Namespace) -> Experiment:
    """Load an existing experiment directory using CLI config values."""
    config = config_from_args(args)
    experiment = Experiment.open(config=config, name=args.experiment_name)
    configure_logging(experiment.run_dir / "logs", verbose=args.verbose)
    LOGGER.info("Using experiment=%s run_dir=%s", experiment.name, experiment.run_dir)
    return experiment


def handle_detect_shots(args: Namespace) -> int:
    """Run TransNetV2 shot detection."""
    experiment = load_experiment(args)
    count = detect_shots(
        experiment=experiment,
        weights_path=args.transnetv2_weights,
        module_dir=args.transnetv2_module_dir,
        force=args.force,
    )
    print(json.dumps({"experiment": experiment.name, "shots": count}))
    return 0


def handle_extract_frames(args: Namespace) -> int:
    """Run OpenCV frame extraction."""
    experiment = load_experiment(args)
    count = extract_frames(experiment=experiment, force=args.force)
    print(json.dumps({"experiment": experiment.name, "frames": count}))
    return 0


def handle_embed_frames(args: Namespace) -> int:
    """Run CLIP image embedding."""
    experiment = load_experiment(args)
    count = embed_frames(experiment=experiment, batch_size=args.batch_size, force=args.force)
    print(json.dumps({"experiment": experiment.name, "embeddings": count}))
    return 0


def handle_build_index(args: Namespace) -> int:
    """Build FAISS index."""
    experiment = load_experiment(args)
    count = build_index(experiment=experiment, force=args.force)
    print(json.dumps({"experiment": experiment.name, "indexed": count}))
    return 0


def handle_search(args: Namespace) -> int:
    """Run text search."""
    experiment = load_experiment(args)
    retriever = build_retriever(experiment)
    results = retriever.search(query=args.query, top_k=args.top_k)
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0


def handle_serve_ui(args: Namespace) -> int:
    """Serve browser UI for one experiment."""
    experiment = load_experiment(args)
    print(f"Serving retrieval UI at http://{args.host}:{args.port}")
    serve_ui(
        experiment=experiment,
        host=args.host,
        port=args.port,
        default_top_k=args.top_k,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "name-experiment":
            return handle_name_experiment(args)
        if args.command == "validate-experiment-name":
            return handle_validate_experiment_name(args)
        if args.command == "ingest":
            return handle_ingest(args)
        if args.command == "detect-shots":
            return handle_detect_shots(args)
        if args.command == "extract-frames":
            return handle_extract_frames(args)
        if args.command == "embed-frames":
            return handle_embed_frames(args)
        if args.command == "build-index":
            return handle_build_index(args)
        if args.command == "search":
            return handle_search(args)
        if args.command == "serve-ui":
            return handle_serve_ui(args)
    except CodeNovaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
