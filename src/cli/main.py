"""Command-line interface for the retrieval pipeline."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import sys

from dotenv import load_dotenv

from config.settings import Experiment, PipelineConfig, validate_experiment_name
from core.errors import CodeNovaError
from core.logging import configure_logging, get_logger
from indexing.build_index import build_index
from indexing.embeddings import embed_frames
from indexing.extract_text import export_text, extract_asr, extract_ocr, import_text
from indexing.frames import extract_frames
from indexing.ingest import ingest_videos
from indexing.shots import detect_shots
from modules.reranker.base import build_reranker
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

    embed_parser = subparsers.add_parser(
        "embed-frames", help="Embed keyframes (SigLIP2/BEiT-3/Vietnamese caption embedding)"
    )
    add_run_args(embed_parser)
    embed_parser.add_argument("--batch-size", default=32, type=int)
    embed_parser.add_argument("--force", action="store_true")

    index_parser = subparsers.add_parser("build-index", help="Build the Qdrant vector index")
    add_run_args(index_parser)
    index_parser.add_argument("--force", action="store_true")

    text_parser = subparsers.add_parser(
        "extract-text", help="Run OCR + ASR and index text into Elasticsearch"
    )
    add_run_args(text_parser)
    text_parser.add_argument("--force", action="store_true")
    text_parser.add_argument("--skip-ocr", action="store_true", help="Skip the OCR sub-stage")
    text_parser.add_argument("--skip-asr", action="store_true", help="Skip the ASR sub-stage")

    export_text_parser = subparsers.add_parser(
        "export-text", help="Dump OCR/ASR documents from Elasticsearch to manifests/text.jsonl"
    )
    add_run_args(export_text_parser)

    import_text_parser = subparsers.add_parser(
        "import-text", help="Load manifests/text.jsonl into Elasticsearch"
    )
    add_run_args(import_text_parser)

    search_parser = subparsers.add_parser("search", help="Search videos by text query")
    add_run_args(search_parser)
    search_parser.add_argument("query")

    ui_parser = subparsers.add_parser("serve-ui", help="Serve the local retrieval UI")
    add_run_args(ui_parser)
    ui_parser.add_argument("--host", default=os.environ.get("CODENOVA_UI_HOST", "127.0.0.1"))
    ui_parser.add_argument(
        "--port", default=int(os.environ.get("CODENOVA_UI_PORT", "7860")), type=int
    )
    ui_parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "Cross-encoder reranker model name or HuggingFace ID. "
            "e.g. 'blip2-itm' or 'Salesforce/blip2-itm-vit-b'. "
            "Omit to disable reranking."
        ),
    )
    ui_parser.add_argument(
        "--reranker-top-k",
        default=10,
        type=int,
        help="Number of results to keep after reranking (default: 10).",
    )
    validate_parser = subparsers.add_parser("validate-experiment-name", help="Validate a run name")
    validate_parser.add_argument("name", help="Experiment name to validate")

    return parser


def add_config_args(parser: ArgumentParser) -> None:
    """Attach shared pipeline config arguments."""
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path)
    parser.add_argument(
        "--embedding-models",
        default=os.environ.get("EMBEDDING_MODELS", "beit3-large"),
        help=(
            "Comma-separated embedding models. BEiT-3 large only by default; "
            "add siglip2/vietnamese-embedding for SRRF fusion + rerank. "
            "Defaults to $EMBEDDING_MODELS."
        ),
    )
    parser.add_argument("--frame-sampling", default="shot-percentile")
    parser.add_argument("--index-backend", default="qdrant")
    parser.add_argument(
        "--keyframe-percentiles",
        default="0.15,0.5,0.85",
        help="Comma-separated shot percentiles to sample keyframes at",
    )
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--device", default="auto")


def add_run_args(parser: ArgumentParser) -> None:
    """Attach shared existing-run arguments."""
    add_config_args(parser)
    parser.add_argument("--experiment-name", required=True, help="Experiment name")


def parse_percentiles(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated percentile string into a tuple of floats."""
    return tuple(float(part) for part in raw.split(",") if part.strip())


def parse_embedding_models(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated embedding model list."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def config_from_args(args: Namespace) -> PipelineConfig:
    """Create a pipeline config from parsed CLI args."""
    return PipelineConfig(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        embedding_models=parse_embedding_models(args.embedding_models),
        frame_sampling=args.frame_sampling,
        index_backend=args.index_backend,
        keyframe_percentiles=parse_percentiles(args.keyframe_percentiles),
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
    """Run keyframe embedding for every configured model."""
    experiment = load_experiment(args)
    count = embed_frames(
        experiment=experiment,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps({"experiment": experiment.name, "embeddings": count}))
    return 0


def handle_build_index(args: Namespace) -> int:
    """Build Qdrant index."""
    experiment = load_experiment(args)
    count = build_index(experiment=experiment, force=args.force)
    print(json.dumps({"experiment": experiment.name, "indexed": count}))
    return 0


def handle_extract_text(args: Namespace) -> int:
    """Run OCR + ASR and index text into Elasticsearch."""
    experiment = load_experiment(args)
    ocr_count = 0 if args.skip_ocr else extract_ocr(experiment=experiment, force=args.force)
    asr_count = 0 if args.skip_asr else extract_asr(experiment=experiment, force=args.force)
    print(
        json.dumps(
            {"experiment": experiment.name, "ocr_indexed": ocr_count, "asr_indexed": asr_count}
        )
    )
    return 0


def handle_export_text(args: Namespace) -> int:
    """Dump OCR/ASR documents from Elasticsearch to a local JSONL file."""
    experiment = load_experiment(args)
    count = export_text(experiment=experiment)
    print(json.dumps({"experiment": experiment.name, "exported": count}))
    return 0


def handle_import_text(args: Namespace) -> int:
    """Load a local JSONL text export into Elasticsearch."""
    experiment = load_experiment(args)
    count = import_text(experiment=experiment)
    print(json.dumps({"experiment": experiment.name, "imported": count}))
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

    # Build reranker if --reranker-model was supplied; None disables reranking.
    reranker = build_reranker(
        model_name=getattr(args, "reranker_model", None),
        device=args.device,
    )
    if reranker:
        LOGGER.info(
            "Reranker enabled: model=%s top_k=%s",
            getattr(reranker, "model_name", "?"),
            getattr(args, "reranker_top_k", 10),
        )
    else:
        LOGGER.info("Reranker: disabled (no --reranker-model supplied)")

    print(f"Serving retrieval UI at http://{args.host}:{args.port}")
    serve_ui(
        experiment=experiment,
        host=args.host,
        port=args.port,
        default_top_k=args.top_k,
        reranker=reranker,
        reranker_top_k=getattr(args, "reranker_top_k", 10),
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
        if args.command == "extract-text":
            return handle_extract_text(args)
        if args.command == "export-text":
            return handle_export_text(args)
        if args.command == "import-text":
            return handle_import_text(args)
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
