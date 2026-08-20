"""Command-line interface for the retrieval pipeline."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
import json
import os
import sys

from dotenv import load_dotenv

# Phai chay TRUOC moi import noi bo: nhieu module (vd modules/embedding/siglip.py)
# doc os.environ.get("SOME_FLAG", ...) ngay o module-level luc import, khong phai
# luc goi ham. Neu load_dotenv() chay sau (vd trong main()), cac module da import
# se dong bang gia tri mac dinh truoc khi .env kip nap - .env bi lang phi vo hieu.
load_dotenv()

from config.settings import Experiment, PipelineConfig, validate_experiment_name  # noqa: E402
from core.errors import CodeNovaError, ExperimentNameError  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from indexing.build_index import build_index  # noqa: E402
from indexing.embeddings import embed_frames  # noqa: E402
from indexing.extract_text import (  # noqa: E402
    drop_ocr_watermarks,
    export_text,
    extract_asr,
    extract_ocr,
    import_text,
)
from indexing.frames import extract_frames  # noqa: E402
from indexing.frame_paths import (  # noqa: E402
    apply_frame_path_migration,
    plan_frame_path_migration,
    write_frame_path_migration_audit,
)
from indexing.ingest import ingest_videos  # noqa: E402
from indexing.manifest import JsonlManifest  # noqa: E402
from indexing.preflight import (  # noqa: E402
    build_preflight_plan,
    verify_preflight_plan,
    write_preflight_plan,
)
from indexing.readiness import write_readiness  # noqa: E402
from indexing.shots import detect_shots  # noqa: E402
from indexing.state import JobState  # noqa: E402
from indexing.validation import validate_experiment_artifacts  # noqa: E402
from modules.reranker.base import build_reranker  # noqa: E402
from retrieval import build_retriever  # noqa: E402
from ui.server import serve_ui  # noqa: E402

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
        "--plan", type=Path, help="Approved preflight plan to verify before discovery"
    )
    ingest_parser.add_argument(
        "--force", action="store_true", help="Re-record completed discovery items"
    )

    shots_parser = subparsers.add_parser("detect-shots", help="Detect shots with TransNetV2")
    add_run_args(shots_parser)
    shots_parser.add_argument(
        "--transnetv2-weights",
        type=Path,
        help="Converted PyTorch weights; downloaded and converted automatically if missing.",
    )
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
    embed_parser.add_argument(
        "--caption-missing",
        action="store_true",
        help=(
            "vietnamese-embedding: caption frames that have none yet instead of "
            "skipping them. Without it, only frames already in captions.jsonl are embedded."
        ),
    )

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

    watermarks_parser = subparsers.add_parser(
        "drop-ocr-watermarks",
        help="Strip recurring station-watermark lines from already-extracted OCR text",
    )
    add_run_args(watermarks_parser)
    watermarks_parser.add_argument(
        "--drop-ratio",
        type=float,
        default=None,
        help="Line document-frequency threshold (default: OCR_WATERMARK_DROP_RATIO env or 0.15)",
    )

    export_text_parser = subparsers.add_parser(
        "export-text", help="Dump text documents from Elasticsearch to manifests/text.jsonl"
    )
    add_run_args(export_text_parser)

    import_text_parser = subparsers.add_parser(
        "import-text", help="Load text.jsonl and captions.jsonl into Elasticsearch"
    )
    add_run_args(import_text_parser)
    import_text_parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Do not import manifests/captions.jsonl when it exists",
    )

    preflight_parser = subparsers.add_parser(
        "preflight-index", help="Inventory config, device and dataset before indexing"
    )
    add_run_args(preflight_parser)
    preflight_parser.add_argument("--input", required=True, type=Path)
    preflight_parser.add_argument(
        "--approve", action="store_true", help="Mark the generated immutable plan approved"
    )

    validate_index_parser = subparsers.add_parser(
        "validate-index", help="Validate offline artifacts and write readiness.json"
    )
    add_run_args(validate_index_parser)

    offline_parser = subparsers.add_parser(
        "offline-index",
        help="Run approved preflight, all vector stages and the final quality gate",
    )
    add_config_args(offline_parser)
    offline_parser.add_argument("--input", required=True, type=Path)
    offline_parser.add_argument("--experiment-name", required=True)
    offline_parser.add_argument("--resume", action="store_true")
    offline_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the printed immutable plan and start indexing",
    )
    offline_parser.add_argument("--force", action="store_true")
    offline_parser.add_argument("--batch-size", default=32, type=int)
    offline_parser.add_argument("--caption-missing", action="store_true")
    offline_parser.add_argument("--with-text", action="store_true")
    offline_parser.add_argument("--skip-ocr", action="store_true")
    offline_parser.add_argument("--skip-asr", action="store_true")
    offline_parser.add_argument("--transnetv2-weights", type=Path)
    offline_parser.add_argument("--transnetv2-module-dir", type=Path)

    repair_parser = subparsers.add_parser(
        "repair-manifest", help="Inspect or explicitly remove corrupt JSONL lines"
    )
    add_run_args(repair_parser)
    repair_parser.add_argument(
        "manifest", choices=("videos", "shots", "frames", "captions", "text")
    )
    repair_parser.add_argument(
        "--apply", action="store_true", help="Apply repair; default is a read-only dry run"
    )

    migrate_paths_parser = subparsers.add_parser(
        "migrate-frame-paths",
        help="Convert legacy frame paths to experiment-relative paths",
    )
    add_run_args(migrate_paths_parser)
    migrate_paths_parser.add_argument(
        "--legacy-root",
        required=True,
        type=Path,
        help="Root against which old relative frame paths were originally written",
    )
    migrate_paths_parser.add_argument(
        "--apply", action="store_true", help="Apply migration; default is a read-only dry run"
    )

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
    parser.add_argument("--data-dir", default=None, type=Path)
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path)
    parser.add_argument(
        "--embedding-models",
        default=None,
        help=(
            "Comma-separated embedding models. Jina CLIP v2 only by default; "
            "add beit3-large/siglip2/vietnamese-embedding for SRRF fusion + rerank. "
            "Defaults to $EMBEDDING_MODELS."
        ),
    )
    parser.add_argument("--frame-sampling", default=None)
    parser.add_argument("--index-backend", default=None)
    parser.add_argument(
        "--keyframe-percentiles",
        default=None,
        help="Comma-separated shot percentiles to sample keyframes at",
    )
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--device", default="auto")


def add_run_args(parser: ArgumentParser) -> None:
    """Attach existing-run options without replacing persisted artifact config."""
    parser.add_argument("--data-dir", default=None, type=Path)
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path)
    parser.add_argument("--embedding-models", default=None)
    parser.add_argument("--frame-sampling", default=None)
    parser.add_argument("--index-backend", default=None)
    parser.add_argument("--keyframe-percentiles", default=None)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment-name", required=True, help="Experiment name")


def parse_percentiles(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated percentile string into a tuple of floats."""
    return tuple(float(part) for part in raw.split(",") if part.strip())


def parse_embedding_models(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated embedding model list."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def config_from_args(args: Namespace) -> PipelineConfig:
    """Create a pipeline config from parsed CLI args."""
    defaults = PipelineConfig()
    embedding_models = getattr(args, "embedding_models", None)
    percentiles = getattr(args, "keyframe_percentiles", None)
    return PipelineConfig(
        data_dir=getattr(args, "data_dir", None) or defaults.data_dir,
        runs_dir=args.runs_dir,
        embedding_models=(
            parse_embedding_models(embedding_models)
            if embedding_models is not None
            else parse_embedding_models(
                os.environ.get("EMBEDDING_MODELS", ",".join(defaults.embedding_models))
            )
        ),
        frame_sampling=getattr(args, "frame_sampling", None) or defaults.frame_sampling,
        index_backend=getattr(args, "index_backend", None) or defaults.index_backend,
        keyframe_percentiles=(
            parse_percentiles(percentiles)
            if percentiles is not None
            else defaults.keyframe_percentiles
        ),
        top_k=args.top_k,
        device=args.device,
    )


def artifact_overrides_from_args(args: Namespace) -> dict[str, object]:
    """Return only artifact options explicitly supplied for an existing run."""
    overrides: dict[str, object] = {}
    if getattr(args, "data_dir", None) is not None:
        overrides["data_dir"] = args.data_dir
    if getattr(args, "embedding_models", None) is not None:
        overrides["embedding_models"] = parse_embedding_models(args.embedding_models)
    if getattr(args, "frame_sampling", None) is not None:
        overrides["frame_sampling"] = args.frame_sampling
    if getattr(args, "index_backend", None) is not None:
        overrides["index_backend"] = args.index_backend
    if getattr(args, "keyframe_percentiles", None) is not None:
        overrides["keyframe_percentiles"] = parse_percentiles(args.keyframe_percentiles)
    return overrides


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
        artifact_overrides=artifact_overrides_from_args(args) if args.resume else None,
    )
    execution_id = configure_logging(
        experiment.run_dir / "logs",
        verbose=args.verbose,
        command=args.command,
        experiment=experiment.name,
    )
    LOGGER.info(
        "event=EXECUTION_STARTED execution_id=%s experiment=%s run_dir=%s",
        execution_id,
        experiment.name,
        experiment.run_dir,
    )
    if args.plan:
        plan = verify_preflight_plan(experiment, args.input, args.plan)
        LOGGER.info("event=PREFLIGHT_VERIFIED plan_id=%s", plan.get("plan_id"))
    count = ingest_videos(experiment=experiment, input_dir=args.input, force=args.force)
    print(
        json.dumps(
            {"experiment": experiment.name, "recorded": count, "run_dir": str(experiment.run_dir)}
        )
    )
    return _stage_exit_code(experiment, "DISCOVER")


def _stage_exit_code(experiment: Experiment, *stages: str) -> int:
    """Return failure when a stage completed only partially."""
    state = JobState(experiment.run_dir / "jobs.sqlite")
    failures = [failure for stage in stages for failure in state.failures(stage)]
    if failures:
        LOGGER.error("event=STAGE_PARTIAL_FAILURE count=%s stages=%s", len(failures), stages)
        return 1
    return 0


def load_experiment(args: Namespace) -> Experiment:
    """Load an existing experiment directory using CLI config values."""
    config = config_from_args(args)
    experiment = Experiment.open(
        config=config,
        name=args.experiment_name,
        artifact_overrides=artifact_overrides_from_args(args),
    )
    execution_id = configure_logging(
        experiment.run_dir / "logs",
        verbose=args.verbose,
        command=args.command,
        experiment=experiment.name,
    )
    LOGGER.info(
        "event=EXECUTION_STARTED execution_id=%s experiment=%s run_dir=%s",
        execution_id,
        experiment.name,
        experiment.run_dir,
    )
    LOGGER.info(
        "event=EXPERIMENT_CONFIG_RESTORED config_hash=%s artifact_config=%s runtime_overrides=%s",
        experiment.config.config_hash(),
        experiment.config.artifact_payload(),
        experiment.config.runtime_payload(),
    )
    return experiment


def handle_preflight_index(args: Namespace) -> int:
    config = config_from_args(args)
    run_dir = config.runs_dir / args.experiment_name
    experiment = (
        Experiment.open(
            config=config,
            name=args.experiment_name,
            artifact_overrides=artifact_overrides_from_args(args),
        )
        if run_dir.exists()
        else Experiment.create(config=config, name=args.experiment_name)
    )
    execution_id = configure_logging(
        experiment.run_dir / "logs",
        verbose=args.verbose,
        command=args.command,
        experiment=experiment.name,
    )
    LOGGER.info(
        "event=EXECUTION_STARTED execution_id=%s experiment=%s run_dir=%s",
        execution_id,
        experiment.name,
        experiment.run_dir,
    )
    plan = build_preflight_plan(experiment, args.input, approved=args.approve)
    path = write_preflight_plan(experiment, plan)
    LOGGER.info(
        "event=PREFLIGHT_COMPLETED status=%s videos=%s device=%s plan=%s",
        plan["status"],
        plan["dataset"]["video_count"],
        plan["device"]["resolved"],
        path,
    )
    print(json.dumps({**plan, "path": str(path)}, indent=2, ensure_ascii=False))
    return 0 if args.approve else 1


def handle_validate_index(args: Namespace) -> int:
    experiment = load_experiment(args)
    LOGGER.info("event=VALIDATION_STARTED experiment=%s", experiment.name)
    report = validate_experiment_artifacts(experiment)
    path = write_readiness(experiment, report)
    for issue in report.issues:
        log = LOGGER.error if issue.severity == "ERROR" else LOGGER.warning
        log(
            "event=VALIDATION_ISSUE severity=%s code=%s stage=%s item_id=%s message=%s",
            issue.severity,
            issue.code,
            issue.stage,
            issue.item_id,
            issue.message,
        )
    LOGGER.info("event=VALIDATION_COMPLETED status=%s readiness=%s", report.status, path)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.status == "READY" else 1


def handle_offline_index(args: Namespace) -> int:
    """Run the vector indexing workflow behind one approval and one final gate."""
    from core.external_setup import TRANSNETV2_PYTORCH_DIR

    config = config_from_args(args)
    run_dir = config.runs_dir / args.experiment_name
    if run_dir.exists():
        if not args.resume:
            raise ExperimentNameError(
                f"Experiment '{args.experiment_name}' already exists. Use --resume to continue."
            )
        experiment = Experiment.open(
            config=config,
            name=args.experiment_name,
            artifact_overrides=artifact_overrides_from_args(args),
        )
    else:
        experiment = Experiment.create(config=config, name=args.experiment_name)
    execution_id = configure_logging(
        experiment.run_dir / "logs",
        verbose=args.verbose,
        command=args.command,
        experiment=experiment.name,
    )
    LOGGER.info("event=EXECUTION_STARTED execution_id=%s", execution_id)
    plan = build_preflight_plan(experiment, args.input, approved=args.approve)
    plan_path = write_preflight_plan(experiment, plan)
    print(json.dumps({**plan, "path": str(plan_path)}, indent=2, ensure_ascii=False))
    if not args.approve:
        LOGGER.warning("event=PREFLIGHT_AWAITING_APPROVAL plan=%s", plan_path)
        return 1
    verify_preflight_plan(experiment, args.input, plan_path)

    pipeline_error: Exception | None = None
    try:
        ingest_videos(experiment, args.input, force=args.force)
        detect_shots(
            experiment,
            weights_path=args.transnetv2_weights,
            module_dir=args.transnetv2_module_dir or TRANSNETV2_PYTORCH_DIR,
            force=args.force,
        )
        extract_frames(experiment, force=args.force)
        embed_frames(
            experiment,
            batch_size=args.batch_size,
            force=args.force,
            caption_missing=args.caption_missing,
        )
        if args.with_text:
            if not args.skip_ocr:
                extract_ocr(experiment, force=args.force)
            if not args.skip_asr:
                extract_asr(experiment, force=args.force)
        build_index(experiment, force=args.force)
    except Exception as exc:  # quality gate still runs and records a failed execution
        pipeline_error = exc
        LOGGER.exception("event=OFFLINE_PIPELINE_FAILED error=%s", exc)

    report = validate_experiment_artifacts(experiment)
    readiness_path = write_readiness(experiment, report)
    LOGGER.info(
        "event=OFFLINE_PIPELINE_COMPLETED status=%s readiness=%s",
        report.status,
        readiness_path,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if pipeline_error is None and report.status == "READY" else 1


def handle_repair_manifest(args: Namespace) -> int:
    """Dry-run or repair one known manifest and persist an audit record."""
    experiment = load_experiment(args)
    manifest_path = experiment.run_dir / "manifests" / f"{args.manifest}.jsonl"
    manifest = JsonlManifest(manifest_path)
    result = manifest.repair_corrupt_lines(dry_run=not args.apply, backup=True)
    payload = {
        "experiment": experiment.name,
        "manifest": str(manifest_path),
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "valid_records": len(result.rows),
        "corrupt_lines": [
            {
                "line_number": line.line_number,
                "content": line.content,
                "error": line.error,
            }
            for line in result.corrupt_lines
        ],
        "changed": bool(args.apply and result.corrupt_lines),
        "created_at": datetime.now(UTC).isoformat(),
    }
    audit_dir = experiment.run_dir / "logs" / "repairs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{args.manifest}_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "audit_path": str(audit_path)}, indent=2, ensure_ascii=False))
    return 0


def handle_migrate_frame_paths(args: Namespace) -> int:
    """Dry-run or explicitly migrate all frame manifests for one experiment."""
    experiment = load_experiment(args)
    plan = plan_frame_path_migration(experiment, args.legacy_root)
    if args.apply:
        audit_path = apply_frame_path_migration(experiment, plan)
        mode = "APPLY"
    else:
        audit_path = write_frame_path_migration_audit(experiment, plan, mode="DRY_RUN")
        mode = "DRY_RUN"
    payload = {
        **plan.to_dict(mode=mode, changed=bool(args.apply and plan.changed_records)),
        "audit_path": str(audit_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not plan.issues else 1


def handle_detect_shots(args: Namespace) -> int:
    """Run TransNetV2 shot detection."""
    from core.external_setup import TRANSNETV2_PYTORCH_DIR

    experiment = load_experiment(args)
    count = detect_shots(
        experiment=experiment,
        weights_path=args.transnetv2_weights,
        module_dir=args.transnetv2_module_dir or TRANSNETV2_PYTORCH_DIR,
        force=args.force,
    )
    print(json.dumps({"experiment": experiment.name, "shots": count}))
    return _stage_exit_code(experiment, "SHOT_DETECT")


def handle_extract_frames(args: Namespace) -> int:
    """Run OpenCV frame extraction."""
    experiment = load_experiment(args)
    count = extract_frames(experiment=experiment, force=args.force)
    print(json.dumps({"experiment": experiment.name, "frames": count}))
    return _stage_exit_code(experiment, "FRAME_EXTRACT")


def handle_embed_frames(args: Namespace) -> int:
    """Run keyframe embedding for every configured model."""
    experiment = load_experiment(args)
    count = embed_frames(
        experiment=experiment,
        batch_size=args.batch_size,
        force=args.force,
        caption_missing=args.caption_missing,
    )
    print(json.dumps({"experiment": experiment.name, "embeddings": count}))
    return _stage_exit_code(experiment, "EMBED")


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
    stages = tuple(
        stage
        for stage, skipped in (("EXTRACT_OCR", args.skip_ocr), ("EXTRACT_ASR", args.skip_asr))
        if not skipped
    )
    return _stage_exit_code(experiment, *stages)


def handle_drop_ocr_watermarks(args: Namespace) -> int:
    """Strip recurring station-watermark lines from already-extracted OCR text."""
    experiment = load_experiment(args)
    kwargs = {} if args.drop_ratio is None else {"drop_ratio": args.drop_ratio}
    count = drop_ocr_watermarks(experiment=experiment, **kwargs)
    print(json.dumps({"experiment": experiment.name, "documents_updated": count}))
    return 0


def handle_export_text(args: Namespace) -> int:
    """Dump text documents from Elasticsearch to a local JSONL file."""
    experiment = load_experiment(args)
    count = export_text(experiment=experiment)
    print(json.dumps({"experiment": experiment.name, "exported": count}))
    return 0


def handle_import_text(args: Namespace) -> int:
    """Load a local JSONL text export into Elasticsearch."""
    experiment = load_experiment(args)
    count = import_text(experiment=experiment, include_captions=not args.no_captions)
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
        if args.command == "drop-ocr-watermarks":
            return handle_drop_ocr_watermarks(args)
        if args.command == "export-text":
            return handle_export_text(args)
        if args.command == "import-text":
            return handle_import_text(args)
        if args.command == "preflight-index":
            return handle_preflight_index(args)
        if args.command == "validate-index":
            return handle_validate_index(args)
        if args.command == "offline-index":
            return handle_offline_index(args)
        if args.command == "repair-manifest":
            return handle_repair_manifest(args)
        if args.command == "migrate-frame-paths":
            return handle_migrate_frame_paths(args)
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
