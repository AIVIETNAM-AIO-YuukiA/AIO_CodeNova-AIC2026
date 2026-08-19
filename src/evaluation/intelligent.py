"""Evaluate Intelligent Search with frame- and moment-level qrels.

Usage::

    uv run python -m evaluation.intelligent \
      --experiment-name result \
      --qrels eval/intelligent_qrels.jsonl \
      --top-k 5,10,20 \
      --ablation all

The evaluator deliberately consumes manually labelled qrels.  It never
manufactures relevance labels from the system being evaluated, which would
make comparisons circular.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log2
from pathlib import Path
from statistics import mean
from time import perf_counter

from config.settings import Experiment, PipelineConfig


@dataclass(frozen=True)
class RelevantMoment:
    video_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class QueryQrel:
    query_id: str
    query: str
    relevant_frame_ids: frozenset[str]
    relevant_moments: tuple[RelevantMoment, ...]
    expected_modalities: frozenset[str]
    group: str = "unspecified"


ABLATIONS: dict[str, dict] = {
    "kis": {"enable_kis": True, "enable_ocr": False, "enable_asr": False, "enable_caption": False},
    "kis_ocr": {"enable_kis": True, "enable_ocr": True, "enable_asr": False, "enable_caption": False},
    "kis_asr": {"enable_kis": True, "enable_ocr": False, "enable_asr": True, "enable_caption": False},
    "kis_caption": {"enable_kis": True, "enable_ocr": False, "enable_asr": False, "enable_caption": True},
    "all_fixed": {"fusion_mode": "fixed"},
    "all_adaptive": {"fusion_mode": "adaptive"},
    "all_no_llm": {"fusion_mode": "adaptive", "use_llm": False},
    "asr_nearest": {"enable_ocr": False, "enable_caption": False, "temporal_asr": False},
    "all_no_rerank": {
        "fusion_mode": "adaptive",
        "use_reranker": False,
        "use_evidence_reranker": False,
    },
    "joint_text": {"fusion_mode": "adaptive", "text_search_mode": "joint"},
}


def load_qrels(path: Path) -> list[QueryQrel]:
    """Load and validate a JSONL qrels file."""
    if not path.is_file():
        raise ValueError(f"Qrels file does not exist: {path}")
    qrels: list[QueryQrel] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            raise ValueError(f"qrels line {line_number} requires query_id and query")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id in qrels: {query_id}")
        seen.add(query_id)
        moments = tuple(
            RelevantMoment(
                video_id=str(item["video_id"]),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
            )
            for item in row.get("relevant_moments", [])
        )
        if any(moment.end_sec < moment.start_sec for moment in moments):
            raise ValueError(f"qrels line {line_number} contains an inverted time interval")
        frames = frozenset(str(value) for value in row.get("relevant_frame_ids", []) if value)
        if not frames and not moments:
            raise ValueError(f"qrels line {line_number} has no relevance labels")
        qrels.append(
            QueryQrel(
                query_id=query_id,
                query=query,
                relevant_frame_ids=frames,
                relevant_moments=moments,
                expected_modalities=frozenset(
                    str(value).lower() for value in row.get("expected_modalities", [])
                ),
                group=str(row.get("group", "unspecified")),
            )
        )
    if not qrels:
        raise ValueError(f"Qrels file is empty: {path}")
    return qrels


def evaluate_ranking(qrel: QueryQrel, results: list[dict], cutoffs: list[int]) -> dict:
    """Compute binary frame ranking and temporal localization metrics."""
    target_sets = [_ranking_targets(qrel, result) for result in results]
    target_count = (
        len(qrel.relevant_frame_ids)
        if qrel.relevant_frame_ids
        else len(qrel.relevant_moments)
    )
    novelty_gains: list[float] = []
    discovered: set[tuple[str, object]] = set()
    for targets in target_sets:
        novel = targets - discovered
        novelty_gains.append(float(bool(novel)))
        discovered.update(targets)

    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        found_targets = set().union(*target_sets[:cutoff]) if target_sets[:cutoff] else set()
        metrics[f"recall@{cutoff}"] = (
            len(found_targets) / target_count if target_count else 0.0
        )
        gains = novelty_gains[:cutoff]
        dcg = sum(gain / log2(rank + 2) for rank, gain in enumerate(gains))
        ideal_count = min(target_count, cutoff)
        idcg = sum(1.0 / log2(rank + 2) for rank in range(ideal_count))
        metrics[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
    candidate_targets = set().union(*target_sets[:50]) if target_sets[:50] else set()
    metrics["candidate_recall@50"] = (
        len(candidate_targets) / target_count if target_count else 0.0
    )

    reciprocal_rank = 0.0
    for rank, gain in enumerate(novelty_gains, start=1):
        if gain:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = reciprocal_rank

    temporal_ious: list[float] = []
    top1_temporal_ious: list[float] = []
    for result_rank, result in enumerate(results):
        video_id = str(result.get("video_id", ""))
        candidate_intervals = _candidate_intervals(result)
        for moment in qrel.relevant_moments:
            if moment.video_id == video_id:
                current = [
                    interval_iou(candidate, (moment.start_sec, moment.end_sec))
                    for candidate in candidate_intervals
                ]
                temporal_ious.extend(current)
                if result_rank == 0:
                    top1_temporal_ious.extend(current)
    best_iou = max(temporal_ious, default=0.0)
    top1_iou = max(top1_temporal_ious, default=0.0)
    metrics["temporal_iou"] = best_iou
    metrics["temporal_r@1_iou0.5"] = float(top1_iou >= 0.5)
    metrics["temporal_r@1_iou0.7"] = float(top1_iou >= 0.7)
    return metrics


def _ranking_targets(qrel: QueryQrel, result: dict) -> set[tuple[str, object]]:
    """Return distinct relevance targets hit by one result.

    Frame qrels are preferred when supplied. Moment targets are used for
    temporal-only qrels so their Recall/nDCG/MRR are meaningful rather than
    being forced to zero.
    """
    frame_id = str(result.get("frame_id", ""))
    if qrel.relevant_frame_ids:
        return {("frame", frame_id)} if frame_id in qrel.relevant_frame_ids else set()

    video_id = str(result.get("video_id", ""))
    intervals = _candidate_intervals(result)
    return {
        ("moment", index)
        for index, moment in enumerate(qrel.relevant_moments)
        if moment.video_id == video_id
        and any(
            interval_iou(candidate, (moment.start_sec, moment.end_sec)) > 0
            for candidate in intervals
        )
    }


def _candidate_intervals(result: dict) -> list[tuple[float, float]]:
    intervals = [
        (float(item["segment_start_sec"]), float(item["segment_end_sec"]))
        for item in result.get("evidence", [])
        if item.get("source") == "asr"
        and item.get("segment_start_sec") is not None
        and item.get("segment_end_sec") is not None
    ]
    timestamp = result.get("timestamp_sec")
    if not intervals and timestamp is not None:
        intervals = [(float(timestamp) - 1.0, float(timestamp) + 1.0)]
    return intervals


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def evaluate_ablation(
    qrels: list[QueryQrel],
    search: Callable[..., dict],
    options: dict,
    cutoffs: list[int],
) -> dict:
    """Run one ablation using an injected search callable."""
    rows: list[dict] = []
    max_k = max(max(cutoffs), 50)
    for qrel in qrels:
        started = perf_counter()
        response = search(query=qrel.query, top_k=max_k, **options)
        latency_ms = (perf_counter() - started) * 1000.0
        results = list(response.get("results", []))
        ranking = evaluate_ranking(qrel, results, cutoffs)
        analysis = response.get("analysis") or {}
        states = response.get("component_status") or {}
        component_counts = response.get("component_counts") or {}
        routed = {
            name
            for name, state in states.items()
            if state in {"used", "no_hits", "error"}
            and name in {"ocr", "asr", "caption"}
        }
        expected = {name for name in qrel.expected_modalities if name != "kis"}
        route_precision = (
            len(routed & expected) / len(routed) if routed else float(not expected)
        )
        route_recall = len(routed & expected) / len(expected) if expected else 1.0
        usage = analysis.get("llm_usage") or {}
        rows.append(
            {
                "query_id": qrel.query_id,
                "group": qrel.group,
                "latency_ms": latency_ms,
                "route_precision": route_precision,
                "route_recall": route_recall,
                "llm_fallback": analysis.get("routing_mode") == "fallback",
                "llm_attempts": int(analysis.get("llm_attempts", 0) or 0),
                "modality_hit_count": sum(
                    int(component_counts.get(name, 0) > 0)
                    for name in ("kis", "ocr", "asr", "caption")
                ),
                "openrouter_cost_usd": float(
                    usage.get("estimated_cost_usd") or usage.get("cost") or 0.0
                ),
                **ranking,
            }
        )
    groups = {
        group: _summarize_rows([row for row in rows if row["group"] == group])
        for group in sorted({str(row["group"]) for row in rows})
    }
    return {"summary": _summarize_rows(rows), "groups": groups, "queries": rows}


def _summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    """Aggregate metrics for the full run or one labelled query group."""
    numeric_keys = [
        key for key, value in rows[0].items() if isinstance(value, (int, float))
    ]
    summary: dict[str, float | int] = {
        key: mean(float(row[key]) for row in rows) for key in numeric_keys
    }
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    summary["latency_p50_ms"] = percentile(latencies, 50)
    summary["latency_p95_ms"] = percentile(latencies, 95)
    summary["queries"] = len(rows)
    summary["openrouter_cost_usd"] = sum(
        float(row["openrouter_cost_usd"]) for row in rows
    )
    return summary


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percent / 100.0
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate Intelligent Search ablations")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--ablation", default="all")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def run(args: Namespace) -> dict:
    cutoffs = sorted({int(value) for value in str(args.top_k).split(",") if value.strip()})
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("--top-k must contain positive comma-separated integers")
    requested = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    unknown = set(requested) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablation(s): {sorted(unknown)}")
    qrels = load_qrels(args.qrels)
    experiment = Experiment.open(
        PipelineConfig(runs_dir=args.runs_dir, device=args.device, top_k=max(cutoffs)),
        args.experiment_name,
    )
    from retrieval.intelligent_search import intelligent_search

    report = {
        "experiment": experiment.name,
        "created_at": datetime.now(UTC).isoformat(),
        "qrels": str(args.qrels),
        "cutoffs": cutoffs,
        "ablations": {},
    }
    for name in requested:
        report["ablations"][name] = evaluate_ablation(
            qrels,
            lambda **kwargs: intelligent_search(experiment, **kwargs),
            ABLATIONS[name],
            cutoffs,
        )
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = args.runs_dir / args.experiment_name / "evaluation" / f"intelligent_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ablations": list(report["ablations"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
