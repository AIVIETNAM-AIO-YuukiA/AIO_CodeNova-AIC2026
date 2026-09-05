"""Offline evaluation for grounded Video Question Answering.

The evaluator only consumes manually labelled qrels.  It supports partial
labels: an item may provide an answer and a relevant video without inventing
frame IDs or timestamps that have not been annotated yet.

Usage::

    uv run python -m evaluation.vqa \
      --experiment-name result \
      --qrels eval/vqa_qrels.jsonl \
      --top-k 20,50
"""

from __future__ import annotations

import inspect
import json
import re
import unicodedata
from argparse import ArgumentParser, Namespace
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv

# This module is commonly invoked directly with ``python -m`` rather than
# through the CodeNova CLI, so it must load the project environment itself.
# Do this before importing settings/retrieval modules whose defaults are read
# from environment variables at import time.
load_dotenv()

from config.settings import Experiment, PipelineConfig  # noqa: E402


@dataclass(frozen=True)
class RelevantVideo:
    """One relevant video, identified by an internal ID and/or display name."""

    video_id: str = ""
    video_name: str = ""


@dataclass(frozen=True)
class RelevantMoment:
    """A labelled answer-bearing interval in one video."""

    start_sec: float
    end_sec: float
    video_id: str = ""
    video_name: str = ""


@dataclass(frozen=True)
class VqaQrel:
    """Ground truth for one VQA retrieval-and-answering query."""

    query_id: str
    query: str
    question: str
    context: str
    acceptable_answers: tuple[str, ...]
    relevant_videos: tuple[RelevantVideo, ...]
    relevant_frame_ids: frozenset[str]
    relevant_moments: tuple[RelevantMoment, ...]
    group: str = "unspecified"


def load_qrels(path: Path) -> list[VqaQrel]:
    """Load a UTF-8 JSONL qrels file and validate supplied labels."""
    if not path.is_file():
        raise ValueError(f"Qrels file does not exist: {path}")

    qrels: list[VqaQrel] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"qrels line {line_number} must be a JSON object")

        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        question = str(row.get("question", "")).strip()
        if not query_id or not query or not question:
            raise ValueError(
                f"qrels line {line_number} requires query_id, query, and question"
            )
        if query_id in seen:
            raise ValueError(f"Duplicate query_id in qrels: {query_id}")
        seen.add(query_id)

        answers_value = row.get("acceptable_answers", row.get("answers", []))
        if answers_value is None:
            answers_value = []
        if isinstance(answers_value, str):
            answers_value = [answers_value]
        if not isinstance(answers_value, list):
            raise ValueError(
                f"qrels line {line_number} acceptable_answers must be a string or array"
            )
        answers = tuple(str(value).strip() for value in answers_value if str(value).strip())

        videos = tuple(
            _parse_relevant_video(item, line_number)
            for item in _require_array(row, "relevant_videos", line_number)
        )
        legacy_video_ids = [
            RelevantVideo(video_id=str(value).strip())
            for value in _require_array(row, "relevant_video_ids", line_number)
            if str(value).strip()
        ]
        legacy_video_names = [
            RelevantVideo(video_name=str(value).strip())
            for value in _require_array(row, "relevant_video_names", line_number)
            if str(value).strip()
        ]
        videos = _deduplicate_videos((*videos, *legacy_video_ids, *legacy_video_names))

        moments = tuple(
            _parse_relevant_moment(item, line_number)
            for item in _require_array(row, "relevant_moments", line_number)
        )
        frames = frozenset(
            str(value).strip()
            for value in _require_array(row, "relevant_frame_ids", line_number)
            if str(value).strip()
        )
        if not answers and not videos and not frames and not moments:
            raise ValueError(f"qrels line {line_number} has no answer or relevance labels")

        qrels.append(
            VqaQrel(
                query_id=query_id,
                query=query,
                question=question,
                context=str(row.get("context", "")).strip(),
                acceptable_answers=answers,
                relevant_videos=videos,
                relevant_frame_ids=frames,
                relevant_moments=moments,
                group=str(row.get("group", "unspecified")).strip() or "unspecified",
            )
        )

    if not qrels:
        raise ValueError(f"Qrels file is empty: {path}")
    return qrels


def _require_array(row: dict, key: str, line_number: int) -> list:
    value = row.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"qrels line {line_number} field {key} must be an array")
    return value


def _parse_relevant_video(item: object, line_number: int) -> RelevantVideo:
    if isinstance(item, str):
        video = RelevantVideo(video_name=item.strip())
    elif isinstance(item, dict):
        video = RelevantVideo(
            video_id=str(item.get("video_id", "")).strip(),
            video_name=str(item.get("video_name", "")).strip(),
        )
    else:
        raise ValueError(f"qrels line {line_number} has an invalid relevant_videos entry")
    if not video.video_id and not video.video_name:
        raise ValueError(f"qrels line {line_number} has a relevant video without an ID or name")
    return video


def _parse_relevant_moment(item: object, line_number: int) -> RelevantMoment:
    if not isinstance(item, dict):
        raise ValueError(f"qrels line {line_number} has an invalid relevant_moments entry")
    try:
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"qrels line {line_number} has a moment without numeric start_sec/end_sec"
        ) from exc
    video_id = str(item.get("video_id", "")).strip()
    video_name = str(item.get("video_name", "")).strip()
    if not video_id and not video_name:
        raise ValueError(f"qrels line {line_number} has a moment without video_id/video_name")
    if not isfinite(start_sec) or not isfinite(end_sec) or end_sec <= start_sec:
        raise ValueError(f"qrels line {line_number} contains an invalid time interval")
    return RelevantMoment(
        start_sec=start_sec,
        end_sec=end_sec,
        video_id=video_id,
        video_name=video_name,
    )


def _deduplicate_videos(videos: Iterable[RelevantVideo]) -> tuple[RelevantVideo, ...]:
    output: list[RelevantVideo] = []
    seen: set[tuple[str, str]] = set()
    for video in videos:
        key = (video.video_id.casefold(), canonical_video_name(video.video_name))
        if key not in seen:
            seen.add(key)
            output.append(video)
    return tuple(output)


def normalize_answer(answer: object) -> str:
    """Normalize case, punctuation, and common answer prefixes without removing accents."""
    text = unicodedata.normalize("NFKC", str(answer or "")).casefold().strip()
    text = "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char for char in text
    )
    text = " ".join(text.split())
    prefixes = (
        r"^(?:x|đáp án|câu trả lời)\s+(?:là)\s+",
        r"^(?:the answer|answer|x)\s+(?:is)\s+",
    )
    for prefix in prefixes:
        text = re.sub(prefix, "", text, count=1)
    return text.strip()


def answer_exact_match(prediction: object, references: Iterable[str]) -> float | None:
    normalized_references = {normalize_answer(value) for value in references}
    normalized_references.discard("")
    if not normalized_references:
        return None
    return float(normalize_answer(prediction) in normalized_references)


def answer_token_f1(prediction: object, references: Iterable[str]) -> float | None:
    normalized_references = [normalize_answer(value) for value in references]
    normalized_references = [value for value in normalized_references if value]
    if not normalized_references:
        return None
    prediction_tokens = normalize_answer(prediction).split()
    return max(
        _token_f1(prediction_tokens, reference.split())
        for reference in normalized_references
    )


def _token_f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def canonical_video_name(value: object) -> str:
    """Turn paths such as ``data\\raw_videos\\L26_V254.mp4`` into ``l26_v254``."""
    text = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def evaluate_response(qrel: VqaQrel, response: dict) -> dict[str, object]:
    """Score one VQA response against every ground-truth field that exists."""
    answer = str(response.get("answer", "") or "")
    evidence = _extract_evidence_frames(response)
    predicted_moments = _extract_predicted_moments(response)
    candidates = _extract_candidate_videos(response)
    return {
        "answer": answer,
        "answer_status": str(response.get("answer_status", "") or ""),
        "answer_confidence": _optional_float(response.get("answer_confidence")),
        "answer_em": answer_exact_match(answer, qrel.acceptable_answers),
        "answer_f1": answer_token_f1(answer, qrel.acceptable_answers),
        "video_recall@3": video_recall_at_k(qrel, candidates, 3),
        "evidence_accuracy": evidence_accuracy(qrel, evidence),
        "temporal_iou": temporal_iou(qrel, predicted_moments),
        "selected_video": _selected_video_key(response),
        "evidence_frame_ids": sorted(
            {str(item.get("frame_id", "")) for item in evidence if item.get("frame_id")}
        ),
        "openrouter_calls": _openrouter_calls(response),
        "openrouter_cost": _openrouter_cost(response),
    }


def video_recall_at_k(
    qrel: VqaQrel,
    candidates: list[RelevantVideo],
    cutoff: int,
) -> float | None:
    targets = _video_targets(qrel)
    if not targets:
        return None
    matched = sum(
        any(_videos_match(target, candidate) for candidate in candidates[:cutoff])
        for target in targets
    )
    return matched / len(targets)


def evidence_accuracy(qrel: VqaQrel, evidence: list[dict]) -> float | None:
    """Binary grounded-evidence hit using frame labels first, then moment labels."""
    if qrel.relevant_frame_ids:
        predicted = {str(item.get("frame_id", "")) for item in evidence}
        return float(bool(predicted & qrel.relevant_frame_ids))
    if not qrel.relevant_moments:
        return None
    for item in evidence:
        timestamp = _optional_float(item.get("timestamp_sec"))
        if timestamp is None:
            continue
        candidate = RelevantVideo(
            video_id=str(item.get("video_id", "") or ""),
            video_name=str(item.get("video_name", "") or ""),
        )
        for moment in qrel.relevant_moments:
            target = RelevantVideo(moment.video_id, moment.video_name)
            if _videos_match(target, candidate) and moment.start_sec <= timestamp <= moment.end_sec:
                return 1.0
    return 0.0


def temporal_iou(qrel: VqaQrel, predicted: list[RelevantMoment]) -> float | None:
    if not qrel.relevant_moments:
        return None
    scores = [
        interval_iou(
            (candidate.start_sec, candidate.end_sec),
            (target.start_sec, target.end_sec),
        )
        for candidate in predicted
        for target in qrel.relevant_moments
        if _videos_match(
            RelevantVideo(target.video_id, target.video_name),
            RelevantVideo(candidate.video_id, candidate.video_name),
        )
    ]
    return max(scores, default=0.0)


def evaluate_queries(
    qrels: list[VqaQrel],
    search: Callable[..., dict],
    top_ks: list[int],
    search_options: dict | None = None,
) -> dict:
    """Run VQA at every requested Top K and report accuracy plus stability."""
    if not top_ks or any(value <= 0 for value in top_ks):
        raise ValueError("top_ks must contain positive integers")
    top_ks = sorted(set(top_ks))
    options = dict(search_options or {})
    query_rows: list[dict] = []
    all_runs: list[dict] = []

    for qrel in qrels:
        runs: list[dict] = []
        for top_k in top_ks:
            started = perf_counter()
            error = ""
            try:
                response = search(
                    query=qrel.query,
                    question=qrel.question,
                    context=qrel.context,
                    top_k=top_k,
                    **options,
                )
                if not isinstance(response, dict):
                    raise TypeError("VQA search must return a dictionary")
            except Exception as exc:  # Keep the remaining labelled queries evaluable.
                response = {"answer": "", "answer_status": "error"}
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (perf_counter() - started) * 1000.0
            scored = evaluate_response(qrel, response)
            run_row = {
                "query_id": qrel.query_id,
                "group": qrel.group,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "error": error,
                **scored,
            }
            runs.append(run_row)
            all_runs.append(run_row)

        stability = evaluate_top_k_stability(runs)
        query_rows.append(
            {
                "query_id": qrel.query_id,
                "group": qrel.group,
                **stability,
                "runs": runs,
            }
        )

    largest_k = max(top_ks)
    canonical_runs = [row for row in all_runs if row["top_k"] == largest_k]
    groups = {
        group: _summarize(
            [row for row in canonical_runs if row["group"] == group],
            [row for row in query_rows if row["group"] == group],
            [row for row in all_runs if row["group"] == group],
        )
        for group in sorted({str(row["group"]) for row in query_rows})
    }
    by_top_k = {
        str(top_k): _summarize(
            [row for row in all_runs if row["top_k"] == top_k],
            [],
            [row for row in all_runs if row["top_k"] == top_k],
        )
        for top_k in top_ks
    }
    return {
        "summary": _summarize(canonical_runs, query_rows, all_runs),
        "by_top_k": by_top_k,
        "groups": groups,
        "queries": query_rows,
    }


def evaluate_top_k_stability(runs: list[dict]) -> dict[str, float | None]:
    if len(runs) < 2:
        return {
            "answer_top_k_stability": None,
            "video_top_k_stability": None,
            "evidence_top_k_jaccard": None,
            "top_k_stability": None,
        }
    answers = [
        (
            str(row.get("answer_status", "") or ""),
            normalize_answer(row.get("answer")),
        )
        for row in runs
    ]
    videos = [
        (
            str(row.get("answer_status", "") or ""),
            str(row.get("selected_video", "")),
        )
        for row in runs
    ]
    # Repeated abstention is stable output too. Accuracy is measured by the
    # answer/evidence metrics; this metric only asks whether Top K changed the
    # system decision.
    answer_stability = float(len(set(answers)) == 1)
    video_stability = float(len(set(videos)) == 1)

    evidence_sets = [set(row.get("evidence_frame_ids", [])) for row in runs]
    pairwise: list[float] = []
    for left_index, left in enumerate(evidence_sets):
        for right in evidence_sets[left_index + 1 :]:
            if left or right:
                pairwise.append(len(left & right) / len(left | right))
    evidence_jaccard = mean(pairwise) if pairwise else None
    return {
        "answer_top_k_stability": answer_stability,
        "video_top_k_stability": video_stability,
        "evidence_top_k_jaccard": evidence_jaccard,
        "top_k_stability": mean((answer_stability, video_stability)),
    }


def _summarize(
    canonical_runs: list[dict],
    query_rows: list[dict],
    measured_runs: list[dict],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "queries": len(canonical_runs),
        "requests": len(measured_runs),
        "errors": sum(bool(row.get("error")) for row in measured_runs),
    }
    for key in (
        "answer_em",
        "answer_f1",
        "video_recall@3",
        "evidence_accuracy",
        "temporal_iou",
        "answer_confidence",
    ):
        values = [float(row[key]) for row in canonical_runs if row.get(key) is not None]
        if values:
            summary[key] = mean(values)
            summary[f"{key}_evaluated"] = len(values)
    for key in (
        "answer_top_k_stability",
        "video_top_k_stability",
        "evidence_top_k_jaccard",
        "top_k_stability",
    ):
        values = [float(row[key]) for row in query_rows if row.get(key) is not None]
        if values:
            summary[key] = mean(values)

    latencies = sorted(float(row["latency_ms"]) for row in measured_runs)
    summary["latency_p50_ms"] = percentile(latencies, 50)
    summary["latency_p95_ms"] = percentile(latencies, 95)
    summary["openrouter_calls_total"] = sum(
        int(row.get("openrouter_calls", 0)) for row in measured_runs
    )
    summary["openrouter_cost_total"] = sum(
        float(row.get("openrouter_cost", 0.0)) for row in measured_runs
    )
    summary["openrouter_calls_per_request"] = (
        summary["openrouter_calls_total"] / len(measured_runs) if measured_runs else 0.0
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


def _video_targets(qrel: VqaQrel) -> tuple[RelevantVideo, ...]:
    if qrel.relevant_videos:
        return qrel.relevant_videos
    return _deduplicate_videos(
        RelevantVideo(moment.video_id, moment.video_name) for moment in qrel.relevant_moments
    )


def _videos_match(target: RelevantVideo, candidate: RelevantVideo) -> bool:
    if target.video_id and candidate.video_id:
        if target.video_id.casefold() == candidate.video_id.casefold():
            return True
    target_name = canonical_video_name(target.video_name)
    candidate_name = canonical_video_name(candidate.video_name)
    return bool(target_name and candidate_name and target_name == candidate_name)


def _extract_candidate_videos(response: dict) -> list[RelevantVideo]:
    records: list[object] = []
    # ``candidates`` is the retrieval-order list. Do not prepend the final
    # selection: video Recall@3 measures candidate retrieval, not answer
    # selection, and reordering it would silently turn the metric into another
    # quantity.
    if isinstance(response.get("candidates"), list):
        records.extend(response["candidates"])
    for key in ("candidate_answers",):
        value = response.get(key)
        if isinstance(value, list):
            records.extend(value)
    if not records and isinstance(response.get("selected_candidate"), dict):
        records.append(response["selected_candidate"])
    if not records and isinstance(response.get("results"), list):
        records.extend(response["results"])

    videos: list[RelevantVideo] = []
    for item in records:
        record = _unwrap_candidate(item)
        if not record:
            continue
        video = RelevantVideo(
            video_id=str(record.get("video_id", "") or ""),
            video_name=str(record.get("video_name", "") or ""),
        )
        if video.video_id or video.video_name:
            videos.append(video)
    return list(_deduplicate_videos(videos))


def _unwrap_candidate(item: object) -> dict:
    if not isinstance(item, dict):
        return {}
    for key in ("candidate", "selected_candidate", "moment"):
        nested = item.get(key)
        if isinstance(nested, dict):
            merged = dict(item)
            merged.update(nested)
            return merged
    return item


def _selected_video_key(response: dict) -> str:
    record = response.get("selected_candidate")
    record = _unwrap_candidate(record) if isinstance(record, dict) else {}
    if not record:
        return ""
    video = RelevantVideo(
        video_id=str(record.get("video_id", "") or ""),
        video_name=str(record.get("video_name", "") or ""),
    )
    return canonical_video_name(video.video_name) or video.video_id.casefold()


def _extract_evidence_frames(response: dict) -> list[dict]:
    items: list[object] = []
    top_level = response.get("evidence_frames")
    if isinstance(top_level, list):
        items.extend(top_level)
    selected = response.get("selected_candidate")
    nested = (
        selected.get("evidence_frames", [])
        if isinstance(selected, dict)
        and isinstance(selected.get("evidence_frames"), list)
        else []
    )
    supporting = response.get("supporting_frame_ids")
    supporting_ids = {
        str(value) for value in supporting if value
    } if isinstance(supporting, list) else set()

    if supporting_ids and items:
        # The citation contract is authoritative. A context frame may be sent
        # to the verifier without supporting the final answer, so its mere
        # presence in the top-level payload must not earn evidence credit.
        items = [
            item
            for item in items
            if isinstance(item, str) and item in supporting_ids
            or isinstance(item, dict)
            and str(item.get("frame_id", "")) in supporting_ids
        ]

    # Grounded responses expose only verifier-cited frames at the top level.
    # Never turn all 4-6 context frames from selected_candidate into credited
    # evidence. If only IDs are available, enrich those IDs from the nested
    # records; legacy responses without a citation contract may fall back to
    # their nested evidence list.
    if not items and supporting_ids:
        nested_by_id = {
            str(item.get("frame_id")): item
            for item in nested
            if isinstance(item, dict) and item.get("frame_id")
        }
        items.extend(
            nested_by_id.get(frame_id, {"frame_id": frame_id})
            for frame_id in supporting_ids
        )
    elif not items and not supporting_ids:
        items.extend(nested)

    default_record = _unwrap_candidate(selected) if isinstance(selected, dict) else {}
    output: list[dict] = []
    for item in items:
        if isinstance(item, str):
            record = {"frame_id": item}
        elif isinstance(item, dict):
            record = dict(item)
        else:
            continue
        for key in ("video_id", "video_name"):
            if not record.get(key) and default_record.get(key):
                record[key] = default_record[key]
        output.append(record)

    known_ids = {str(item.get("frame_id", "")) for item in output}
    output.extend(
        {"frame_id": frame_id}
        for frame_id in supporting_ids
        if frame_id not in known_ids
    )
    seen: set[tuple[str, str, object]] = set()
    deduplicated: list[dict] = []
    for item in output:
        key = (
            str(item.get("frame_id", "")),
            str(item.get("video_id", "")),
            item.get("timestamp_sec"),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def _extract_predicted_moments(response: dict) -> list[RelevantMoment]:
    records: list[dict] = []
    for key in ("selected_candidate", "selected_moment"):
        value = response.get(key)
        if isinstance(value, dict):
            records.append(_unwrap_candidate(value))
    moments: list[RelevantMoment] = []
    for record in records:
        interval = _point_aware_interval(record)
        if interval is None:
            continue
        moments.append(
            RelevantMoment(
                start_sec=interval[0],
                end_sec=interval[1],
                video_id=str(record.get("video_id", "") or ""),
                video_name=str(record.get("video_name", "") or ""),
            )
        )
    return moments


def _point_aware_interval(record: dict) -> tuple[float, float] | None:
    """Expand point candidates using the actual multi-frame evidence span."""
    interval = _extract_interval(record)
    if interval is None or interval[1] > interval[0]:
        return interval
    evidence = record.get("evidence_frames")
    timestamps = [
        timestamp
        for item in evidence if isinstance(item, dict)
        if (timestamp := _optional_float(item.get("timestamp_sec"))) is not None
    ] if isinstance(evidence, list) else []
    if len(timestamps) >= 2:
        start, end = min(timestamps), max(timestamps)
        if end > start:
            return start, end
    timestamp = timestamps[0] if timestamps else interval[0]
    return max(0.0, timestamp - 0.5), timestamp + 0.5


def _extract_interval(record: dict) -> tuple[float, float] | None:
    for start_key, end_key in (
        ("start_sec", "end_sec"),
        ("start_timestamp", "end_timestamp"),
        ("start_time_sec", "end_time_sec"),
    ):
        start = _optional_float(record.get(start_key))
        end = _optional_float(record.get(end_key))
        if start is not None and end is not None and end >= start:
            return start, end
    return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _openrouter_calls(response: dict) -> int:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    for key in ("openrouter_http_requests", "openrouter_calls", "request_count", "requests"):
        value = usage.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    return 0


def _openrouter_cost(response: dict) -> float:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    try:
        value = float(usage.get("cost", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) and value >= 0 else 0.0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate grounded VQA accuracy and stability")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--top-k", default="20,50")
    parser.add_argument("--pipeline-mode", choices=("grounded", "legacy"), default="grounded")
    parser.add_argument("--enabled-models", default="")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def run(args: Namespace) -> dict:
    top_ks = sorted({int(value) for value in str(args.top_k).split(",") if value.strip()})
    if not top_ks or any(value <= 0 for value in top_ks):
        raise ValueError("--top-k must contain positive comma-separated integers")
    qrels = load_qrels(args.qrels)
    experiment = Experiment.open(
        PipelineConfig(runs_dir=args.runs_dir, device=args.device, top_k=max(top_ks)),
        args.experiment_name,
    )
    from retrieval.vqa import vqa_search

    enabled_models = [
        value.strip() for value in str(args.enabled_models).split(",") if value.strip()
    ]
    options = {
        "pipeline_mode": args.pipeline_mode,
        "enabled_models": enabled_models or None,
        "use_reranker": not args.no_reranker,
        "use_llm": not args.no_llm,
    }

    def search(**kwargs) -> dict:
        return _call_with_supported_arguments(vqa_search, experiment=experiment, **kwargs)

    evaluation = evaluate_queries(qrels, search, top_ks, options)
    return {
        "experiment": experiment.name,
        "created_at": datetime.now(UTC).isoformat(),
        "qrels": str(args.qrels),
        "top_k": top_ks,
        "pipeline_mode": args.pipeline_mode,
        **evaluation,
    }


def _call_with_supported_arguments(callable_: Callable[..., dict], **kwargs) -> dict:
    """Keep the evaluator usable against both legacy and grounded VQA signatures."""
    parameters = inspect.signature(callable_).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    supported = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items() if key in parameters}
    )
    return callable_(**supported)


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = args.runs_dir / args.experiment_name / "evaluation" / f"vqa_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "queries": report["summary"]["queries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
