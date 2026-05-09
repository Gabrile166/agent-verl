"""Metrics for progress delta classification benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from ..core.schema import ProgressDeltaSample

_ALLOWED_LABELS = ["increase", "decrease", "same", "INVALID"]


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalize_prediction_label(result: Dict[str, Any]) -> str:
    label = str(result.get("predicted_label", "")).strip().lower()
    if label in {"increase", "decrease", "same"}:
        return label
    return "INVALID"


def _delta_bucket(delta: float) -> str:
    if abs(delta) <= 1e-9:
        return "zero"
    if delta <= -0.2:
        return "negative_large"
    if delta < 0:
        return "negative_small"
    if delta < 0.2:
        return "positive_small"
    return "positive_large"


def summarize_results(results: Sequence[Dict[str, Any]], samples: Sequence[ProgressDeltaSample]) -> Dict[str, Any]:
    result_by_id = {
        str(item.get("sample_id", "")): item
        for item in results
        if str(item.get("sample_id", ""))
    }

    num_samples = len(samples)
    invalid_count = 0
    correct_count = 0

    accuracy_by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_env = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_generation_mode = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_delta_bucket = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        predicted_label = _normalize_prediction_label(result or {})
        if predicted_label == "INVALID":
            invalid_count += 1
        is_correct = predicted_label == sample.label
        if is_correct:
            correct_count += 1

        generation_mode = str(sample.details.get("generation_mode") or "unknown")
        delta_bucket = _delta_bucket(sample.progress_delta)

        accuracy_by_label[sample.label]["total"] += 1
        accuracy_by_env[sample.env]["total"] += 1
        accuracy_by_generation_mode[generation_mode]["total"] += 1
        accuracy_by_delta_bucket[delta_bucket]["total"] += 1
        confusion_matrix[sample.label][predicted_label] += 1

        if is_correct:
            accuracy_by_label[sample.label]["correct"] += 1
            accuracy_by_env[sample.env]["correct"] += 1
            accuracy_by_generation_mode[generation_mode]["correct"] += 1
            accuracy_by_delta_bucket[delta_bucket]["correct"] += 1

    return {
        "num_samples": num_samples,
        "num_results": len(result_by_id),
        "accuracy": _safe_div(correct_count, num_samples),
        "invalid_response_rate": _safe_div(invalid_count, num_samples),
        "accuracy_by_label": {
            label: _safe_div(stats["correct"], stats["total"])
            for label, stats in sorted(accuracy_by_label.items())
        },
        "accuracy_by_env": {
            env: _safe_div(stats["correct"], stats["total"])
            for env, stats in sorted(accuracy_by_env.items())
        },
        "accuracy_by_generation_mode": {
            mode: _safe_div(stats["correct"], stats["total"])
            for mode, stats in sorted(accuracy_by_generation_mode.items())
        },
        "accuracy_by_delta_bucket": {
            bucket: _safe_div(stats["correct"], stats["total"])
            for bucket, stats in sorted(accuracy_by_delta_bucket.items())
        },
        "confusion_matrix": {
            gold_label: {predicted: count for predicted, count in sorted(predictions.items())}
            for gold_label, predictions in sorted(confusion_matrix.items())
        },
    }
