"""Metrics for trajectory milestone localization benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from ..core.schema import TrajectoryMilestoneSample


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_results(results: Sequence[Dict[str, Any]], samples: Sequence[TrajectoryMilestoneSample]) -> Dict[str, Any]:
    result_by_id = {
        str(item.get("sample_id", "")): item
        for item in results
        if str(item.get("sample_id", ""))
    }

    correct_count = 0
    invalid_count = 0
    off_by_one_count = 0
    absolute_errors_on_valid: List[int] = []
    absolute_errors_with_invalid: List[int] = []

    accuracy_by_env = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_task_id = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    accuracy_by_label_index = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        result = result_by_id.get(sample.sample_id) or {}
        predicted_label = str(result.get("predicted_label", "")).strip() or "INVALID"
        predicted_label_index = result.get("predicted_label_index")
        is_invalid = predicted_label == "INVALID" or predicted_label_index is None
        is_correct = predicted_label == sample.label
        off_by_one = bool(result.get("off_by_one", False)) if not is_invalid else False

        if is_correct:
            correct_count += 1
        if is_invalid:
            invalid_count += 1
        if off_by_one:
            off_by_one_count += 1

        task_id = sample.task_id or "unknown"
        label_index_key = str(sample.label_index)
        accuracy_by_env[sample.env]["total"] += 1
        accuracy_by_task_id[task_id]["total"] += 1
        accuracy_by_label[sample.label]["total"] += 1
        accuracy_by_label_index[label_index_key]["total"] += 1
        confusion_matrix[sample.label][predicted_label] += 1

        if is_correct:
            accuracy_by_env[sample.env]["correct"] += 1
            accuracy_by_task_id[task_id]["correct"] += 1
            accuracy_by_label[sample.label]["correct"] += 1
            accuracy_by_label_index[label_index_key]["correct"] += 1

        if is_invalid:
            absolute_errors_with_invalid.append(sample.max_label_index_error())
        else:
            absolute_error = abs(int(predicted_label_index) - sample.label_index)
            absolute_errors_on_valid.append(absolute_error)
            absolute_errors_with_invalid.append(absolute_error)

    mae_on_valid = (
        sum(absolute_errors_on_valid) / len(absolute_errors_on_valid) if absolute_errors_on_valid else None
    )
    mae_with_invalid = (
        sum(absolute_errors_with_invalid) / len(absolute_errors_with_invalid) if absolute_errors_with_invalid else None
    )

    return {
        "num_samples": len(samples),
        "num_results": len(result_by_id),
        "accuracy": _safe_div(correct_count, len(samples)),
        "invalid_response_rate": _safe_div(invalid_count, len(samples)),
        "accuracy_by_env": {
            env: _safe_div(stats["correct"], stats["total"])
            for env, stats in sorted(accuracy_by_env.items())
        },
        "accuracy_by_task_id": {
            task_id: _safe_div(stats["correct"], stats["total"])
            for task_id, stats in sorted(accuracy_by_task_id.items())
        },
        "accuracy_by_label": {
            label: _safe_div(stats["correct"], stats["total"])
            for label, stats in sorted(accuracy_by_label.items())
        },
        "accuracy_by_label_index": {
            label_index: _safe_div(stats["correct"], stats["total"])
            for label_index, stats in sorted(accuracy_by_label_index.items())
        },
        "confusion_matrix": {
            gold_label: {predicted: count for predicted, count in sorted(predictions.items())}
            for gold_label, predictions in sorted(confusion_matrix.items())
        },
        "off_by_one_accuracy": _safe_div(off_by_one_count, len(samples)),
        "mean_absolute_label_index_error_on_valid": mae_on_valid,
        "mean_absolute_label_index_error_with_invalid": mae_with_invalid,
        "invalid_error_handling": "INVALID predictions count as wrong for exact/off_by_one; for MAE-with-invalid they use each sample's maximum possible label-index error.",
    }
