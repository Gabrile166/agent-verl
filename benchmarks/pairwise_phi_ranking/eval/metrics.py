"""Metric computation helpers for pairwise benchmark evaluation outputs."""

from typing import Any, Dict, List

from ..core.schema import BenchmarkSample


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    text = str(label).strip().upper()
    return text


def _is_correct(result: Dict[str, Any]) -> bool:
    predicted = _normalize_label(result.get("predicted_label"))
    if predicted == "INVALID":
        return False

    if "correct" in result:
        return bool(result.get("correct"))

    ground_truth = _normalize_label(result.get("ground_truth_label"))
    if predicted in {"A", "B"} and ground_truth in {"A", "B"}:
        return predicted == ground_truth
    return False


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_pairwise_choice_acc(results: List[Dict[str, Any]]) -> float:
    """Compute pairwise choice accuracy; INVALID predictions are counted as wrong."""
    if not results:
        return 0.0

    correct_count = sum(1 for result in results if _is_correct(result))
    return correct_count / len(results)


def compute_subset_accuracies(results: List[Dict[str, Any]], samples: List[BenchmarkSample]) -> Dict[str, Any]:
    """Compute easy/hard/same-length subset accuracy and per-task-type accuracy."""
    result_by_id = {
        str(result.get("sample_id", "")): result
        for result in results
        if str(result.get("sample_id", ""))
    }

    easy_correct = 0
    easy_total = 0
    hard_correct = 0
    hard_total = 0
    same_len_correct = 0
    same_len_total = 0
    by_task_type: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        if result is None:
            continue

        is_correct = _is_correct(result)

        if sample.difficulty == "easy":
            easy_total += 1
            if is_correct:
                easy_correct += 1

        if sample.difficulty == "hard":
            hard_total += 1
            if is_correct:
                hard_correct += 1

        if abs(len(sample.trajectory_a.steps) - len(sample.trajectory_b.steps)) <= 2:
            same_len_total += 1
            if is_correct:
                same_len_correct += 1

        task_type = sample.task_type or "unknown"
        if task_type not in by_task_type:
            by_task_type[task_type] = {"correct": 0, "total": 0}
        by_task_type[task_type]["total"] += 1
        if is_correct:
            by_task_type[task_type]["correct"] += 1

    acc_by_tasktype = {
        task_type: _safe_div(stats["correct"], stats["total"])
        for task_type, stats in by_task_type.items()
    }

    return {
        "acc_easy": _safe_div(easy_correct, easy_total),
        "acc_hard": _safe_div(hard_correct, hard_total),
        "same_length_subset_acc": _safe_div(same_len_correct, same_len_total),
        "acc_by_tasktype": acc_by_tasktype,
    }


def compute_invalid_response_rate(results: List[Dict[str, Any]]) -> float:
    """Compute how often model outputs are unparsable INVALID responses."""
    if not results:
        return 0.0

    invalid_count = sum(1 for result in results if _normalize_label(result.get("predicted_label")) == "INVALID")
    return invalid_count / len(results)


def compute_accuracy_by_pair_type(results: List[dict], samples: List[BenchmarkSample]) -> Dict[str, float]:
    """Compute accuracy grouped by pair_type and by subset/non-subset split."""
    result_by_id = {
        str(item.get("sample_id", "")): item
        for item in results
        if str(item.get("sample_id", ""))
    }

    counters = {
        "expert_prefix": {"correct": 0, "total": 0},
        "expert_fork": {"correct": 0, "total": 0},
        "perturbed_replay": {"correct": 0, "total": 0},
        "subset_pairs": {"correct": 0, "total": 0},
        "non_subset_pairs": {"correct": 0, "total": 0},
    }

    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        if result is None:
            continue

        is_correct = _is_correct(result)

        if sample.pair_type in counters:
            counters[sample.pair_type]["total"] += 1
            if is_correct:
                counters[sample.pair_type]["correct"] += 1

        subset_key = "subset_pairs" if sample.is_subset_pair else "non_subset_pairs"
        counters[subset_key]["total"] += 1
        if is_correct:
            counters[subset_key]["correct"] += 1

    return {
        "acc_expert_prefix": _safe_div(counters["expert_prefix"]["correct"], counters["expert_prefix"]["total"]),
        "acc_expert_fork": _safe_div(counters["expert_fork"]["correct"], counters["expert_fork"]["total"]),
        "acc_perturbed_replay": _safe_div(
            counters["perturbed_replay"]["correct"], counters["perturbed_replay"]["total"]
        ),
        "acc_subset_pairs": _safe_div(counters["subset_pairs"]["correct"], counters["subset_pairs"]["total"]),
        "acc_non_subset_pairs": _safe_div(
            counters["non_subset_pairs"]["correct"], counters["non_subset_pairs"]["total"]
        ),
    }
