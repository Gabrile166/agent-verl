"""Evaluate trajectory milestone localization predictions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.trajectory_milestone_localization.core.schema import TrajectoryMilestoneSample


_LABEL_TOKEN_PATTERN = re.compile(r"\b([A-Z])\b")



def _load_dataset(path: Path) -> List[TrajectoryMilestoneSample]:
    rows: List[TrajectoryMilestoneSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(TrajectoryMilestoneSample.from_dict(json.loads(line)))
    return rows



def _load_predictions(path: Path, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(json.loads(line)))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows



def _normalize_label_value(value: Any) -> str:
    return str(value or "").strip().upper()



def _extract_prediction(value: Any, allowed_ids: Sequence[str]) -> Tuple[str, Dict[str, Any]]:
    normalized = _normalize_label_value(value)
    metadata = {
        "normalized_text": normalized,
        "matched_ids": [],
        "multiple_ids_detected": False,
    }
    if not normalized:
        return "INVALID", metadata
    if normalized in allowed_ids:
        metadata["matched_ids"] = [normalized]
        return normalized, metadata

    matches = [token for token in _LABEL_TOKEN_PATTERN.findall(normalized) if token in allowed_ids]
    unique_matches: List[str] = []
    for token in matches:
        if token not in unique_matches:
            unique_matches.append(token)
    metadata["matched_ids"] = unique_matches
    metadata["multiple_ids_detected"] = len(unique_matches) > 1
    if len(unique_matches) == 1:
        return unique_matches[0], metadata
    return "INVALID", metadata



def _resolve_prediction_record(
    prediction: Dict[str, Any],
    allowed_ids: Sequence[str],
    prediction_field: str,
    raw_response_field: str,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    predicted, metadata = _extract_prediction(prediction.get(prediction_field), allowed_ids)
    if predicted != "INVALID":
        return predicted, prediction_field, metadata

    if raw_response_field:
        predicted, raw_metadata = _extract_prediction(prediction.get(raw_response_field), allowed_ids)
        if predicted != "INVALID":
            return predicted, raw_response_field, raw_metadata
        if raw_metadata.get("multiple_ids_detected"):
            return "INVALID", raw_response_field, raw_metadata
    return "INVALID", None, metadata



def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator



def evaluate_predictions(
    samples: Sequence[TrajectoryMilestoneSample],
    predictions: Sequence[Dict[str, Any]],
    prediction_field: str,
    raw_response_field: str,
) -> Dict[str, Any]:
    prediction_by_id = {
        str(item.get("sample_id", "")): item
        for item in predictions
        if str(item.get("sample_id", ""))
    }

    correct = 0
    total = 0
    invalid = 0
    missing = 0
    multi_id_invalid = 0
    by_task_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_milestone_idx: Dict[int, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    details: List[Dict[str, Any]] = []

    for sample in samples:
        total += 1
        prediction = prediction_by_id.get(sample.sample_id)
        task_type = sample.task_id or str(sample.details.get("alfworld_task_type", "unknown") or "unknown")
        by_task_type[task_type]["total"] += 1
        by_milestone_idx[sample.label_index]["total"] += 1

        if prediction is None:
            missing += 1
            invalid += 1
            confusion[sample.label]["MISSING"] += 1
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "ground_truth_label": sample.label,
                    "predicted_label": "INVALID",
                    "matched_field": None,
                    "correct": False,
                    "missing_prediction": True,
                    "multiple_ids_detected": False,
                    "matched_ids": [],
                }
            )
            continue

        allowed_ids = [milestone.id for milestone in sample.milestones]
        predicted_label, matched_field, parse_meta = _resolve_prediction_record(
            prediction=prediction,
            allowed_ids=allowed_ids,
            prediction_field=prediction_field,
            raw_response_field=raw_response_field,
        )
        is_correct = predicted_label == sample.label
        if predicted_label == "INVALID":
            invalid += 1
            if parse_meta.get("multiple_ids_detected"):
                multi_id_invalid += 1
        if is_correct:
            correct += 1
            by_task_type[task_type]["correct"] += 1
            by_milestone_idx[sample.label_index]["correct"] += 1
        confusion[sample.label][predicted_label] += 1
        details.append(
            {
                "sample_id": sample.sample_id,
                "ground_truth_label": sample.label,
                "predicted_label": predicted_label,
                "matched_field": matched_field,
                "correct": is_correct,
                "missing_prediction": False,
                "multiple_ids_detected": bool(parse_meta.get("multiple_ids_detected", False)),
                "matched_ids": list(parse_meta.get("matched_ids", [])),
                "normalized_text": parse_meta.get("normalized_text", ""),
            }
        )

    accuracy_by_task_type = {
        task_type: _safe_div(values["correct"], values["total"])
        for task_type, values in sorted(by_task_type.items())
    }
    accuracy_by_milestone_idx = {
        str(index): _safe_div(values["correct"], values["total"])
        for index, values in sorted(by_milestone_idx.items())
    }
    confusion_dict = {
        ground_truth: dict(sorted(counter.items()))
        for ground_truth, counter in sorted(confusion.items())
    }

    return {
        "num_samples": total,
        "num_predictions": len(prediction_by_id),
        "num_correct": correct,
        "num_invalid_or_missing": invalid,
        "num_missing_predictions": missing,
        "num_multi_id_invalid": multi_id_invalid,
        "overall_accuracy": _safe_div(correct, total),
        "invalid_response_rate": _safe_div(invalid, total),
        "accuracy_by_task_type": accuracy_by_task_type,
        "accuracy_by_milestone_idx": accuracy_by_milestone_idx,
        "confusion_matrix": confusion_dict,
        "details": details,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trajectory milestone localization predictions")
    parser.add_argument("--input", required=True, help="Ground-truth milestone localization dataset jsonl")
    parser.add_argument("--predictions", required=True, help="Prediction jsonl with sample_id and predicted labels")
    parser.add_argument("--output", default="", help="Optional metrics json path")
    parser.add_argument("--prediction-field", default="predicted_label", help="Field used for normalized model label")
    parser.add_argument("--raw-response-field", default="raw_response", help="Fallback field used for label extraction")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N predictions when > 0")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    dataset = _load_dataset(Path(args.input))
    predictions = _load_predictions(Path(args.predictions), max_rows=args.max_samples)
    metrics = evaluate_predictions(
        samples=dataset,
        predictions=predictions,
        prediction_field=args.prediction_field,
        raw_response_field=args.raw_response_field,
    )

    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
