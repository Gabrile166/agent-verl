#!/usr/bin/env python3
"""Recompute pairwise accuracy with deterministic rule-based answer extraction."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPLICIT_PATTERNS = [
    re.compile(r"answer\s*[:：]\s*([ab])\b", re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:：]?\s*([ab])\b", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s*([ab])\b", re.IGNORECASE),
    re.compile(r"my\s+answer\s+is\s*([ab])\b", re.IGNORECASE),
    re.compile(r"i\s+choose\s*([ab])\b", re.IGNORECASE),
    re.compile(r"choose\s*([ab])\b", re.IGNORECASE),
    re.compile(r"pick\s*([ab])\b", re.IGNORECASE),
    re.compile(r"trajectory\s*([ab])\s+(?:has|made|is|shows|appears)", re.IGNORECASE),
]

FINAL_LINE_PATTERNS = [
    re.compile(r"^([ab])$", re.IGNORECASE),
    re.compile(r"^(?:answer|final answer)\s*[:：]\s*([ab])$", re.IGNORECASE),
    re.compile(r"^(?:the answer is|i choose|choose|pick)\s*([ab])$", re.IGNORECASE),
]

LEGACY_ANSWER_PATTERN = re.compile(r"answer\s*[:：]\s*([ab])\b", re.IGNORECASE)
LEGACY_TOKEN_PATTERN = re.compile(r"\b([ab])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute pairwise accuracy with rule-based answer extraction")
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl")
    parser.add_argument(
        "--mode",
        choices=["strict", "legacy", "both"],
        default="both",
        help="strict uses conservative rules; legacy reproduces the old regex fallback; both prints both.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum mismatch examples to keep in the summary",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the summary json",
    )
    return parser.parse_args()



def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def extract_strict(response: str) -> Tuple[str, str]:
    if not response:
        return "INVALID", "empty"

    for pattern in EXPLICIT_PATTERNS:
        matches = list(pattern.finditer(response))
        if matches:
            return matches[-1].group(1).upper(), "explicit_pattern"

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        final_line = lines[-1]
        for pattern in FINAL_LINE_PATTERNS:
            match = pattern.fullmatch(final_line)
            if match:
                return match.group(1).upper(), "final_line"

    return "INVALID", "invalid"



def extract_legacy(response: str) -> Tuple[str, str]:
    if not response:
        return "INVALID", "empty"

    answer_matches = list(LEGACY_ANSWER_PATTERN.finditer(response))
    if answer_matches:
        return answer_matches[-1].group(1).upper(), "explicit_answer"

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines and re.fullmatch(r"[ab]", lines[-1], flags=re.IGNORECASE):
        return lines[-1].upper(), "final_line"

    token_matches = list(LEGACY_TOKEN_PATTERN.finditer(response))
    if token_matches:
        return token_matches[-1].group(1).upper(), "last_token"

    return "INVALID", "invalid"



def compute_accuracy(records: Iterable[Dict[str, Any]], label_key: str) -> float:
    total = 0
    correct = 0
    for record in records:
        total += 1
        predicted = normalize_label(record.get(label_key))
        ground_truth = normalize_label(record.get("ground_truth_label"))
        if predicted in {"A", "B"} and predicted == ground_truth:
            correct += 1
    return (correct / total) if total else 0.0



def collect_summary(rows: List[Dict[str, Any]], max_examples: int) -> Dict[str, Any]:
    strict_method_counter: Counter[str] = Counter()
    legacy_method_counter: Counter[str] = Counter()
    strict_invalid = 0
    legacy_invalid = 0
    strict_vs_original_examples: List[Dict[str, Any]] = []
    legacy_vs_original_examples: List[Dict[str, Any]] = []

    for row in rows:
        raw_response = str(row.get("raw_response") or "")
        strict_label, strict_method = extract_strict(raw_response)
        legacy_label, legacy_method = extract_legacy(raw_response)
        row["rule_label_strict"] = strict_label
        row["rule_method_strict"] = strict_method
        row["rule_label_legacy"] = legacy_label
        row["rule_method_legacy"] = legacy_method

        strict_method_counter[strict_method] += 1
        legacy_method_counter[legacy_method] += 1
        if strict_label == "INVALID":
            strict_invalid += 1
        if legacy_label == "INVALID":
            legacy_invalid += 1

        original_label = normalize_label(row.get("predicted_label"))
        if strict_label != original_label and len(strict_vs_original_examples) < max_examples:
            strict_vs_original_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "ground_truth_label": row.get("ground_truth_label"),
                    "original_predicted_label": row.get("predicted_label"),
                    "rule_label_strict": strict_label,
                    "rule_method_strict": strict_method,
                    "raw_response_tail": raw_response[-300:],
                }
            )
        if legacy_label != original_label and len(legacy_vs_original_examples) < max_examples:
            legacy_vs_original_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "ground_truth_label": row.get("ground_truth_label"),
                    "original_predicted_label": row.get("predicted_label"),
                    "rule_label_legacy": legacy_label,
                    "rule_method_legacy": legacy_method,
                    "raw_response_tail": raw_response[-300:],
                }
            )

    total = len(rows)
    summary = {
        "num_samples": total,
        "original_accuracy_from_predicted_label": compute_accuracy(rows, "predicted_label"),
        "strict_rule_accuracy": compute_accuracy(rows, "rule_label_strict"),
        "legacy_rule_accuracy": compute_accuracy(rows, "rule_label_legacy"),
        "strict_invalid_rate": (strict_invalid / total) if total else 0.0,
        "legacy_invalid_rate": (legacy_invalid / total) if total else 0.0,
        "strict_rule_method_counts": dict(strict_method_counter),
        "legacy_rule_method_counts": dict(legacy_method_counter),
        "strict_rule_diff_examples": strict_vs_original_examples,
        "legacy_rule_diff_examples": legacy_vs_original_examples,
    }
    return summary



def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    summary = collect_summary(rows, max_examples=max(0, args.max_examples))

    if args.mode == "strict":
        output = {
            "num_samples": summary["num_samples"],
            "original_accuracy_from_predicted_label": summary["original_accuracy_from_predicted_label"],
            "strict_rule_accuracy": summary["strict_rule_accuracy"],
            "strict_invalid_rate": summary["strict_invalid_rate"],
            "strict_rule_method_counts": summary["strict_rule_method_counts"],
            "strict_rule_diff_examples": summary["strict_rule_diff_examples"],
        }
    elif args.mode == "legacy":
        output = {
            "num_samples": summary["num_samples"],
            "original_accuracy_from_predicted_label": summary["original_accuracy_from_predicted_label"],
            "legacy_rule_accuracy": summary["legacy_rule_accuracy"],
            "legacy_invalid_rate": summary["legacy_invalid_rate"],
            "legacy_rule_method_counts": summary["legacy_rule_method_counts"],
            "legacy_rule_diff_examples": summary["legacy_rule_diff_examples"],
        }
    else:
        output = summary

    output["input_file"] = str(input_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
