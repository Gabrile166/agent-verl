"""Inspect SciWorld milestone transitions against rule-based score jumps."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.pairwise_phi_ranking.core.schema import BenchmarkSample
from benchmarks.trajectory_milestone_localization.build.build_milestone_localization_sciworld import (
    MilestoneBuilder,
    _step_to_dict,
    _trace_trajectory_scores,
)


_VARIATION_PATTERN = re.compile(r"(?:^|_)v(\d+)(?:_|$)")


def _parse_variation(sample_id: str) -> int | None:
    match = _VARIATION_PATTERN.search(str(sample_id or ""))
    if not match:
        return None
    return int(match.group(1))


def _load_task_variations(path: Path, limit: int) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = BenchmarkSample.from_dict(json.loads(line))
            if not str(row.track or "").lower().startswith("sciworld"):
                continue
            task_name = str(row.task_type or "").strip()
            variation = _parse_variation(row.sample_id)
            if not task_name or variation is None:
                continue
            key = (task_name, variation)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if limit > 0 and len(pairs) >= limit:
                break
    return pairs


def verify_alignment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    builder = MilestoneBuilder(args)
    task_variations = _load_task_variations(Path(args.source_jsonl), limit=args.num_tasks)
    reports: List[Dict[str, Any]] = []

    try:
        for task_name, variation in task_variations:
            try:
                expert_trace = builder.get_expert_trace(
                    task_name=task_name,
                    variation=variation,
                    simplification_str=args.sciworld_simplification,
                )
                trace = _trace_trajectory_scores(
                    builder=builder,
                    task_name=task_name,
                    variation=variation,
                    simplification_str=args.sciworld_simplification,
                    trajectory_steps=[_step_to_dict(step) for step in expert_trace.get("expert_steps", [])],
                )
                if trace.get("error"):
                    reports.append(
                        {
                            "task_name": task_name,
                            "variation": variation,
                            "error": trace["error"],
                        }
                    )
                    continue

                prefix_records = trace.get("prefix_records", [])
                changed_points = [
                    {
                        "prefix_len": int(record["prefix_len"]),
                        "action": str(record["action"]),
                        "raw_score_before": float(record["raw_score_before"]),
                        "raw_score_after": float(record["raw_score_after"]),
                        "label_before": str(record["label_id_before"]),
                        "label_after": str(record["label_id_after"]),
                        "milestone_description_after": str(record["milestone_description_after"]),
                        "observation_after": str(record["observation_after"]),
                    }
                    for record in prefix_records
                    if int(record["label_index_after"]) != int(record["label_index_before"])
                ]
                reports.append(
                    {
                        "task_name": task_name,
                        "variation": variation,
                        "task_description": expert_trace.get("task_description"),
                        "initial_raw_score": expert_trace.get("initial_raw_score"),
                        "expert_final_raw_score": expert_trace.get("expert_final_raw_score"),
                        "milestones": [milestone.to_dict() for milestone in trace.get("milestones", [])],
                        "score_events": [
                            {
                                "step_index": int(event["step_index"]),
                                "action": str(event["action"]),
                                "raw_score_before": float(event["raw_score_before"]),
                                "raw_score_after": float(event["raw_score_after"]),
                                "observation_after": str(event["obs_after"]),
                            }
                            for event in expert_trace.get("score_events", [])
                        ],
                        "changed_points": changed_points,
                        "num_score_events": len(expert_trace.get("score_events", [])),
                        "num_changed_points": len(changed_points),
                        "error": None,
                    }
                )
            except Exception as exc:
                reports.append(
                    {
                        "task_name": task_name,
                        "variation": variation,
                        "error": str(exc),
                    }
                )
    finally:
        builder.close()
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SciWorld milestone text against rule-based score jumps")
    parser.add_argument("--source-jsonl", required=True, help="Pairwise SciWorld jsonl used to pick task variations")
    parser.add_argument("--output", required=True, help="Verification output json path")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of distinct task/variation pairs to inspect")
    parser.add_argument("--sciworld-simplification", type=str, default="easy")
    parser.add_argument("--sciworld-env-step-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = verify_alignment(args)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {"reports": reports, "num_reports": len(reports)}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
