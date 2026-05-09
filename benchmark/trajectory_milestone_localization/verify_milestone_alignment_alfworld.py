"""Inspect ALFWorld milestone transitions against solver subgoal changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld import AlfworldForkHelper
from benchmarks.trajectory_milestone_localization.build.build_milestone_localization_alfworld import _trace_trajectory_subgoals



def _build_game_map(helper: AlfworldForkHelper) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for game_file in helper.game_files:
        instance_id = os.path.basename(os.path.dirname(game_file))
        mapping.setdefault(instance_id, game_file)
    return mapping



def _load_task_instance_ids(path: Path, limit: int) -> List[str]:
    task_ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row.get("task_type", ""))
            if not task_id or task_id in task_ids:
                continue
            task_ids.append(task_id)
            if limit > 0 and len(task_ids) >= limit:
                break
    return task_ids



def verify_alignment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    helper = AlfworldForkHelper(config_path=args.alfworld_config)
    game_map = _build_game_map(helper)
    task_ids = _load_task_instance_ids(Path(args.source_jsonl), limit=args.num_tasks)
    reports: List[Dict[str, Any]] = []

    for task_id in task_ids:
        game_file = game_map.get(task_id)
        if not game_file:
            reports.append({"task_instance_id": task_id, "error": "missing_game_file"})
            continue

        expert = helper.collect_expert_trajectory(game_file, max_steps=args.alfworld_max_steps)
        if not expert or not expert.get("steps"):
            reports.append({"task_instance_id": task_id, "error": "failed_to_collect_expert"})
            continue

        trace = _trace_trajectory_subgoals(
            helper=helper,
            game_file=game_file,
            trajectory_steps=expert["steps"],
            max_steps=max(args.alfworld_max_steps, len(expert["steps"]) + 10),
        )
        if trace.get("error"):
            reports.append({"task_instance_id": task_id, "error": trace["error"]})
            continue

        prefix_records = trace.get("prefix_records", [])
        changed_points = [
            {
                "prefix_len": int(record["prefix_len"]),
                "action": str(record["action"]),
                "subgoal_idx_before": int(record["subgoal_idx_before"]),
                "subgoal_idx_after": int(record["subgoal_idx_after"]),
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
                "task_instance_id": task_id,
                "alfworld_task_type": trace.get("alfworld_task_type"),
                "task_description": trace.get("task_description"),
                "num_subgoals": trace.get("num_subgoals"),
                "milestones": [milestone.to_dict() for milestone in trace.get("milestones", [])],
                "changed_points": changed_points,
                "num_changed_points": len(changed_points),
                "error": None,
            }
        )
    return reports



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ALFWorld milestone text against solver transitions")
    parser.add_argument("--source-jsonl", required=True, help="Pairwise ALFWorld jsonl used to pick task instances")
    parser.add_argument("--output", required=True, help="Verification output json path")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of distinct task instances to inspect")
    parser.add_argument("--alfworld-config", type=str, default=None)
    parser.add_argument("--alfworld-max-steps", type=int, default=80)
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
