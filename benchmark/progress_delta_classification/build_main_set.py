"""Build an optimized main evaluation set for progress delta classification."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

for candidate_root in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (candidate_root / "benchmarks").exists() and str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))
        break

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from benchmarks.common_eval.io import dump_json, load_benchmark_file
    from benchmarks.progress_delta_classification.core.schema import ProgressDeltaSample
else:
    from ..common_eval.io import dump_json, load_benchmark_file
    from .core.schema import ProgressDeltaSample

DELTA_POS_THRESHOLD = 0.15
DELTA_NEG_THRESHOLD = -0.15
DELTA_SAME_THRESHOLD = 0.03
RANDOM_MODE = "random_decrease_mining"
SLICE_MODE = "slice_existing_trajectory"
PAIRWISE_REGRESSION_MODE = "paired_branch_regression"
DEFAULT_OPTIMIZATION_VERSION = "progress_delta_main_v2"

WEAK_ACTION_EXACT = {
    "look",
    "look around",
    "inventory",
    "help",
    "examine",
}
WEAK_ACTION_PREFIXES = (
    "wait",
    "examine ",
)
NAVIGATION_PREFIXES = (
    "go to ",
    "open door to ",
    "close door to ",
)
FAILURE_RESULT_FRAGMENTS = (
    "no known action matches that input",
    "ambiguous request",
    "already open",
    "already closed",
    "can't",
    "cannot",
    "nothing happens",
    "available commands:",
)
DESTRUCTIVE_ACTION_FRAGMENTS = (
    "reset task",
    "drop ",
    "deactivate ",
    "turn off ",
    "switch off ",
    "close ",
    "move ",
    "put ",
)
DESTRUCTIVE_RESULT_FRAGMENTS = (
    "you reset the goal progress",
    "is now deactivated",
    "is now closed",
    "you move the",
    "you put the",
    "you drop the",
)
OBVIOUS_NEGATIVE_ACTION_FRAGMENTS = (
    "reset task",
    "drop ",
    "deactivate ",
    "turn off ",
    "switch off ",
)
OBVIOUS_NEGATIVE_RESULT_FRAGMENTS = (
    "you reset the goal progress",
    "is now deactivated",
    "you drop the",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized main sets for progress delta classification")
    parser.add_argument("--alfworld-candidates", required=True, help="Path to ALFWorld candidate JSONL file")
    parser.add_argument("--sciworld-candidates", required=True, help="Path to SciWorld candidate JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory for optimized JSON outputs")
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=0,
        help="Cap the selected count per label when > 0; otherwise use the largest balanced target.",
    )
    parser.add_argument(
        "--optimization-version",
        default=DEFAULT_OPTIMIZATION_VERSION,
        help="Version tag written into sample details and selection summary.",
    )
    parser.add_argument(
        "--allow-decrease-generation-modes",
        default="",
        help="Comma-separated generation modes allowed for decrease samples; empty means allow all.",
    )
    parser.add_argument(
        "--drop-obvious-negative-cues",
        action="store_true",
        help="Drop decrease samples with explicit reset/deactivate/drop cues.",
    )
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _canonical_label(delta: float) -> Optional[str]:
    if delta >= DELTA_POS_THRESHOLD:
        return "increase"
    if delta <= DELTA_NEG_THRESHOLD:
        return "decrease"
    if abs(delta) <= DELTA_SAME_THRESHOLD:
        return "same"
    return None


def _is_weak_action(action: str) -> bool:
    normalized = _normalize_text(action)
    if normalized in WEAK_ACTION_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in WEAK_ACTION_PREFIXES)


def _is_navigation_action(action: str) -> bool:
    normalized = _normalize_text(action)
    return any(normalized.startswith(prefix) for prefix in NAVIGATION_PREFIXES)


def _is_failed_result(result: str) -> bool:
    normalized = _normalize_text(result)
    return any(fragment in normalized for fragment in FAILURE_RESULT_FRAGMENTS)


def _is_repeated_observation(step: Dict[str, Any]) -> bool:
    before = _normalize_text(step.get("obs_before"))
    after = _normalize_text(step.get("obs_after"))
    return bool(before) and before == after


def _has_destructive_signal(step: Dict[str, Any]) -> bool:
    action = _normalize_text(step.get("action"))
    result = _normalize_text(step.get("obs_after"))
    if any(fragment in action for fragment in DESTRUCTIVE_ACTION_FRAGMENTS):
        return True
    return any(fragment in result for fragment in DESTRUCTIVE_RESULT_FRAGMENTS)


def _sample_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    steps = list(row.get("added_steps") or [])
    weak_all = all(_is_weak_action(str(step.get("action", ""))) for step in steps)
    failed_all = all(_is_failed_result(str(step.get("obs_after", ""))) for step in steps)
    repeated_all = all(_is_repeated_observation(step) for step in steps)
    navigation_all = all(_is_navigation_action(str(step.get("action", ""))) for step in steps)
    destructive_any = any(_has_destructive_signal(step) for step in steps)
    return {
        "weak_all": weak_all,
        "failed_all": failed_all,
        "repeated_all": repeated_all,
        "navigation_all": navigation_all,
        "destructive_any": destructive_any,
    }


def _has_obvious_negative_cue(step: Dict[str, Any]) -> bool:
    action = _normalize_text(step.get("action"))
    result = _normalize_text(step.get("obs_after"))
    if any(fragment in action for fragment in OBVIOUS_NEGATIVE_ACTION_FRAGMENTS):
        return True
    return any(fragment in result for fragment in OBVIOUS_NEGATIVE_RESULT_FRAGMENTS)


def _passes_rule_filters(
    row: Dict[str, Any],
    relabeled: str,
    *,
    allowed_decrease_modes: Optional[Sequence[str]] = None,
    drop_obvious_negative_cues: bool = False,
) -> Tuple[bool, str]:
    flags = _sample_flags(row)
    generation_mode = str(row.get("details", {}).get("generation_mode") or "")
    delta = float(row.get("progress_delta", 0.0))
    allowed_mode_set = {mode for mode in (allowed_decrease_modes or []) if mode}

    if relabeled in {"increase", "decrease"} and flags["weak_all"]:
        return False, "weak_added_steps"
    if relabeled in {"increase", "decrease"} and flags["failed_all"]:
        return False, "failed_added_steps"
    if relabeled in {"increase", "decrease"} and flags["repeated_all"]:
        return False, "repeated_observation_only"

    if relabeled == "decrease" and allowed_mode_set and generation_mode not in allowed_mode_set:
        return False, "decrease_generation_mode_not_allowed"
    if relabeled == "decrease" and drop_obvious_negative_cues:
        if any(_has_obvious_negative_cue(step) for step in list(row.get("added_steps") or [])):
            return False, "obvious_negative_cue"

    if relabeled == "decrease" and generation_mode == RANDOM_MODE:
        if flags["failed_all"] or flags["weak_all"]:
            return False, "random_decrease_is_weak_or_failed"
        if flags["navigation_all"] and not flags["destructive_any"]:
            return False, "random_decrease_navigation_only"
        if not flags["destructive_any"] and abs(delta) < 0.5:
            return False, "random_decrease_not_clear_enough"

    return True, "keep"


def _sort_key(row: Dict[str, Any], label: str) -> Tuple[Any, ...]:
    details = row.get("details", {})
    generation_mode = str(details.get("generation_mode") or "")
    delta = float(row.get("progress_delta", 0.0))
    added_len = int(details.get("added_len") or len(row.get("added_steps") or []))
    prefix_len = int(details.get("prefix_len") or len(row.get("trajectory_prefix") or []))
    source_id = str(details.get("source_sample_id") or row.get("sample_id") or "")
    random_rank = 1 if generation_mode == RANDOM_MODE else 0
    if label == "same":
        delta_rank = abs(delta)
    else:
        delta_rank = -abs(delta)
    return (random_rank, delta_rank, added_len, prefix_len, source_id, str(row.get("sample_id") or ""))


def _annotate_sample(
    row: Dict[str, Any],
    optimized_label: str,
    filter_reason: str,
    optimization_version: str,
) -> Dict[str, Any]:
    enriched = copy.deepcopy(row)
    details = dict(enriched.get("details") or {})
    details["original_label"] = str(row.get("label") or "")
    details["optimized_label"] = optimized_label
    details["optimization_version"] = optimization_version
    details["filter_reason"] = filter_reason
    enriched["details"] = details
    enriched["label"] = optimized_label
    ProgressDeltaSample.from_dict(enriched)
    return enriched


def _load_candidates(
    path: str,
    *,
    optimization_version: str,
    allowed_decrease_modes: Optional[Sequence[str]] = None,
    drop_obvious_negative_cues: bool = False,
) -> List[Dict[str, Any]]:
    rows = load_benchmark_file(path)
    candidates: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            ProgressDeltaSample.from_dict(row)
        except Exception as exc:
            raise ValueError(f"Invalid progress delta sample #{index} in {path}: {exc}") from exc

        relabeled = _canonical_label(float(row.get("progress_delta", 0.0)))
        if relabeled is None:
            continue

        keep, reason = _passes_rule_filters(
            row,
            relabeled,
            allowed_decrease_modes=allowed_decrease_modes,
            drop_obvious_negative_cues=drop_obvious_negative_cues,
        )
        if not keep:
            continue

        candidates.append(_annotate_sample(row, relabeled, reason, optimization_version=optimization_version))
    return candidates


def _select_balanced_rows(
    rows: Sequence[Dict[str, Any]],
    env_name: str,
    target_cap: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label") or "")].append(row)

    label_counts = {label: len(by_label.get(label, [])) for label in ("increase", "decrease", "same")}
    available_target = min(label_counts.values()) if label_counts else 0
    target = min(available_target, target_cap) if target_cap > 0 else available_target
    if target <= 0:
        raise ValueError(f"No balanced set can be built for env={env_name}: {label_counts}")

    selected: List[Dict[str, Any]] = []
    selection_summary: Dict[str, Any] = {
        "env": env_name,
        "available_by_label": label_counts,
        "available_balanced_target_per_label": available_target,
        "requested_target_per_label": target_cap if target_cap > 0 else None,
        "target_per_label": target,
        "selected_by_label": {},
        "selected_by_generation_mode": {},
        "ratio_relaxed_labels": [],
    }

    for label in ("increase", "decrease", "same"):
        candidates = sorted(by_label[label], key=lambda row: _sort_key(row, label))
        slice_rows = [row for row in candidates if str(row.get("details", {}).get("generation_mode") or "") == SLICE_MODE]
        random_rows = [row for row in candidates if str(row.get("details", {}).get("generation_mode") or "") == RANDOM_MODE]
        pairwise_rows = [
            row
            for row in candidates
            if str(row.get("details", {}).get("generation_mode") or "") == PAIRWISE_REGRESSION_MODE
        ]
        other_rows = [
            row
            for row in candidates
            if str(row.get("details", {}).get("generation_mode") or "") not in {SLICE_MODE, RANDOM_MODE, PAIRWISE_REGRESSION_MODE}
        ]

        chosen: List[Dict[str, Any]] = []
        seen_sources = set()
        if label == "decrease":
            preferred_groups = [pairwise_rows, slice_rows, other_rows, random_rows]
        else:
            preferred_groups = [slice_rows, pairwise_rows, other_rows, random_rows]

        def pick_from(group: Sequence[Dict[str, Any]]) -> None:
            for row in group:
                if len(chosen) >= target:
                    break
                source_key = str(row.get("details", {}).get("source_sample_id") or row.get("sample_id") or "")
                if source_key and source_key in seen_sources:
                    continue
                chosen.append(row)
                if source_key:
                    seen_sources.add(source_key)

        for group in preferred_groups:
            pick_from(group)

        if len(chosen) < target:
            for group in preferred_groups:
                for row in group:
                    if len(chosen) >= target:
                        break
                    if row in chosen:
                        continue
                    chosen.append(row)

        if len(chosen) != target:
            raise ValueError(f"Failed to select {target} samples for env={env_name}, label={label}")

        random_count = sum(
            1
            for row in chosen
            if str(row.get("details", {}).get("generation_mode") or "") == RANDOM_MODE
        )
        random_cap = int(target * 0.3)
        if target > 0:
            random_cap = max(random_cap, 1)
        if random_count > random_cap:
            selection_summary["ratio_relaxed_labels"].append(label)

        mode_counts = Counter(str(row.get("details", {}).get("generation_mode") or "unknown") for row in chosen)
        selection_summary["selected_by_label"][label] = len(chosen)
        selection_summary["selected_by_generation_mode"][label] = dict(sorted(mode_counts.items()))
        selected.extend(chosen)

    selected = sorted(selected, key=lambda row: str(row.get("sample_id") or ""))
    return selected, selection_summary


def _write_json_array(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_decrease_modes = [
        part.strip() for part in str(args.allow_decrease_generation_modes or "").split(",") if part.strip()
    ]
    optimization_version = str(args.optimization_version or DEFAULT_OPTIMIZATION_VERSION)
    all_candidates = {
        "alfworld": _load_candidates(
            args.alfworld_candidates,
            optimization_version=optimization_version,
            allowed_decrease_modes=allowed_decrease_modes,
            drop_obvious_negative_cues=bool(args.drop_obvious_negative_cues),
        ),
        "sciworld": _load_candidates(
            args.sciworld_candidates,
            optimization_version=optimization_version,
            allowed_decrease_modes=allowed_decrease_modes,
            drop_obvious_negative_cues=bool(args.drop_obvious_negative_cues),
        ),
    }

    selected_by_env: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Any] = {}
    for env_name, rows in all_candidates.items():
        selected_rows, env_summary = _select_balanced_rows(
            rows,
            env_name=env_name,
            target_cap=max(0, int(args.target_per_label or 0)),
        )
        selected_by_env[env_name] = selected_rows
        summaries[env_name] = env_summary

    alfworld_path = output_dir / "progress_delta_alfworld_balanced_main.json"
    sciworld_path = output_dir / "progress_delta_sciworld_balanced_main.json"
    _write_json_array(alfworld_path, selected_by_env["alfworld"])
    _write_json_array(sciworld_path, selected_by_env["sciworld"])

    summary_payload = {
        "optimization_version": optimization_version,
        "delta_thresholds": {
            "increase_ge": DELTA_POS_THRESHOLD,
            "decrease_le": DELTA_NEG_THRESHOLD,
            "same_abs_le": DELTA_SAME_THRESHOLD,
        },
        "inputs": {
            "alfworld_candidates": args.alfworld_candidates,
            "sciworld_candidates": args.sciworld_candidates,
        },
        "selection_policy": {
            "allow_decrease_generation_modes": allowed_decrease_modes,
            "drop_obvious_negative_cues": bool(args.drop_obvious_negative_cues),
        },
        "outputs": {
            "alfworld": str(alfworld_path),
            "sciworld": str(sciworld_path),
        },
        "env_summaries": summaries,
    }
    dump_json(str(output_dir / "selection_summary.json"), summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
