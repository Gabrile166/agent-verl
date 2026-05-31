"""Run the heuristic potential baseline on benchmark JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.common_eval.heuristic_potential import HeuristicPotentialSolver
from benchmark.common_eval.io import (
    JsonlPredictionWriter,
    dump_json,
    ensure_output_dir,
    load_benchmark_file,
    load_existing_predictions,
    prepare_prediction_file,
)
from benchmark.pairwise_phi_ranking.core.schema import BenchmarkSample
from benchmark.pairwise_phi_ranking.eval.metrics import (
    compute_accuracy_by_pair_type,
    compute_invalid_response_rate,
    compute_pairwise_choice_acc,
    compute_subset_accuracies,
)
from benchmark.progress_delta_classification.core.schema import ProgressDeltaSample
from benchmark.progress_delta_classification.eval.metrics import summarize_results as summarize_delta_results
from benchmark.trajectory_milestone_localization.core.schema import TrajectoryMilestoneSample
from benchmark.trajectory_milestone_localization.eval.metrics import summarize_results as summarize_milestone_results


BENCHMARK_AUTO = "auto"
BENCHMARK_PAIRWISE = "pairwise_phi_ranking"
BENCHMARK_DELTA = "progress_delta_classification"
BENCHMARK_MILESTONE = "trajectory_milestone_localization"
BENCHMARK_CHOICES = [BENCHMARK_AUTO, BENCHMARK_PAIRWISE, BENCHMARK_DELTA, BENCHMARK_MILESTONE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a heuristic Phi(s) baseline on benchmark samples")
    parser.add_argument("--input", required=True, help="Benchmark JSON or JSONL file path")
    parser.add_argument("--output-dir", "--output_folder", dest="output_dir", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARK_CHOICES, default=BENCHMARK_AUTO)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--same-threshold", type=float, default=0.05)
    parser.add_argument("--milestone-threshold", type=float, default=0.55)
    parser.add_argument(
        "--profile",
        choices=["coarse", "generic", "task_aware"],
        default="coarse",
        help=(
            "coarse is the leakage-resistant default; generic keeps semantic task parsing; "
            "task_aware keeps stronger environment/task-specific rules for ablation."
        ),
    )
    parser.add_argument("--print-every", "--print_every", dest="print_every", type=int, default=50)
    return parser.parse_args()


def _detect_benchmark(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot detect benchmark type from an empty input file")
    row = rows[0]
    if "trajectory_a" in row and "trajectory_b" in row:
        return BENCHMARK_PAIRWISE
    if "trajectory_prefix" in row and "added_steps" in row:
        return BENCHMARK_DELTA
    if "milestones" in row and "trajectory" in row:
        return BENCHMARK_MILESTONE
    raise ValueError("Could not detect benchmark type from the first row")


def _env_from_pairwise(sample: BenchmarkSample) -> str:
    track = str(sample.track or "").lower()
    if "alfworld" in track:
        return "alfworld"
    if "sciworld" in track:
        return "sciworld"
    return track


def _load_pairwise(rows: Sequence[Dict[str, Any]]) -> List[BenchmarkSample]:
    return [BenchmarkSample.from_dict(row) for row in rows]


def _load_delta(rows: Sequence[Dict[str, Any]]) -> List[ProgressDeltaSample]:
    return [ProgressDeltaSample.from_dict(row) for row in rows]


def _load_milestone(rows: Sequence[Dict[str, Any]]) -> List[TrajectoryMilestoneSample]:
    return [TrajectoryMilestoneSample.from_dict(row) for row in rows]


def _pairwise_result(sample: BenchmarkSample, solver: HeuristicPotentialSolver) -> Dict[str, Any]:
    env = _env_from_pairwise(sample)
    predicted_label, score_a, score_b = solver.compare_pair(
        sample.task_description,
        sample.trajectory_a,
        sample.trajectory_b,
        env=env,
    )
    return {
        "sample_id": sample.sample_id,
        "benchmark": BENCHMARK_PAIRWISE,
        "predicted_label": predicted_label,
        "ground_truth_label": sample.label,
        "correct": predicted_label == sample.label,
        "pair_type": sample.pair_type,
        "difficulty": sample.difficulty,
        "track": sample.track,
        "task_type": sample.task_type,
        "is_subset_pair": sample.is_subset_pair,
        "uses_expert_branch": sample.uses_expert_branch,
        "num_steps_a": len(sample.trajectory_a.steps),
        "num_steps_b": len(sample.trajectory_b.steps),
        "heuristic_score_a": score_a.phi,
        "heuristic_score_b": score_b.phi,
        "heuristic_stage_a": score_a.stage,
        "heuristic_stage_b": score_b.stage,
        "raw_response": f"score_a={score_a.phi:.4f}; score_b={score_b.phi:.4f}",
        "extractor_output": predicted_label,
        "error": None,
        "metadata": {
            "heuristic_profile": solver.profile,
            "score_a": score_a.to_dict(),
            "score_b": score_b.to_dict(),
        },
    }


def _delta_result(sample: ProgressDeltaSample, solver: HeuristicPotentialSolver) -> Dict[str, Any]:
    predicted_label, before, after, delta = solver.classify_delta(
        sample.task_description,
        sample.trajectory_prefix,
        sample.added_steps,
        env=sample.env,
    )
    return {
        "sample_id": sample.sample_id,
        "benchmark": BENCHMARK_DELTA,
        "env": sample.env,
        "predicted_label": predicted_label,
        "ground_truth_label": sample.label,
        "correct": predicted_label == sample.label,
        "raw_response": f"phi_before={before.phi:.4f}; phi_after={after.phi:.4f}; delta={delta:.4f}",
        "extractor_output": predicted_label,
        "prompt": "",
        "prompt_tokens": None,
        "latency_sec": None,
        "error": None,
        "metadata": {
            "heuristic_profile": solver.profile,
            "heuristic_phi_before": before.phi,
            "heuristic_phi_after": after.phi,
            "heuristic_delta": delta,
            "heuristic_before": before.to_dict(),
            "heuristic_after": after.to_dict(),
            "progress_before": sample.progress_before,
            "progress_after": sample.progress_after,
            "progress_delta": sample.progress_delta,
            "generation_mode": str(sample.details.get("generation_mode") or "") or None,
        },
    }


def _milestone_result(sample: TrajectoryMilestoneSample, solver: HeuristicPotentialSolver) -> Dict[str, Any]:
    predicted_label, predicted_index, phi, potential, scored_milestones = solver.localize_milestone(
        sample.task_description,
        sample.trajectory,
        sample.milestones,
        env=sample.env,
    )
    if not predicted_label:
        predicted_label = "INVALID"
        predicted_index_value = None
    else:
        predicted_index_value = predicted_index
    off_by_one = predicted_index_value is not None and abs(int(predicted_index_value) - sample.label_index) <= 1
    return {
        "sample_id": sample.sample_id,
        "benchmark": BENCHMARK_MILESTONE,
        "env": sample.env,
        "predicted_label": predicted_label,
        "ground_truth_label": sample.label,
        "predicted_label_index": predicted_index_value,
        "ground_truth_label_index": sample.label_index,
        "correct": predicted_label == sample.label,
        "off_by_one": bool(off_by_one),
        "raw_response": f"milestone={predicted_label}; phi={phi:.4f}",
        "extractor_output": predicted_label,
        "prompt": "",
        "prompt_tokens": None,
        "latency_sec": None,
        "error": None,
        "metadata": {
            "heuristic_profile": solver.profile,
            "task_id": sample.task_id or None,
            "num_milestones": len(sample.milestones),
            "heuristic_phi": phi,
            "heuristic_potential": potential.to_dict(),
            "milestone_scores": scored_milestones,
            "label_definition": str(sample.details.get("label_definition") or "") or None,
            "generation_mode": str(sample.details.get("generation_mode") or "") or None,
        },
    }


def _summarize_pairwise(results: Sequence[Dict[str, Any]], samples: Sequence[BenchmarkSample]) -> Dict[str, Any]:
    result_list = list(results)
    sample_list = list(samples)
    metrics: Dict[str, Any] = {
        "num_samples": len(sample_list),
        "num_results": len(result_list),
        "pairwise_choice_acc": compute_pairwise_choice_acc(result_list),
        "invalid_response_rate": compute_invalid_response_rate(result_list),
    }
    metrics.update(compute_subset_accuracies(result_list, sample_list))
    metrics.update(compute_accuracy_by_pair_type(result_list, sample_list))
    return metrics


def _evaluate_rows(
    rows: Sequence[Dict[str, Any]],
    benchmark_name: str,
    prediction_path: str,
    resume: bool,
    solver: HeuristicPotentialSolver,
    print_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepare_prediction_file(prediction_path, resume)
    existing = load_existing_predictions(prediction_path) if resume else {}
    writer = JsonlPredictionWriter(prediction_path)
    started_at = time.time()
    results: List[Dict[str, Any]] = list(existing.values())

    try:
        if benchmark_name == BENCHMARK_PAIRWISE:
            samples = _load_pairwise(rows)
            for idx, sample in enumerate(samples, start=1):
                if sample.sample_id in existing:
                    continue
                result = _pairwise_result(sample, solver)
                writer.write(result)
                results.append(result)
                _maybe_print_progress(idx, len(samples), print_every)
            metrics = _summarize_pairwise(_order_results(results, samples), samples)
        elif benchmark_name == BENCHMARK_DELTA:
            samples = _load_delta(rows)
            for idx, sample in enumerate(samples, start=1):
                if sample.sample_id in existing:
                    continue
                result = _delta_result(sample, solver)
                writer.write(result)
                results.append(result)
                _maybe_print_progress(idx, len(samples), print_every)
            metrics = summarize_delta_results(_order_results(results, samples), samples)
        elif benchmark_name == BENCHMARK_MILESTONE:
            samples = _load_milestone(rows)
            for idx, sample in enumerate(samples, start=1):
                if sample.sample_id in existing:
                    continue
                result = _milestone_result(sample, solver)
                writer.write(result)
                results.append(result)
                _maybe_print_progress(idx, len(samples), print_every)
            metrics = summarize_milestone_results(_order_results(results, samples), samples)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
    finally:
        writer.close()

    metrics = {
        "benchmark": benchmark_name,
        "heuristic_baseline": f"text_env_potential_{solver.profile}_v2",
        "heuristic_profile": solver.profile,
        "runtime_sec": time.time() - started_at,
        **metrics,
    }
    return results, metrics


def _order_results(results: Sequence[Dict[str, Any]], samples: Sequence[Any]) -> List[Dict[str, Any]]:
    by_id = {str(row.get("sample_id", "")): row for row in results if str(row.get("sample_id", ""))}
    return [by_id[sample.sample_id] for sample in samples if sample.sample_id in by_id]


def _maybe_print_progress(index: int, total: int, print_every: int) -> None:
    if print_every > 0 and (index % print_every == 0 or index == total):
        print(f"[progress] completed {index}/{total}")


def dump_summary(output_dir: str, input_path: str, metrics_path: str, prediction_path: str, args: argparse.Namespace) -> str:
    summary_path = os.path.join(output_dir, "predictions_summary.json")
    payload = {
        "benchmark": args.benchmark,
        "input": input_path,
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "predictions_path": prediction_path,
        "timestamp": int(time.time()),
        "config": vars(args),
    }
    dump_json(summary_path, payload)
    return summary_path


def main() -> None:
    args = parse_args()
    rows = load_benchmark_file(args.input, max_samples=args.max_samples)
    benchmark_name = _detect_benchmark(rows) if args.benchmark == BENCHMARK_AUTO else args.benchmark
    prediction_path, metrics_path, summary_path = ensure_output_dir(args.output_dir)
    solver = HeuristicPotentialSolver(
        same_threshold=args.same_threshold,
        milestone_threshold=args.milestone_threshold,
        profile=args.profile,
    )

    _, metrics = _evaluate_rows(
        rows=rows,
        benchmark_name=benchmark_name,
        prediction_path=prediction_path,
        resume=args.resume,
        solver=solver,
        print_every=args.print_every,
    )
    metrics_payload = {
        "input": args.input,
        "output_dir": args.output_dir,
        **metrics,
    }
    dump_json(metrics_path, metrics_payload)
    dump_summary(args.output_dir, args.input, metrics_path, prediction_path, args)
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[outputs] predictions={prediction_path}")
    print(f"[outputs] metrics={metrics_path}")
    print(f"[outputs] summary={summary_path}")


if __name__ == "__main__":
    main()
