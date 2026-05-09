"""Verify rule-based process scorers for ALFWorld and SciWorld."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.pairwise_phi_ranking.build.build_expert_fork import SciWorldForkHelper
from benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld import AlfworldForkHelper
from benchmarks.pairwise_phi_ranking.core.process_scorers import (
    score_alfworld_process,
    score_process,
    score_sciworld_process,
)


def summarize_text(text: str, limit: int = 100) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def unique_preserve_order(values: Sequence[int]) -> List[int]:
    seen = set()
    result: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def choose_alfworld_cutoffs(total_steps: int) -> List[int]:
    if total_steps <= 0:
        return [0]
    candidates = [
        0,
        1,
        total_steps // 3,
        (2 * total_steps) // 3,
        max(total_steps - 1, 0),
        total_steps,
    ]
    clipped = [min(max(value, 0), total_steps) for value in candidates]
    return unique_preserve_order(clipped)


def _extract_actions_from_steps(steps: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(step.get("action", "")) for step in steps]


def verify_alfworld(args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 80)
    print("ALFWorld process scorer verification")
    print("=" * 80)

    helper = AlfworldForkHelper(config_path=args.alfworld_config)
    selected_games = list(helper.game_files[: args.alfworld_num_games])
    if args.alfworld_game_file:
        selected_games = list(args.alfworld_game_file)

    results: List[Dict[str, Any]] = []
    usable = True

    for game_file in selected_games:
        game_name = os.path.basename(os.path.dirname(game_file))
        print(f"\n[ALFWorld] game={game_name}")
        expert = helper.collect_expert_trajectory(game_file, max_steps=args.alfworld_max_steps)
        if not expert:
            usable = False
            print("  failed: could not collect expert trajectory")
            results.append({"game": game_name, "usable": False, "error": "collect_expert_trajectory failed"})
            continue

        total_expert_steps = int(expert.get("num_steps", 0) or 0)
        won = bool(expert.get("won", False))
        print(f"  total_expert_steps={total_expert_steps}, expert_won={won}")
        if total_expert_steps <= 0 or not won:
            usable = False
            error = "expert did not finish from start state"
            print(f"  failed: {error}")
            results.append({"game": game_name, "usable": False, "error": error})
            continue

        monotonic_ok = True
        initial_consistent = True
        final_consistent = True
        model_takeover_ok = True
        prefix_steps = expert["steps"]
        previous_score = None
        cutoff_results: List[Dict[str, Any]] = []

        for cutoff in choose_alfworld_cutoffs(total_expert_steps):
            prefix = prefix_steps[:cutoff]
            score = score_alfworld_process(
                {
                    "game_file": game_file,
                    "helper": helper,
                    "max_steps": max(args.alfworld_max_steps, total_expert_steps + 20),
                    "initial_step_budget": max(args.alfworld_max_steps, total_expert_steps + 20),
                    "remaining_step_budget": max(total_expert_steps * 2, 20),
                },
                prefix,
            )
            cutoff_results.append({"cutoff": cutoff, **score})
            current_score = score["process_score"]
            print(
                "  "
                f"cutoff={cutoff:>3} | process_score={current_score} | "
                f"remaining={score['remaining_expert_steps']} | total={score['total_expert_steps']} | "
                f"done={score['done']} | success={score['success']} | error={score['error']}"
            )
            if cutoff == 0:
                initial_consistent = (
                    score["remaining_expert_steps"] == score["total_expert_steps"]
                    and score["process_score"] == 0.0
                )
            if cutoff == total_expert_steps:
                final_consistent = (
                    score["remaining_expert_steps"] == 0
                    and score["process_score"] == 1.0
                    and score["done"]
                    and score["success"]
                )
            if current_score is not None and previous_score is not None and current_score + 1e-9 < previous_score:
                monotonic_ok = False
            if current_score is not None:
                previous_score = current_score

        if total_expert_steps >= 4:
            model_like_actions = _extract_actions_from_steps(prefix_steps[: max(1, total_expert_steps // 2)])
            if model_like_actions:
                model_like_actions[-1] = "look"
            model_takeover = score_alfworld_process(
                {
                    "game_file": game_file,
                    "helper": helper,
                    "max_steps": max(args.alfworld_max_steps, total_expert_steps + 20),
                    "initial_step_budget": max(args.alfworld_max_steps, total_expert_steps + 20),
                    "remaining_step_budget": max(total_expert_steps * 2, 20),
                },
                model_like_actions,
            )
            print(
                "  model_takeover | "
                f"process_score={model_takeover['process_score']} | remaining={model_takeover['remaining_expert_steps']} | "
                f"done={model_takeover['done']} | success={model_takeover['success']} | error={model_takeover['error']}"
            )
            model_takeover_ok = model_takeover["success"] or (model_takeover["process_score"] is None and model_takeover["error"] is not None)

        if not monotonic_ok:
            usable = False
            print("  warning: process_score is not monotonic on expert prefixes")
        else:
            print("  monotonic_check=pass")

        if not initial_consistent:
            usable = False
            print("  warning: initial state is not normalized to remaining == total and score == 0")
        if not final_consistent:
            usable = False
            print("  warning: final state is not normalized to remaining == 0 and score == 1")
        if not model_takeover_ok:
            usable = False
            print("  warning: model-like intermediate state cannot be handled cleanly by expert takeover")

        results.append(
            {
                "game": game_name,
                "usable": won and monotonic_ok and initial_consistent and final_consistent and model_takeover_ok,
                "cutoffs": cutoff_results,
                "monotonic_ok": monotonic_ok,
                "initial_consistent": initial_consistent,
                "final_consistent": final_consistent,
                "model_takeover_ok": model_takeover_ok,
            }
        )

    return {
        "env": "alfworld",
        "usable": usable,
        "results": results,
    }


def verify_sciworld(args: argparse.Namespace) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("SciWorld process scorer verification")
    print("=" * 80)

    trace_helper = SciWorldForkHelper(env_step_limit=args.sciworld_env_step_limit)
    score_helper = SciWorldForkHelper(env_step_limit=args.sciworld_env_step_limit)

    selected_tasks = list(trace_helper.task_names[: args.sciworld_num_tasks])
    if args.sciworld_task:
        selected_tasks = list(args.sciworld_task)

    task_results: List[Dict[str, Any]] = []
    usable = True

    for task_name in selected_tasks:
        print(f"\n[SciWorld] task={task_name} variation={args.sciworld_variation}")
        env = trace_helper.env
        env.load(task_name, args.sciworld_variation, args.sciworld_simplification, generateGoldPath=True)
        observation, info = env.reset()
        initial_raw = float(info.get("score", 0.0))
        initial_result = score_sciworld_process(
            {
                "task_name": task_name,
                "variation": args.sciworld_variation,
                "simplification_str": args.sciworld_simplification,
                "helper": score_helper,
                "generate_gold_path": True,
            },
            [],
        )
        print(
            "  initial | "
            f"obs={summarize_text(observation)} | raw_score={initial_raw} | "
            f"process_score={initial_result['process_score']} | done={initial_result['done']} | "
            f"success={initial_result['success']}"
        )

        env.load(task_name, args.sciworld_variation, args.sciworld_simplification, generateGoldPath=False)
        probe_obs, probe_info = env.reset()
        probe_raw_before = float(probe_info.get("score", 0.0))
        probe_obs_after, probe_reward, probe_done, probe_info_after = env.step("look around")
        probe_raw_after = float(probe_info_after.get("score", 0.0))
        print(
            "  no_progress_probe | "
            f"action=look around | before={probe_raw_before} | after={probe_raw_after} | "
            f"done={probe_done}"
        )

        env.load(task_name, args.sciworld_variation, args.sciworld_simplification, generateGoldPath=True)
        observation, info = env.reset()
        gold_actions = list(env.get_gold_action_sequence())
        prefix_actions: List[str] = []
        previous_raw = float(info.get("score", 0.0))
        score_decreased = False
        final_result = None
        printed_steps = 0

        for raw_step_index, action in enumerate(gold_actions):
            observation, reward, done, info = env.step(action)
            prefix_actions.append(action)
            raw_score = float(info.get("score", 0.0))
            scorer_result = score_process(
                "sciworld",
                {
                    "task_name": task_name,
                    "variation": args.sciworld_variation,
                    "simplification_str": args.sciworld_simplification,
                    "helper": score_helper,
                    "generate_gold_path": True,
                },
                prefix_actions,
            )
            if raw_score + 1e-9 < previous_raw:
                score_decreased = True
            previous_raw = raw_score
            printed_steps += 1
            final_result = scorer_result
            print(
                "  "
                f"step={raw_step_index:>3} | action={action!r} | "
                f"obs={summarize_text(observation)} | raw_score={raw_score:>6.1f} | "
                f"process_score={scorer_result['process_score']} | done={scorer_result['done']} | "
                f"success={scorer_result['success']}"
            )
            if done or printed_steps >= args.sciworld_max_steps:
                break

        final_raw = None if final_result is None else final_result["details"].get("raw_score")
        final_done = bool(final_result and final_result["done"])
        final_success = bool(final_result and final_result["success"])
        initial_near_zero = initial_raw <= 1.0
        final_near_one = final_raw is not None and final_raw >= 99.0
        no_progress_ok = abs(probe_raw_after - probe_raw_before) < 1e-9
        success_score_aligned = (not final_success) or (final_raw is not None and final_raw >= 99.0)
        score_done_aligned = (final_raw is None) or (final_raw < 99.0) or final_done
        task_usable = bool(initial_near_zero and final_near_one and no_progress_ok and not score_decreased and success_score_aligned and score_done_aligned)
        if not task_usable:
            usable = False

        print(
            "  summary | "
            f"initial_near_zero={initial_near_zero} | final_near_one={final_near_one} | "
            f"no_progress_ok={no_progress_ok} | score_decreased={score_decreased} | "
            f"final_done={final_done} | final_success={final_success} | "
            f"success_score_aligned={success_score_aligned} | score_done_aligned={score_done_aligned}"
        )

        task_results.append(
            {
                "task": task_name,
                "usable": task_usable,
                "initial_raw": initial_raw,
                "final_raw": final_raw,
                "initial_near_zero": initial_near_zero,
                "final_near_one": final_near_one,
                "no_progress_ok": no_progress_ok,
                "score_decreased": score_decreased,
                "final_done": final_done,
                "final_success": final_success,
                "success_score_aligned": success_score_aligned,
                "score_done_aligned": score_done_aligned,
                "trajectory_len": printed_steps,
            }
        )

    trace_helper.close()
    score_helper.close()
    return {
        "env": "sciworld",
        "usable": usable,
        "results": task_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ALFWorld/SciWorld rule-based process scorers")
    parser.add_argument("--env", choices=["alfworld", "sciworld", "both"], default="both")
    parser.add_argument("--alfworld-num-games", type=int, default=3)
    parser.add_argument("--alfworld-max-steps", type=int, default=80)
    parser.add_argument("--alfworld-config", type=str, default=None)
    parser.add_argument("--alfworld-game-file", action="append", default=[])
    parser.add_argument("--sciworld-num-tasks", type=int, default=5)
    parser.add_argument("--sciworld-max-steps", type=int, default=80)
    parser.add_argument("--sciworld-variation", type=int, default=0)
    parser.add_argument("--sciworld-simplification", type=str, default="easy")
    parser.add_argument("--sciworld-env-step-limit", type=int, default=100)
    parser.add_argument("--sciworld-task", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict[str, Any]] = []

    if args.env in {"alfworld", "both"}:
        results.append(verify_alfworld(args))
    if args.env in {"sciworld", "both"}:
        results.append(verify_sciworld(args))

    print("\n" + "=" * 80)
    print("Verification summary")
    print("=" * 80)
    for result in results:
        print(f"- {result['env']}: usable={result['usable']}")


if __name__ == "__main__":
    main()
