"""Rule-based process scorers for ALFWorld and SciWorld."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


_ALFWORLD_TOTAL_STEPS_CACHE: Dict[str, int] = {}


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_prefix_actions(trajectory_prefix: Optional[Sequence[Any]]) -> List[str]:
    if not trajectory_prefix:
        return []

    actions: List[str] = []
    for step in trajectory_prefix:
        if isinstance(step, str):
            actions.append(step)
            continue
        if isinstance(step, dict):
            if "action" not in step:
                raise ValueError("Each trajectory step dict must contain an 'action' field")
            actions.append(str(step["action"]))
            continue
        if hasattr(step, "action"):
            actions.append(str(step.action))
            continue
        raise TypeError(f"Unsupported trajectory step type: {type(step)!r}")
    return actions


def _resolve_alfworld_context(task_or_env_state: Any) -> Dict[str, Any]:
    if isinstance(task_or_env_state, str):
        return {"game_file": task_or_env_state}
    if isinstance(task_or_env_state, dict):
        return dict(task_or_env_state)
    raise TypeError("ALFWorld task_or_env_state must be a game_file string or dict")


def _resolve_sciworld_context(task_or_env_state: Any) -> Dict[str, Any]:
    if isinstance(task_or_env_state, str):
        return {
            "task_name": task_or_env_state,
            "variation": 0,
            "simplification_str": "easy",
        }
    if isinstance(task_or_env_state, dict):
        return dict(task_or_env_state)
    raise TypeError("SciWorld task_or_env_state must be a task_name string or dict")


def _run_alfworld_expert_continuation(
    helper: Any,
    game_file: str,
    prefix_actions: Sequence[str],
    max_steps: int,
    continuation_budget: int,
) -> Dict[str, Any]:
    env_step_budget = max(max_steps, len(prefix_actions) + continuation_budget + 5)
    env = None
    try:
        env = helper._make_env(game_file, max_steps=env_step_budget, add_expert=True)
        current_obs, info = helper._reset_env(env)
        done = False
        replayed_prefix_steps = 0

        for step_index, action in enumerate(prefix_actions):
            try:
                current_obs, done, info = helper._step_env(env, action)
            except Exception as exc:
                return {
                    "remaining_expert_steps": None,
                    "done": False,
                    "success": False,
                    "replayed_prefix_steps": replayed_prefix_steps,
                    "error": f"failed while replaying prefix at step {step_index}: {exc}",
                }
            replayed_prefix_steps += 1
            if done:
                break

        success = bool(helper._unpack_info(info, "won", False))
        if success:
            return {
                "remaining_expert_steps": 0,
                "done": True,
                "success": True,
                "replayed_prefix_steps": replayed_prefix_steps,
                "error": None,
            }

        if done:
            return {
                "remaining_expert_steps": None,
                "done": True,
                "success": False,
                "replayed_prefix_steps": replayed_prefix_steps,
                "error": "trajectory prefix reached a terminal non-success state",
            }

        remaining_expert_steps = 0
        for _ in range(continuation_budget):
            valid_actions = list(helper._unpack_info(info, "admissible_commands", []) or [])
            expert_action = helper._get_expert_action(info, valid_actions)
            if not expert_action:
                return {
                    "remaining_expert_steps": remaining_expert_steps,
                    "done": False,
                    "success": False,
                    "replayed_prefix_steps": replayed_prefix_steps,
                    "error": "expert action unavailable at current state",
                }

            current_obs, done, info = helper._step_env(env, expert_action)
            remaining_expert_steps += 1
            success = bool(helper._unpack_info(info, "won", False))

            if success:
                return {
                    "remaining_expert_steps": remaining_expert_steps,
                    "done": True,
                    "success": True,
                    "replayed_prefix_steps": replayed_prefix_steps,
                    "error": None,
                }
            if done:
                return {
                    "remaining_expert_steps": remaining_expert_steps,
                    "done": True,
                    "success": False,
                    "replayed_prefix_steps": replayed_prefix_steps,
                    "error": "expert continuation reached a terminal non-success state",
                }

        return {
            "remaining_expert_steps": remaining_expert_steps,
            "done": False,
            "success": False,
            "replayed_prefix_steps": replayed_prefix_steps,
            "error": (
                "expert could not finish from the current state within the continuation budget "
                f"({continuation_budget} steps)"
            ),
        }
    except Exception as exc:
        return {
            "remaining_expert_steps": None,
            "done": False,
            "success": False,
            "replayed_prefix_steps": 0,
            "error": f"ALFWorld continuation failed: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def score_alfworld_process(
    task_or_env_state: Any,
    trajectory_prefix: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Score ALFWorld progress by counting expert steps remaining from the current replayed state."""
    ctx = _resolve_alfworld_context(task_or_env_state)
    game_file = ctx.get("game_file")
    if not game_file:
        return {
            "process_score": None,
            "remaining_expert_steps": None,
            "total_expert_steps": None,
            "done": False,
            "success": False,
            "error": "missing ALFWorld game_file",
        }

    try:
        prefix_actions = _normalize_prefix_actions(trajectory_prefix)
    except Exception as exc:
        return {
            "process_score": None,
            "remaining_expert_steps": None,
            "total_expert_steps": None,
            "done": False,
            "success": False,
            "error": f"failed to parse trajectory_prefix: {exc}",
        }

    if "helper" in ctx and ctx.get("helper") is not None:
        helper = ctx["helper"]
    else:
        from benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld import AlfworldForkHelper

        helper = AlfworldForkHelper(config_path=ctx.get("config_path"))
    max_steps = max(int(ctx.get("max_steps", 200) or 200), 1)

    initial_step_budget = max(int(ctx.get("initial_step_budget", max_steps) or max_steps), 1)
    total_expert_steps = ctx.get("total_expert_steps")
    total_run = None
    if total_expert_steps is None:
        total_expert_steps = _ALFWORLD_TOTAL_STEPS_CACHE.get(game_file)
    if total_expert_steps is None:
        total_run = _run_alfworld_expert_continuation(
            helper=helper,
            game_file=game_file,
            prefix_actions=[],
            max_steps=max_steps,
            continuation_budget=initial_step_budget,
        )
        if not total_run["success"]:
            return {
                "process_score": None,
                "remaining_expert_steps": None,
                "total_expert_steps": None,
                "done": False,
                "success": False,
                "error": total_run["error"],
            }
        total_expert_steps = total_run["remaining_expert_steps"]
        _ALFWORLD_TOTAL_STEPS_CACHE[game_file] = int(total_expert_steps)

    total_expert_steps = int(total_expert_steps)
    if total_expert_steps <= 0:
        return {
            "process_score": None,
            "remaining_expert_steps": None,
            "total_expert_steps": total_expert_steps,
            "done": False,
            "success": False,
            "error": "total_expert_steps must be greater than 0",
        }

    if not prefix_actions:
        return {
            "process_score": 0.0,
            "remaining_expert_steps": total_expert_steps,
            "total_expert_steps": total_expert_steps,
            "done": bool(total_run["done"]) if total_run is not None else True,
            "success": bool(total_run["success"]) if total_run is not None else True,
            "error": total_run["error"] if total_run is not None else None,
        }

    remaining_step_budget = max(
        int(ctx.get("remaining_step_budget", max(total_expert_steps * 2, 20)) or max(total_expert_steps * 2, 20)),
        1,
    )
    current_run = _run_alfworld_expert_continuation(
        helper=helper,
        game_file=game_file,
        prefix_actions=prefix_actions,
        max_steps=max_steps,
        continuation_budget=remaining_step_budget,
    )
    remaining_expert_steps = current_run["remaining_expert_steps"]
    if remaining_expert_steps is None:
        return {
            "process_score": None,
            "remaining_expert_steps": None,
            "total_expert_steps": total_expert_steps,
            "done": current_run["done"],
            "success": current_run["success"],
            "error": current_run["error"],
        }

    process_score = _clip_score(1.0 - (remaining_expert_steps / float(total_expert_steps)))
    return {
        "process_score": process_score,
        "remaining_expert_steps": remaining_expert_steps,
        "total_expert_steps": total_expert_steps,
        "done": current_run["done"],
        "success": current_run["success"],
        "error": current_run["error"],
    }


def _extract_sciworld_raw_score(info: Dict[str, Any]) -> Optional[float]:
    if "score" not in info:
        return None
    try:
        return float(info["score"])
    except (TypeError, ValueError):
        return None


def score_sciworld_process(
    task_or_env_state: Any,
    trajectory_prefix: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Score SciWorld progress using the environment's rule-based task score."""
    ctx = _resolve_sciworld_context(task_or_env_state)
    task_name = ctx.get("task_name")
    if not task_name:
        return {
            "process_score": None,
            "raw_score": None,
            "done": False,
            "success": False,
            "error": "missing SciWorld task_name",
        }

    try:
        prefix_actions = _normalize_prefix_actions(trajectory_prefix)
    except Exception as exc:
        return {
            "process_score": None,
            "raw_score": None,
            "done": False,
            "success": False,
            "error": f"failed to parse trajectory_prefix: {exc}",
        }

    created_helper = "helper" not in ctx or ctx.get("helper") is None
    if created_helper:
        from benchmarks.pairwise_phi_ranking.build.build_expert_fork import SciWorldForkHelper

        helper = SciWorldForkHelper(env_step_limit=int(ctx.get("env_step_limit", 100) or 100))
    else:
        helper = ctx["helper"]
    env = helper.env

    try:
        env.load(
            task_name,
            int(ctx.get("variation", 0) or 0),
            ctx.get("simplification_str", "easy") or "easy",
            generateGoldPath=bool(ctx.get("generate_gold_path", False)),
        )
        current_obs, info = env.reset()
        raw_score = _extract_sciworld_raw_score(info)
        done = False

        for step_index, action in enumerate(prefix_actions):
            try:
                current_obs, reward, done, info = env.step(action)
            except Exception as exc:
                return {
                    "process_score": None,
                    "raw_score": raw_score,
                    "done": False,
                    "success": False,
                    "error": f"failed while replaying prefix at step {step_index}: {exc}",
                }
            raw_score = _extract_sciworld_raw_score(info)
            if done:
                break

        if raw_score is None:
            return {
                "process_score": None,
                "raw_score": None,
                "done": bool(done),
                "success": False,
                "error": "SciWorld info['score'] is unavailable; cannot compute process score",
            }

        process_score = _clip_score(raw_score / 100.0)
        success = bool(done and raw_score > 0.0)
        return {
            "process_score": process_score,
            "raw_score": raw_score,
            "done": bool(done),
            "success": success,
            "error": None,
        }
    except Exception as exc:
        return {
            "process_score": None,
            "raw_score": None,
            "done": False,
            "success": False,
            "error": f"SciWorld scoring failed: {exc}",
        }
    finally:
        if created_helper:
            helper.close()


def score_process(
    env_name: str,
    task_or_env_state: Any,
    trajectory_prefix: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    env_key = str(env_name).strip().lower()
    if env_key == "alfworld":
        result = score_alfworld_process(task_or_env_state, trajectory_prefix)
    elif env_key == "sciworld":
        result = score_sciworld_process(task_or_env_state, trajectory_prefix)
    else:
        return {
            "env": env_key,
            "process_score": None,
            "done": False,
            "success": False,
            "details": {},
            "error": f"unsupported env_name: {env_name}",
        }

    details = dict(result)
    process_score = details.pop("process_score", None)
    done = bool(details.pop("done", False))
    success = bool(details.pop("success", False))
    error = details.pop("error", None)

    return {
        "env": env_key,
        "process_score": process_score,
        "done": done,
        "success": success,
        "details": details,
        "error": error,
    }


__all__ = [
    "score_alfworld_process",
    "score_sciworld_process",
    "score_process",
]
