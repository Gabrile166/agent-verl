from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .api import OpenAICompatibleClient
from .core import EpisodeRecorder, build_agent_prompt, extract_action


SYSTEM_PROMPT = (
    "You are a text-environment agent. Use only information in the current prompt. "
    "Return exactly one action enclosed by <action> and </action>."
)


def run_alfworld_ood(
    *,
    client: OpenAICompatibleClient,
    output_dir: Path,
    start_index: int,
    limit: int | None,
    max_steps: int,
    history_steps: int,
    seed: int,
    resume: bool,
    save_transcripts: bool,
) -> dict[str, Any]:
    try:
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld is not installed. Run: pip install -e '.[all]'"
        ) from exc

    data_dir = Path(
        os.path.expandvars(
            os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
        )
    ).expanduser()
    if not (data_dir / "json_2.1.1" / "valid_unseen").is_dir():
        raise RuntimeError(
            f"ALFWorld OOD data not found under {data_dir}. "
            "Set ALFWORLD_DATA and run alfworld-download."
        )

    config_path = Path(__file__).with_name("alfworld_config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps
    config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
    catalog = get_environment("AlfredTWEnv")(
        config,
        train_eval="eval_out_of_distribution",
    )
    game_files = sorted(str(path) for path in catalog.game_files)
    selected = game_files[start_index:]
    if limit is not None:
        selected = selected[:limit]

    recorder = EpisodeRecorder(output_dir, resume=resume)
    for offset, game_file in enumerate(selected):
        game_index = start_index + offset
        episode_id = f"alfworld_ood:{game_index}"
        if recorder.contains(episode_id):
            continue

        catalog.game_files = [game_file]
        env = catalog.init_env(batch_size=1)
        if hasattr(env, "seed"):
            env.seed(seed + game_index)
        try:
            observations, infos = env.reset()
            observation = str(observations[0])
            task = _extract_task(observation)
            available = _first(infos.get("admissible_commands", [[]]), [])
            history: list[dict[str, str]] = []
            transcript: list[dict[str, Any]] = []
            invalid_actions = 0
            api_total_tokens = 0
            success = False
            final_score = 0.0

            for step in range(1, max_steps + 1):
                prompt = build_agent_prompt(
                    environment="ALFWorld OOD",
                    task=task,
                    observation=observation,
                    available_actions=[str(action) for action in available if action != "help"],
                    history=history,
                    history_steps=history_steps,
                    step_number=step,
                )
                result = client.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                api_total_tokens += result.usage["total_tokens"]
                action = extract_action(result.content)
                if action is None:
                    invalid_actions += 1
                    action = "__invalid_action__"

                next_observations, scores, dones, infos = env.step([action])
                next_observation = str(next_observations[0])
                final_score = float(_first(scores, 0.0))
                success = bool(_first(infos.get("won", [False]), False))
                done = bool(_first(dones, False))

                history.append({"observation": observation, "action": action})
                if save_transcripts:
                    transcript.append(
                        {
                            "step": step,
                            "observation": observation,
                            "model_output": result.content,
                            "action": action,
                            "reward_or_score": final_score,
                        }
                    )
                observation = next_observation
                available = _first(infos.get("admissible_commands", [[]]), [])
                if done or success:
                    break

            episode = {
                "episode_id": episode_id,
                "environment": "alfworld_ood",
                "game_file": str(Path(game_file).relative_to(data_dir)),
                "success": success,
                "score": final_score,
                "steps": len(history),
                "invalid_actions": invalid_actions,
                "api_total_tokens": api_total_tokens,
                "max_steps": max_steps,
                "history_steps": history_steps,
            }
            if save_transcripts:
                episode["transcript"] = transcript
            recorder.append(episode)
            print(
                f"[ALFWorld OOD] {game_index + 1}/{len(game_files)} "
                f"success={int(success)} steps={len(history)}"
            )
        finally:
            env.close()

    return recorder.write_summary()


def _extract_task(observation: str) -> str:
    marker = "Your task is to:"
    if marker not in observation:
        raise ValueError("ALFWorld task description was not found in the initial observation")
    return observation.split(marker, 1)[1].strip()


def _first(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    try:
        if getattr(value, "ndim", 0) > 0:
            return value[0]
    except (IndexError, TypeError):
        return default
    return value if value is not None else default
