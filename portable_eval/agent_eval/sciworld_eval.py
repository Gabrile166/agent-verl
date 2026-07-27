from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .api import OpenAICompatibleClient
from .core import EpisodeRecorder, build_agent_prompt, extract_action


SYSTEM_PROMPT = (
    "You are a ScienceWorld agent. Complete the task using valid text commands. "
    "Return exactly one action enclosed by <action> and </action>."
)

# This compactly reproduces the exact 1,684-pair set in
# agent_system/.../variations_idx/L1_idx.json["test"]. The original order is
# shuffled; deterministic task/variation order makes resume portable.
L1_TEST_RANGES: dict[int, tuple[int, int]] = {
    0: (21, 29),
    1: (21, 29),
    2: (24, 31),
    3: (27, 35),
    4: (27, 35),
    5: (225, 299),
    6: (225, 299),
    7: (225, 299),
    8: (225, 299),
    9: (21, 29),
    10: (93, 125),
    11: (93, 125),
    12: (9, 13),
    13: (6, 9),
    14: (126, 167),
    15: (1038, 1385),
    16: (120, 161),
    17: (93, 124),
    18: (93, 124),
    19: (93, 124),
    20: (327, 435),
    21: (225, 299),
    22: (21, 29),
    23: (90, 119),
    24: (360, 479),
    25: (15, 19),
    26: (15, 19),
    27: (675, 899),
    28: (450, 599),
}


def iter_l1_test_variations() -> Iterator[tuple[int, int]]:
    for task_id, (start, end) in L1_TEST_RANGES.items():
        for variation_id in range(start, end + 1):
            yield task_id, variation_id


def run_sciworld_l1(
    *,
    client: OpenAICompatibleClient,
    output_dir: Path,
    start_index: int,
    limit: int | None,
    max_steps: int,
    history_steps: int,
    resume: bool,
    save_transcripts: bool,
    simplifications: str,
    jar_path: str | None,
) -> dict[str, Any]:
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:
        raise RuntimeError(
            "ScienceWorld is not installed. Run: pip install -e '.[all]'"
        ) from exc

    if jar_path:
        env = ScienceWorldEnv("", jar_path, envStepLimit=max_steps)
    else:
        env = ScienceWorldEnv("", envStepLimit=max_steps)

    pairs = list(iter_l1_test_variations())[start_index:]
    if limit is not None:
        pairs = pairs[:limit]
    recorder = EpisodeRecorder(output_dir, resume=resume)

    try:
        task_names = env.get_task_names()
        for offset, (task_id, variation_id) in enumerate(pairs):
            split_index = start_index + offset
            episode_id = f"sciworld_l1:{task_id}:{variation_id}"
            if recorder.contains(episode_id):
                continue

            task_name = task_names[task_id]
            env.load(task_name, variation_id, simplifications)
            observation, _ = env.reset()
            task = env.get_task_description()
            history: list[dict[str, str]] = []
            transcript: list[dict[str, Any]] = []
            invalid_actions = 0
            api_total_tokens = 0
            final_score = 0

            for step in range(1, max_steps + 1):
                available = _scienceworld_actions(env)
                prompt = build_agent_prompt(
                    environment="ScienceWorld L1",
                    task=str(task),
                    observation=str(observation),
                    available_actions=available,
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

                next_observation, _, done, info = env.step(action)
                final_score = int(info.get("score", 0))
                history.append({"observation": str(observation), "action": action})
                if save_transcripts:
                    transcript.append(
                        {
                            "step": step,
                            "observation": observation,
                            "model_output": result.content,
                            "action": action,
                            "score": final_score,
                        }
                    )
                observation = next_observation
                if done:
                    break

            success = final_score >= 100
            episode = {
                "episode_id": episode_id,
                "environment": "sciworld_l1",
                "split_index": split_index,
                "task_id": task_id,
                "task_name": task_name,
                "variation_id": variation_id,
                "success": success,
                "score": final_score,
                "steps": len(history),
                "invalid_actions": invalid_actions,
                "api_total_tokens": api_total_tokens,
                "max_steps": max_steps,
                "history_steps": history_steps,
                "simplifications": simplifications,
            }
            if save_transcripts:
                episode["transcript"] = transcript
            recorder.append(episode)
            print(
                f"[ScienceWorld L1] {split_index + 1}/1684 task={task_id} "
                f"variation={variation_id} score={final_score} steps={len(history)}"
            )
    finally:
        env.close()

    return recorder.write_summary()


def _scienceworld_actions(env: Any) -> list[str]:
    actions = [str(item) for item in env.get_possible_actions()]
    objects = [str(item) for item in env.get_possible_objects()]
    return actions + [f"Objects for OBJ: {', '.join(objects)}"]
