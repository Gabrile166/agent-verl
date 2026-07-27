from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


def extract_action(model_output: str) -> str | None:
    """Return one explicitly tagged action; reject missing or ambiguous outputs."""
    matches = [match.strip() for match in _ACTION_PATTERN.findall(model_output)]
    if len(matches) != 1 or not matches[0]:
        return None
    return matches[0]


def build_agent_prompt(
    *,
    environment: str,
    task: str,
    observation: str,
    available_actions: Iterable[str],
    history: list[dict[str, str]],
    history_steps: int,
    step_number: int,
) -> str:
    recent = history[-max(history_steps, 0) :] if history_steps > 0 else []
    if recent:
        history_text = "\n".join(
            f"Step {step_number - len(recent) + index}: "
            f"observation={item['observation']!r}; action={item['action']!r}"
            for index, item in enumerate(recent)
        )
    else:
        history_text = "(none)"

    actions = "\n".join(f"- {action}" for action in available_actions)
    return (
        f"Environment: {environment}\n"
        f"Task: {task}\n\n"
        f"Recent history (at most {max(history_steps, 0)} steps):\n{history_text}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Available action templates or commands:\n{actions}\n\n"
        "Reason about the task, then return exactly one executable command inside "
        "one pair of tags: <action>command</action>. Do not emit a second action."
    )


class EpisodeRecorder:
    """Append-only episode checkpoint with a regenerated aggregate summary."""

    def __init__(self, output_dir: Path, *, resume: bool) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.episodes: list[dict[str, Any]] = []
        self._ids: set[str] = set()

        if resume and self.episodes_path.exists():
            for line_number, line in enumerate(
                self.episodes_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                episode = json.loads(line)
                episode_id = str(episode["episode_id"])
                if episode_id in self._ids:
                    raise ValueError(
                        f"Duplicate episode_id={episode_id!r} at line {line_number}"
                    )
                self.episodes.append(episode)
                self._ids.add(episode_id)
        elif not resume and self.episodes_path.exists():
            raise FileExistsError(
                f"{self.episodes_path} exists; choose a new output directory or use --resume"
            )

    def contains(self, episode_id: str) -> bool:
        return episode_id in self._ids

    def append(self, episode: dict[str, Any]) -> None:
        episode_id = str(episode["episode_id"])
        if episode_id in self._ids:
            raise ValueError(f"Duplicate episode_id={episode_id!r}")
        with self.episodes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
            stream.flush()
        self.episodes.append(episode)
        self._ids.add(episode_id)
        self.write_summary()

    def write_summary(self) -> dict[str, Any]:
        summary = summarize_episodes(self.episodes)
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)
        return summary


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(episodes)
    if count == 0:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "average_score": 0.0,
            "average_steps": 0.0,
            "invalid_actions": 0,
            "api_total_tokens": 0,
        }

    successes = sum(bool(episode.get("success")) for episode in episodes)
    return {
        "environment": episodes[0].get("environment", "unknown"),
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "average_score": sum(float(item.get("score", 0)) for item in episodes) / count,
        "average_steps": sum(int(item.get("steps", 0)) for item in episodes) / count,
        "invalid_actions": sum(
            int(item.get("invalid_actions", 0)) for item in episodes
        ),
        "api_total_tokens": sum(
            int(item.get("api_total_tokens", 0)) for item in episodes
        ),
    }
