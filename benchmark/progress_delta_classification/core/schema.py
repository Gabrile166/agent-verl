"""Schema definitions for progress delta classification benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_ALLOWED_LABELS = {"increase", "decrease", "same"}
_ALLOWED_ENVS = {"alfworld", "sciworld"}


def _require_text(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{key} must be non-empty")
    return text


def _require_number(data: Dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"{key} must exist")
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc


@dataclass
class TrajectoryStep:
    obs_before: Optional[str]
    action: str
    obs_after: str

    def __post_init__(self) -> None:
        if not str(self.action or "").strip():
            raise ValueError("trajectory step action must be non-empty")
        if not str(self.obs_after or "").strip():
            raise ValueError("trajectory step obs_after must be non-empty")
        if self.obs_before is not None:
            self.obs_before = str(self.obs_before)
        self.action = str(self.action)
        self.obs_after = str(self.obs_after)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "obs_before": self.obs_before,
            "obs_after": self.obs_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        if not isinstance(data, dict):
            raise ValueError("trajectory step must be an object")
        obs_before = data.get("obs_before")
        return cls(
            obs_before=None if obs_before is None else str(obs_before),
            action=str(data.get("action", "")),
            obs_after=str(data.get("obs_after", "")),
        )


@dataclass
class ProgressDeltaSample:
    sample_id: str
    env: str
    task_description: str
    trajectory_prefix: List[TrajectoryStep]
    added_steps: List[TrajectoryStep]
    progress_before: float
    progress_after: float
    progress_delta: float
    label: str
    details: Dict[str, Any]
    task_id: str = ""

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id or "").strip()
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")

        self.env = str(self.env or "").strip().lower()
        if self.env not in _ALLOWED_ENVS:
            raise ValueError(f"env must be one of {sorted(_ALLOWED_ENVS)}")

        self.task_description = str(self.task_description or "").strip()
        if not self.task_description:
            raise ValueError("task_description must be non-empty")

        if not isinstance(self.trajectory_prefix, list):
            raise ValueError("trajectory_prefix must be a list")
        if not isinstance(self.added_steps, list):
            raise ValueError("added_steps must be a list")
        if not self.added_steps:
            raise ValueError("added_steps must be non-empty")

        self.label = str(self.label or "").strip().lower()
        if self.label not in _ALLOWED_LABELS:
            raise ValueError(f"label must be one of {sorted(_ALLOWED_LABELS)}")

        for field_name in ("progress_before", "progress_after", "progress_delta"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            setattr(self, field_name, float(value))

        if not isinstance(self.details, dict):
            raise ValueError("details must be a dict")
        self.task_id = str(self.task_id or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "env": self.env,
            "task_description": self.task_description,
            "trajectory_prefix": [step.to_dict() for step in self.trajectory_prefix],
            "added_steps": [step.to_dict() for step in self.added_steps],
            "progress_before": self.progress_before,
            "progress_after": self.progress_after,
            "progress_delta": self.progress_delta,
            "label": self.label,
            "details": self.details,
        }
        if self.task_id:
            payload["task_id"] = self.task_id
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressDeltaSample":
        if not isinstance(data, dict):
            raise ValueError("progress delta sample must be an object")
        trajectory_prefix = data.get("trajectory_prefix")
        added_steps = data.get("added_steps")
        if trajectory_prefix is None:
            raise ValueError("trajectory_prefix must exist")
        if added_steps is None:
            raise ValueError("added_steps must exist")
        if not isinstance(trajectory_prefix, list):
            raise ValueError("trajectory_prefix must be a list")
        if not isinstance(added_steps, list):
            raise ValueError("added_steps must be a list")

        details = data.get("details", {})
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError("details must be a dict")

        return cls(
            sample_id=_require_text(data, "sample_id"),
            env=_require_text(data, "env"),
            task_description=_require_text(data, "task_description"),
            trajectory_prefix=[TrajectoryStep.from_dict(step) for step in trajectory_prefix],
            added_steps=[TrajectoryStep.from_dict(step) for step in added_steps],
            progress_before=_require_number(data, "progress_before"),
            progress_after=_require_number(data, "progress_after"),
            progress_delta=_require_number(data, "progress_delta"),
            label=_require_text(data, "label"),
            details=details,
            task_id=str(data.get("task_id") or details.get("task_id") or ""),
        )
