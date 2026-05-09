"""Schema definitions for trajectory milestone localization benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_ALLOWED_ENVS = {"alfworld", "sciworld"}


def _require_text(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{key} must be non-empty")
    return text


def _require_int(data: Dict[str, Any], key: str) -> int:
    if key not in data:
        raise ValueError(f"{key} must exist")
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


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
class Milestone:
    id: str
    description: str
    milestone_index: int

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.description = str(self.description or "").strip()
        if not self.id:
            raise ValueError("milestone id must be non-empty")
        if not self.description:
            raise ValueError(f"milestone {self.id} description must be non-empty")
        if not isinstance(self.milestone_index, int):
            raise ValueError("milestone_index must be an integer")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "milestone_index": self.milestone_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        if not isinstance(data, dict):
            raise ValueError("milestone must be an object")
        return cls(
            id=_require_text(data, "id"),
            description=_require_text(data, "description"),
            milestone_index=_require_int(data, "milestone_index"),
        )


@dataclass
class TrajectoryMilestoneSample:
    sample_id: str
    env: str
    task_description: str
    milestones: List[Milestone]
    trajectory: List[TrajectoryStep]
    label: str
    label_index: int
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

        if not isinstance(self.milestones, list) or not self.milestones:
            raise ValueError("milestones must be a non-empty list")
        if not isinstance(self.trajectory, list) or not self.trajectory:
            raise ValueError("trajectory must be a non-empty list")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a dict")

        milestone_ids = [item.id for item in self.milestones]
        if len(set(milestone_ids)) != len(milestone_ids):
            raise ValueError("milestone ids must be unique")
        milestone_index_map = {item.id: item.milestone_index for item in self.milestones}
        if len(set(milestone_index_map.values())) != len(milestone_index_map):
            raise ValueError("milestone_index values must be unique")

        self.label = str(self.label or "").strip()
        if self.label not in milestone_index_map:
            raise ValueError(f"label {self.label!r} not found in milestone ids")
        if not isinstance(self.label_index, int):
            raise ValueError("label_index must be an integer")
        expected_label_index = milestone_index_map[self.label]
        if self.label_index != expected_label_index:
            raise ValueError(
                f"label_index {self.label_index} does not match milestone_index {expected_label_index} for label {self.label}"
            )

        self.task_id = str(self.task_id or "").strip()

    def milestone_ids(self) -> List[str]:
        return [item.id for item in self.milestones]

    def label_to_index_map(self) -> Dict[str, int]:
        return {item.id: item.milestone_index for item in self.milestones}

    def max_label_index_error(self) -> int:
        milestone_indices = [item.milestone_index for item in self.milestones]
        return max(abs(index - self.label_index) for index in milestone_indices)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sample_id": self.sample_id,
            "env": self.env,
            "task_description": self.task_description,
            "milestones": [milestone.to_dict() for milestone in self.milestones],
            "trajectory": [step.to_dict() for step in self.trajectory],
            "label": self.label,
            "label_index": self.label_index,
            "details": self.details,
        }
        if self.task_id:
            payload["task_id"] = self.task_id
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryMilestoneSample":
        if not isinstance(data, dict):
            raise ValueError("trajectory milestone sample must be an object")
        milestones = data.get("milestones")
        trajectory = data.get("trajectory")
        if milestones is None:
            raise ValueError("milestones must exist")
        if trajectory is None:
            raise ValueError("trajectory must exist")
        if not isinstance(milestones, list):
            raise ValueError("milestones must be a list")
        if not isinstance(trajectory, list):
            raise ValueError("trajectory must be a list")

        details = data.get("details", {})
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError("details must be a dict")

        return cls(
            sample_id=_require_text(data, "sample_id"),
            env=_require_text(data, "env"),
            task_description=_require_text(data, "task_description"),
            milestones=[Milestone.from_dict(item) for item in milestones],
            trajectory=[TrajectoryStep.from_dict(item) for item in trajectory],
            label=_require_text(data, "label"),
            label_index=_require_int(data, "label_index"),
            details=details,
            task_id=str(data.get("task_id") or details.get("task_id") or ""),
        )
