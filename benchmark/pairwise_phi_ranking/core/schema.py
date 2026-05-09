"""Core schema definitions for pairwise phi ranking benchmark samples."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrajectoryStep:
    """One transition step in a trajectory."""

    obs_before: str
    action: str
    obs_after: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_before": self.obs_before,
            "action": self.action,
            "obs_after": self.obs_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            obs_before=str(data.get("obs_before", "")),
            action=str(data.get("action", "")),
            obs_after=str(data.get("obs_after", "")),
        )


@dataclass
class TrajectoryData:
    """A trajectory with task-level context and truncation metadata."""

    steps: List[TrajectoryStep]
    task_description: str
    truncated_at_step: int
    is_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "task_description": self.task_description,
            "truncated_at_step": self.truncated_at_step,
            "is_completed": self.is_completed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryData":
        return cls(
            steps=[TrajectoryStep.from_dict(step) for step in data.get("steps", [])],
            task_description=str(data.get("task_description", "")),
            truncated_at_step=int(data.get("truncated_at_step", 0)),
            is_completed=bool(data.get("is_completed", False)),
        )


@dataclass
class BenchmarkSample:
    """A pairwise comparison sample used by benchmark evaluation."""

    sample_id: str
    task_description: str
    trajectory_a: TrajectoryData
    trajectory_b: TrajectoryData
    label: str
    progress_scalar_a: float
    progress_scalar_b: float
    progress_gap: float
    difficulty: str
    task_type: str
    track: str
    pair_type: str = ""
    is_subset_pair: bool = False
    uses_expert_branch: bool = False

    def __post_init__(self) -> None:
        normalized_label = self.label.upper()
        if normalized_label not in {"A", "B"}:
            raise ValueError(f"label must be 'A' or 'B', got: {self.label}")
        self.label = normalized_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_description": self.task_description,
            "trajectory_a": self.trajectory_a.to_dict(),
            "trajectory_b": self.trajectory_b.to_dict(),
            "label": self.label,
            "progress_scalar_a": self.progress_scalar_a,
            "progress_scalar_b": self.progress_scalar_b,
            "progress_gap": self.progress_gap,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "track": self.track,
            "pair_type": self.pair_type,
            "is_subset_pair": self.is_subset_pair,
            "uses_expert_branch": self.uses_expert_branch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            task_description=str(data.get("task_description", "")),
            trajectory_a=TrajectoryData.from_dict(data.get("trajectory_a", {})),
            trajectory_b=TrajectoryData.from_dict(data.get("trajectory_b", {})),
            label=str(data.get("label", "")).upper(),
            progress_scalar_a=float(data.get("progress_scalar_a", 0.0)),
            progress_scalar_b=float(data.get("progress_scalar_b", 0.0)),
            progress_gap=float(data.get("progress_gap", 0.0)),
            difficulty=str(data.get("difficulty", "")),
            task_type=str(data.get("task_type", "")),
            track=str(data.get("track", "")),
            pair_type=str(data.get("pair_type", "")),
            is_subset_pair=bool(data.get("is_subset_pair", False)),
            uses_expert_branch=bool(data.get("uses_expert_branch", False)),
        )
