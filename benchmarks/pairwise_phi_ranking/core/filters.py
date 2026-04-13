"""Filtering and sample post-processing utilities for pairwise benchmark data."""

import random

from .schema import BenchmarkSample


def filter_ties(sample: BenchmarkSample) -> bool:
    """Return True when sample should be filtered as near-tie."""
    return sample.progress_gap < 0.10


def assign_difficulty(progress_gap: float) -> str:
    """Assign difficulty level from progress gap."""
    if progress_gap >= 0.4:
        return "easy"
    if progress_gap >= 0.2:
        return "medium"
    if progress_gap >= 0.1:
        return "hard"
    raise ValueError("progress_gap below 0.10 should be filtered as tie")


def randomize_ab_position(sample: BenchmarkSample) -> BenchmarkSample:
    """Swap trajectory A/B with 50% probability and keep labels consistent."""
    if random.random() >= 0.5:
        return sample

    data = sample.to_dict()
    data["trajectory_a"], data["trajectory_b"] = data["trajectory_b"], data["trajectory_a"]
    data["progress_scalar_a"], data["progress_scalar_b"] = (
        data["progress_scalar_b"],
        data["progress_scalar_a"],
    )

    if data["label"] == "A":
        data["label"] = "B"
    elif data["label"] == "B":
        data["label"] = "A"

    return BenchmarkSample.from_dict(data)


def detect_subset_relationship(sample: BenchmarkSample) -> bool:
    """Return True when either trajectory action sequence is a prefix of the other."""
    actions_a = [step.action for step in sample.trajectory_a.steps]
    actions_b = [step.action for step in sample.trajectory_b.steps]

    shorter, longer = sorted([actions_a, actions_b], key=len)
    return longer[: len(shorter)] == shorter


def filter_prefix_relationship(sample: BenchmarkSample) -> bool:
    """Filter non-expert_prefix samples when A/B actions still show a prefix relation."""
    if sample.pair_type == "expert_prefix":
        return False
    return detect_subset_relationship(sample)
