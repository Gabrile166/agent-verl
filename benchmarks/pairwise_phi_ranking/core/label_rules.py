"""Label decision rules for pairwise trajectory comparison."""

from typing import Optional


def label_sciworld_pair(score_a: float, score_b: float) -> Optional[str]:
    """Return winner label for SciWorld score comparison, or None for tie."""
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return None


def label_alfworld_tw_pair(depth_a: int, depth_b: int) -> Optional[str]:
    """Return winner by prefix depth, or None when depth gap is smaller than 3."""
    if abs(depth_a - depth_b) < 3:
        return None
    if depth_a > depth_b:
        return "A"
    if depth_b > depth_a:
        return "B"
    return None
