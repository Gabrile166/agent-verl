"""Label decision rules for pairwise trajectory comparison."""

from typing import Optional


ALFWORLD_PROGRESS_TIE_GAP = 0.10


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


def label_alfworld_progress_pair(
    progress_a: float,
    progress_b: float,
    min_gap: float = ALFWORLD_PROGRESS_TIE_GAP,
) -> Optional[str]:
    """Return winner by rule-based ALFWorld progress, or None for near ties."""
    if abs(progress_a - progress_b) < min_gap:
        return None
    if progress_a > progress_b:
        return "A"
    if progress_b > progress_a:
        return "B"
    return None
