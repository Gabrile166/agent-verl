"""Prompt construction and response parsing for pairwise trajectory comparison."""

import re
from typing import Any, Dict, List, Optional

from ..core.schema import BenchmarkSample, TrajectoryData


class PairwiseComparisonPromptBuilder:
    """Build pairwise evaluation prompts and parse model outputs."""

    def build_query(self, sample: BenchmarkSample, milestones: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build the final query string with or without milestone context."""
        trajectory_a = self.format_trajectory(sample.trajectory_a)
        trajectory_b = self.format_trajectory(sample.trajectory_b)

        if milestones is None:
            return (
                "You are evaluating the progress of two agent trajectories on the same task.\n\n"
                "## Task Description\n"
                f"{sample.task_description}\n\n"
                "## Trajectory A\n"
                f"{trajectory_a}\n\n"
                "## Trajectory B\n"
                f"{trajectory_b}\n\n"
                "## Question\n"
                "Which trajectory has made MORE progress toward completing the task?\n"
                "Which one is closer to success?\n\n"
                "Provide your reasoning first, then give your final answer.\n\n"
                "Output format:\n"
                "Reasoning: <your analysis>\n"
                "Answer: <A or B>"
            )

        milestones_str = self.format_milestones(milestones)
        return (
            "You are evaluating the progress of two agent trajectories on the same task.\n\n"
            "## Task Description\n"
            f"{sample.task_description}\n\n"
            "## Progress Milestones (for reference)\n"
            f"{milestones_str}\n\n"
            "## Trajectory A\n"
            f"{trajectory_a}\n\n"
            "## Trajectory B\n"
            f"{trajectory_b}\n\n"
            "## Question\n"
            "Based on the milestones above, which trajectory has achieved HIGHER potential Φ\n"
            "(i.e., completed more milestones / closer to task success)?\n\n"
            "Provide your reasoning first, then give your final answer.\n\n"
            "Output format:\n"
            "Reasoning: <your analysis>\n"
            "Answer: <A or B>"
        )

    def format_trajectory(self, traj: TrajectoryData) -> str:
        """Format trajectory steps showing obs_before, action, obs_after, and final state."""
        if not traj.steps:
            return "No steps available.\n[Final State]: N/A"

        lines: List[str] = []
        for idx, step in enumerate(traj.steps, start=1):
            lines.extend(
                [
                    f"Step {idx}:",
                    f"  Environment State: {step.obs_before}",
                    f"  Agent Action: {step.action}",
                    f"  Result: {step.obs_after}",
                ]
            )

        lines.append(f"[Final State]: {traj.steps[-1].obs_after}")
        return "\n".join(lines)

    def format_milestones(self, milestones: List[Dict[str, Any]]) -> str:
        """Format milestones as Mx(Φ=value): description lines."""
        lines: List[str] = []
        has_m0 = False

        for milestone in milestones:
            milestone_id = str(milestone.get("id", "")).strip() or "M?"
            if milestone_id.upper() == "M0":
                has_m0 = True

            phi_raw = milestone.get("phi", "?")
            if isinstance(phi_raw, (int, float)):
                phi_text = f"{float(phi_raw):.2f}"
            else:
                phi_text = str(phi_raw)

            name = str(milestone.get("name", "")).strip()
            criteria = str(milestone.get("criteria", "")).strip()

            desc_parts = [part for part in [name, criteria] if part]
            description = " - ".join(desc_parts) if desc_parts else "(no description)"
            lines.append(f"{milestone_id}(Φ={phi_text}): {description}")

        if not has_m0:
            lines.insert(0, "M0(Φ=0.0): Not started")

        return "\n".join(lines)

    def parse_response(self, response: str) -> str:
        """Parse model response into A/B using fallback priority; otherwise return INVALID."""
        if not response:
            return "INVALID"

        # Priority 1: explicit "Answer: A/B" markers.
        answer_matches = list(re.finditer(r"answer\s*:\s*([ab])\b", response, flags=re.IGNORECASE))
        if answer_matches:
            return answer_matches[-1].group(1).upper()

        # Priority 2: final non-empty line is exactly a standalone A/B.
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines and re.fullmatch(r"[ab]", lines[-1], flags=re.IGNORECASE):
            return lines[-1].upper()

        # Priority 3: last standalone A/B token in the full text.
        token_matches = list(re.finditer(r"\b([ab])\b", response, flags=re.IGNORECASE))
        if token_matches:
            return token_matches[-1].group(1).upper()

        return "INVALID"
