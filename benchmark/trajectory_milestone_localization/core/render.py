"""Render helpers for trajectory milestone localization samples."""

from __future__ import annotations

from typing import List

from .schema import Milestone, TrajectoryMilestoneSample, TrajectoryStep

QUESTION_TEXT = "Which milestone is the highest achieved milestone in the current trajectory?"


def _format_milestones(milestones: List[Milestone]) -> str:
    return "\n".join(f"{milestone.id}. {milestone.description}" for milestone in milestones)


def _format_trajectory(steps: List[TrajectoryStep]) -> str:
    if not steps:
        return "(empty trajectory)"

    lines: List[str] = []
    for idx, step in enumerate(steps, start=1):
        lines.append(f"Step {idx}:")
        lines.append(f"Environment State: {step.obs_before if step.obs_before is not None else '(not provided)'}")
        lines.append(f"Agent Action: {step.action}")
        lines.append(f"Result: {step.obs_after}")
        if idx != len(steps):
            lines.append("")
    return "\n".join(lines)


def _format_answer_options(milestones: List[Milestone]) -> str:
    milestone_ids = [milestone.id for milestone in milestones]
    if len(milestone_ids) <= 1:
        return ", ".join(milestone_ids)
    return ", ".join(milestone_ids[:-1]) + f", or {milestone_ids[-1]}"


def render_model_input(sample: TrajectoryMilestoneSample) -> str:
    answer_options = _format_answer_options(sample.milestones)
    return "\n\n".join(
        [
            "You are evaluating the progress stage of an embodied agent on a task.",
            "## Task\n" + sample.task_description,
            (
                "## Milestones\n"
                + _format_milestones(sample.milestones)
                + "\n\nThe correct answer should be the highest milestone that has already been achieved by the current trajectory.\n"
                + "Do not choose a future milestone that has not yet been achieved."
            ),
            "## Current Trajectory\n" + _format_trajectory(sample.trajectory),
            (
                "## Question\n"
                f"{QUESTION_TEXT}\n\n"
                "Provide your reasoning first, then give your final answer.\n\n"
                "Output format:\n"
                "Reasoning: <your analysis>\n\n"
                f"Answer: <one milestone id, e.g. {answer_options}>"
            ),
        ]
    )


__all__ = ["QUESTION_TEXT", "render_model_input"]
