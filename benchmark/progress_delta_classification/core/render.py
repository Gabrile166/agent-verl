"""Render helpers for progress delta classification samples."""

from __future__ import annotations

from typing import List

from .schema import ProgressDeltaSample, TrajectoryStep

QUESTION_TEXT = "After the newly added steps, did the task completion progress increase, decrease, or stay the same?"


def _format_steps(title: str, steps: List[TrajectoryStep]) -> str:
    lines = [title]
    if not steps:
        lines.append("(empty)")
        return "\n".join(lines)

    for idx, step in enumerate(steps, start=1):
        lines.append(f"Step {idx}:")
        lines.append(f"Environment State: {step.obs_before if step.obs_before is not None else '(not provided)'}")
        lines.append(f"Agent Action: {step.action}")
        lines.append(f"Result: {step.obs_after}")
        if idx != len(steps):
            lines.append("")
    return "\n".join(lines)


def render_model_input(sample: ProgressDeltaSample) -> str:
    return "\n\n".join(
        [
            "You are evaluating the progress of an embodied agent on a task.",
            "## Task\n" + sample.task_description,
            "## Current Trajectory\n" + _format_steps("", sample.trajectory_prefix).lstrip(),
            "## Newly Added Steps\n" + _format_steps("", sample.added_steps).lstrip(),
            (
                "## Question\n"
                f"{QUESTION_TEXT}\n\n"
                "Evaluate only the change caused by the newly added steps relative to the current trajectory prefix.\n"
                "Treat the prefix as the baseline. Ask: after executing the added steps, is the agent in a better position, a worse position, or essentially the same position for completing the task?\n"
                "Do not give credit to the added steps for progress that was already achieved before they started.\n\n"
                "Use these definitions:\n\n"
                "- increase: the newly added steps create clear new progress beyond the prefix. They complete a required subgoal, obtain or use a relevant object, reach a clearly useful location, reveal necessary information, or create a required state that was not already achieved in the prefix.\n\n"
                "- decrease: compared with the prefix, the newly added steps leave the agent in a clearly worse position. They undo earlier progress, move attention or effort away from a more useful prepared state, consume time on a less useful branch after a promising state has already been reached, or otherwise make completion less likely, even if the actions are natural, harmless-looking, or task-related. Decrease does not require explicit destruction or a reset.\n\n"
                "- same: the newly added steps do not materially change progress relative to the prefix. They may be irrelevant, repeated observations, failed actions with no state change, minor exploration without useful new information, or actions whose effect on future completion is small or ambiguous.\n\n"
                "Decision procedure:\n"
                "1. Identify what useful state has already been achieved by the prefix.\n"
                "2. Judge only whether the added steps improve, worsen, or preserve that state.\n"
                "3. If the added steps merely inspect, focus on, or restate something that was already available, do not count the earlier progress again. Such steps are usually same unless they pull the agent away from a more useful state, in which case they are decrease.\n"
                "4. If the added steps look superficially relevant but make the current situation less promising than the prefix, choose decrease.\n"
                "5. If the effect is truly small or unclear, choose same.\n\n"
                "Important:\n"
                "- Judge meaningful progress, not final task success.\n"
                "- Compare after-added-steps against the prefix, not against the empty start of the task.\n"
                "- Do not reward the added steps for progress already achieved in the prefix.\n"
                "- Do not count every movement as progress.\n"
                "- Do not count every irrelevant action as decrease.\n"
                "- Natural or task-related actions can still be decrease if they leave the agent worse off than the prefix.\n"
                "- Observation or focus actions are not automatically increase just because they mention a relevant object.\n\n"
                "Provide your reasoning first, then give your final answer.\n\n"
                "Output format:\n"
                "Reasoning: <your analysis>\n\n"
                "Answer: <increase / decrease / same>"
            ),
        ]
    )


__all__ = ["QUESTION_TEXT", "render_model_input"]
