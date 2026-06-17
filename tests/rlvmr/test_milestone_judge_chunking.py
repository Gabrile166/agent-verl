import json
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "d:/Workspace/Agentic/agent-verl")

from rlvmr.milestone.judge import MilestoneJudge, SegmentJudgeState


MILESTONES = [
    {"id": "M1", "name": "Found item", "phi": 0.25, "criteria": "item found"},
    {"id": "M2", "name": "Prepared item", "phi": 0.50, "criteria": "item prepared"},
    {"id": "M3", "name": "Completed task", "phi": 1.00, "criteria": "task complete"},
]


def _trajectory(length, with_scores=False):
    steps = []
    for i in range(length):
        step = {"action": f"action_{i + 1}", "observation": f"obs_{i + 1}"}
        if with_scores:
            step.update(
                {
                    "task_score_before": float(i),
                    "task_score_after": float(i + 1),
                    "task_score_delta": 1.0,
                }
            )
        steps.append(step)
    return steps


def _response(content):
    result = MagicMock()
    result.choices = [MagicMock()]
    result.choices[0].message.content = content
    return result


def _chunk_json(first_step, last_step, milestone_id="M1", summary="segment done", final_success=False):
    judgments = [{"step": step, "highest_milestone": milestone_id} for step in range(first_step, last_step + 1)]
    return json.dumps(
        {
            "judgments": judgments,
            "ending_highest_milestone": milestone_id,
            "ending_phi": 999.0,
            "final_success": final_success,
            "segment_summary": summary,
        }
    )


def _single_json(length):
    return json.dumps(
        {
            "judgments": [
                {"step": step, "highest_milestone": "M1", "phi": 0.25}
                for step in range(1, length + 1)
            ],
            "final_success": False,
            "reasoning": "single trajectory",
        }
    )


def _make_judge(mock_create, max_retries=1):
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    with patch("rlvmr.milestone.judge.OpenAI", return_value=mock_client):
        judge = MilestoneJudge(
            base_urls=["http://url1", "http://url2"],
            model="test-model",
            milestones=MILESTONES,
            max_retries=max_retries,
        )
    judge.clients = [mock_client, mock_client]
    return judge


def _auto_chunk_create(prompts):
    def _create(*args, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        match = re.search(r"Step (\d+) through Step (\d+)", prompt)
        if match:
            first_step = int(match.group(1))
            last_step = int(match.group(2))
            return _response(_chunk_json(first_step, last_step, milestone_id="M1"))

        step_count = len(re.findall(r"\nStep \d+:", prompt))
        return _response(_single_json(step_count))

    return _create


@pytest.mark.parametrize(
    "total,chunk_size,overlap,expected",
    [
        (1, 10, 1, [(0, 1, 0, 0)]),
        (8, 10, 1, [(0, 8, 0, 0)]),
        (10, 10, 1, [(0, 10, 0, 0)]),
        (11, 10, 1, [(0, 6, 0, 0), (6, 11, 5, 6)]),
        (19, 10, 1, [(0, 10, 0, 0), (10, 19, 9, 10)]),
        (20, 10, 1, [(0, 10, 0, 0), (10, 20, 9, 10)]),
        (21, 10, 1, [(0, 7, 0, 0), (7, 14, 6, 7), (14, 21, 13, 14)]),
        (29, 10, 1, [(0, 10, 0, 0), (10, 20, 9, 10), (20, 29, 19, 20)]),
        (30, 10, 1, [(0, 10, 0, 0), (10, 20, 9, 10), (20, 30, 19, 20)]),
        (31, 10, 1, [(0, 8, 0, 0), (8, 16, 7, 8), (16, 24, 15, 16), (24, 31, 23, 24)]),
        (32, 10, 1, [(0, 8, 0, 0), (8, 16, 7, 8), (16, 24, 15, 16), (24, 32, 23, 24)]),
        (33, 10, 1, [(0, 9, 0, 0), (9, 17, 8, 9), (17, 25, 16, 17), (25, 33, 24, 25)]),
        (41, 10, 1, [(0, 9, 0, 0), (9, 17, 8, 9), (17, 25, 16, 17), (25, 33, 24, 25), (33, 41, 32, 33)]),
        (50, 10, 1, [(0, 10, 0, 0), (10, 20, 9, 10), (20, 30, 19, 20), (30, 40, 29, 30), (40, 50, 39, 40)]),
        (21, 10, 0, [(0, 7, 0, 0), (7, 14, 7, 7), (14, 21, 14, 14)]),
        (21, 10, 5, [(0, 7, 0, 0), (7, 14, 2, 7), (14, 21, 9, 14)]),
        (21, 10, 15, [(0, 7, 0, 0), (7, 14, 0, 7), (14, 21, 5, 14)]),
    ],
)
def test_iter_judge_chunks(total, chunk_size, overlap, expected):
    chunks = list(MilestoneJudge._iter_judge_chunks(total, chunk_size, overlap))
    compact = [
        (c["judge_start"], c["judge_end"], c["context_start"], c["context_end"])
        for c in chunks
    ]
    assert compact == expected


@pytest.mark.parametrize("length", [1, 8, 10])
def test_short_trajectories_use_single_judge_call(length):
    prompts = []
    judge = _make_judge(_auto_chunk_create(prompts))

    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(length),
        MILESTONES,
        chunk_size=10,
        chunk_overlap=1,
    )

    assert len(prompts) == 1
    assert "Steps To Judge" not in prompts[0]
    assert len(result.step_phis) == length
    assert result.chunk_stats["chunk_enabled"] == 0.0


def test_single_judge_with_milestones_uses_phi_map_not_model_phi():
    def _create(*args, **kwargs):
        return _response(
            json.dumps(
                {
                    "judgments": [
                        {"step": 1, "highest_milestone": "M2", "phi": 0.99},
                        {"step": 2, "highest_milestone": "M3", "phi": 0.01},
                    ],
                    "final_success": True,
                    "reasoning": "model phi should be ignored",
                }
            )
        )

    judge = _make_judge(_create)
    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(2),
        MILESTONES,
        chunk_size=0,
    )

    assert result.step_phis == [0.50, 1.00]
    assert result.highest_milestones == ["M2", "M3"]


@pytest.mark.parametrize("length,expected_chunks", [(11, 2), (32, 4), (33, 4), (50, 5)])
def test_chunked_judge_returns_full_length(length, expected_chunks):
    prompts = []
    judge = _make_judge(_auto_chunk_create(prompts))

    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(length),
        MILESTONES,
        chunk_size=10,
        chunk_overlap=1,
    )

    assert len(prompts) == expected_chunks
    assert len(result.step_phis) == length
    assert result.chunk_stats["chunk_enabled"] == 1.0
    assert result.chunk_stats["chunk_count"] == float(expected_chunks)


def test_overlap_steps_are_context_only_and_not_duplicated():
    prompts = []
    judge = _make_judge(_auto_chunk_create(prompts))

    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(11),
        MILESTONES,
        chunk_size=10,
        chunk_overlap=1,
    )

    second_prompt = prompts[1]
    assert len(result.step_phis) == 11
    assert "## Context-Only Previous Steps" in second_prompt
    assert "Step 6:" in second_prompt
    assert "Output exactly one judgment for each of these 5 steps: Step 7 through Step 11." in second_prompt


def test_previous_state_and_global_step_numbers_in_second_prompt():
    prompts = []
    responses = [
        _chunk_json(1, 6, milestone_id="M2", summary="first summary"),
        _chunk_json(7, 11, milestone_id="M3", summary="second summary", final_success=True),
    ]

    def _create(*args, **kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _response(responses[len(prompts) - 1])

    judge = _make_judge(_create)
    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(11),
        MILESTONES,
        chunk_size=10,
        chunk_overlap=1,
    )

    assert "Previous highest milestone: M2" in prompts[1]
    assert "Previous phi: 0.5000" in prompts[1]
    assert "Previous summary: first summary" in prompts[1]
    assert "Step 7:" in prompts[1]
    assert result.final_success is True


def test_include_task_score_fields_in_chunk_prompt():
    prompts = []
    judge = _make_judge(_auto_chunk_create(prompts))

    judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(11, with_scores=True),
        MILESTONES,
        include_task_score=True,
        chunk_size=10,
        chunk_overlap=1,
    )

    assert "Task Score Before" in prompts[0]
    assert "Task Score After" in prompts[0]
    assert "Task Score Delta" in prompts[0]


def test_chunk_failure_fills_with_previous_phi_and_continues():
    prompts = []
    call_count = 0

    def _create(*args, **kwargs):
        nonlocal call_count
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        call_count += 1
        if call_count == 1:
            return _response(_chunk_json(1, 7, milestone_id="M1", summary="first ok"))
        if call_count == 2:
            raise RuntimeError("temporary judge failure")
        return _response(_chunk_json(15, 21, milestone_id="M2", summary="third ok"))

    judge = _make_judge(_create, max_retries=1)
    result = judge.judge_trajectory_with_milestones(
        "test task",
        _trajectory(21),
        MILESTONES,
        chunk_size=10,
        chunk_overlap=1,
    )

    assert len(result.step_phis) == 21
    assert result.step_phis[:7] == [0.25] * 7
    assert result.step_phis[7:14] == [0.25] * 7
    assert result.step_phis[14:] == [0.50] * 7
    assert result.chunk_stats["chunk_failures"] == 1.0
    assert "Previous highest milestone: M1" in prompts[2]


def test_chunk_parser_pads_truncates_and_uses_phi_map():
    judge = _make_judge(MagicMock())
    phi_map = MilestoneJudge._build_phi_map(MILESTONES)
    state = SegmentJudgeState(previous_highest_milestone="M1", previous_phi=0.25)

    too_few = json.dumps({"judgments": [{"step": 11, "highest_milestone": "M2", "phi": 999.0}]})
    parsed = judge._parse_chunk_response_with_phi_map(too_few, 3, phi_map, state)
    assert parsed.step_phis == [0.50, 0.50, 0.50]
    assert parsed.ending_phi == 0.50

    too_many = json.dumps(
        {
            "judgments": [
                {"step": 11, "highest_milestone": "M1"},
                {"step": 12, "highest_milestone": "M2"},
                {"step": 13, "highest_milestone": "M3"},
            ],
            "ending_highest_milestone": "M1",
        }
    )
    parsed = judge._parse_chunk_response_with_phi_map(too_many, 2, phi_map, state)
    assert parsed.step_phis == [0.25, 0.50]
    assert parsed.ending_highest_milestone == "M2"
