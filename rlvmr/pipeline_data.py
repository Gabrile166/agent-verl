# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-level pipeline data model for Milestone GAE.

Domain hierarchy:
    query (uid) → N trajectories (traj_uid)

This module provides typed dataclasses that replace the previous scattered
flat arrays with a unified structure, enabling:
  - Natural deduplication of query-level data (task, expert, milestones)
  - Direct FK lookup from trajectory → query
  - IDE autocomplete and missing-field TypeError at construction
  - Clean serialization via dataclasses.asdict() / pickle
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryRecord:
    """Query-level data (1 per uid, shared by n_rollout trajectories).
    
    Attributes:
        uid: Unique query identifier.
        task: Task description string.
        expert_trajectory: Expert demonstration trajectory (list of steps).
        milestones: Generated milestones (filled by MilestoneGenerator, None until then).
    """
    uid: str
    task: str
    expert_trajectory: List[Dict]
    milestones: Optional[List[Dict]] = None  # filled by generator


@dataclass
class TrajectoryRecord:
    """Trajectory-level data (1 per traj_uid).
    
    Attributes:
        traj_uid: Unique trajectory identifier.
        uid: Foreign key → QueryRecord.uid.
        policy_trajectory: Formatted policy trajectory dict.
        episode_reward: Cumulative episode reward.
        success: Whether this trajectory is successful.
        phis: Milestone achievement scores (filled by MilestoneJudge, None until then).
    """
    traj_uid: str
    uid: str                    # FK → QueryRecord
    policy_trajectory: Dict
    episode_reward: float
    success: bool
    phis: Optional[List[float]] = None  # filled by judge


@dataclass
class PipelineData:
    """Two-level container for milestone GAE pipeline data.
    
    queries: Dict mapping uid → QueryRecord (1 per environment/query).
    trajectories: Dict mapping traj_uid → TrajectoryRecord (1 per rollout trajectory).
    
    The relationship is: each TrajectoryRecord.uid points to a QueryRecord.uid,
    forming a one-to-many (query → trajectories) relationship.
    """
    queries: Dict[str, QueryRecord] = field(default_factory=dict)
    trajectories: Dict[str, TrajectoryRecord] = field(default_factory=dict)

    def get_query_for_traj(self, traj_uid: str) -> QueryRecord:
        """FK lookup: trajectory → query.
        
        KeyError if traj_uid or its parent uid is missing — this is intentional;
        missing data should crash loudly, not silently produce wrong results.
        """
        return self.queries[self.trajectories[traj_uid].uid]


# ==================== Serialization Self-Test ====================
if __name__ == "__main__":
    import pickle

    q = QueryRecord(uid="q1", task="test task", expert_trajectory=[{"action": "go north"}])
    t = TrajectoryRecord(
        traj_uid="t1", uid="q1", policy_trajectory={"traj": [{"obs": "x", "act": "y"}]},
        episode_reward=1.0, success=True
    )
    pd = PipelineData(queries={"q1": q}, trajectories={"t1": t})

    # Round-trip pickle
    data = pickle.dumps(pd)
    pd2 = pickle.loads(data)
    assert pd2.trajectories["t1"].episode_reward == 1.0
    assert pd2.get_query_for_traj("t1").task == "test task"

    # Required field enforcement
    try:
        QueryRecord(uid="q2", task="t")  # type: ignore  # missing expert_trajectory
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # FK lookup KeyError
    try:
        pd2.get_query_for_traj("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    print("[OK] PipelineData serialization + type safety passed")
