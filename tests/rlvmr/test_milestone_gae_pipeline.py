"""
Comprehensive unit tests for Milestone GAE pipeline.
Tests the flow from PipelineData to advantage computation.
"""

import pytest
import torch
import numpy as np
import pickle
from typing import Dict, List, Any

import sys
sys.path.insert(0, 'd:/Workspace/Agentic/agent-verl')

from rlvmr.core_milestone_gae import (
    compute_milestone_gae,
    compute_milestone_gae_from_batch,
    TrajectoryData,
)
from rlvmr.pipeline_data import (
    PipelineData,
    QueryRecord,
    TrajectoryRecord,
)


# ==================== Helper ====================
def _build_pipeline_data(traj_configs):
    """Build a PipelineData from a list of (traj_uid, uid, phis, reward, success) tuples.
    
    Automatically creates QueryRecords for each unique uid.
    """
    queries = {}
    trajectories = {}
    for t_uid, uid, phis, reward, success in traj_configs:
        if uid not in queries:
            queries[uid] = QueryRecord(
                uid=uid,
                task=f"task for {uid}",
                expert_trajectory=[{"action": "expert_action"}],
            )
        trajectories[t_uid] = TrajectoryRecord(
            traj_uid=t_uid,
            uid=uid,
            policy_trajectory={"traj": [{"action": "a", "obs": "o"}]},
            episode_reward=reward,
            success=success,
            phis=phis,
        )
    return PipelineData(queries=queries, trajectories=trajectories)


# ==================== Priority 1: PipelineData unit tests ====================
class TestQueryRecord:
    """Test QueryRecord construction and field access."""
    
    def test_required_fields(self):
        q = QueryRecord(uid="q1", task="test", expert_trajectory=[{"a": 1}])
        assert q.uid == "q1"
        assert q.task == "test"
        assert q.milestones is None
    
    def test_missing_required_field_raises(self):
        with pytest.raises(TypeError):
            QueryRecord(uid="q1", task="test")  # type: ignore  # missing expert_trajectory
    
    def test_milestones_mutable(self):
        q = QueryRecord(uid="q1", task="test", expert_trajectory=[])
        q.milestones = [{"id": "M1", "name": "m1"}]
        assert len(q.milestones) == 1


class TestTrajectoryRecord:
    """Test TrajectoryRecord construction and field access."""
    
    def test_required_fields(self):
        t = TrajectoryRecord(
            traj_uid="t1", uid="q1", policy_trajectory={},
            episode_reward=5.0, success=True,
        )
        assert t.episode_reward == 5.0
        assert t.phis is None
    
    def test_missing_required_field_raises(self):
        with pytest.raises(TypeError):
            TrajectoryRecord(traj_uid="t1", uid="q1")  # type: ignore  # missing fields
    
    def test_phis_mutable(self):
        t = TrajectoryRecord(
            traj_uid="t1", uid="q1", policy_trajectory={},
            episode_reward=0.0, success=False,
        )
        t.phis = [0.1, 0.5, 0.9]
        assert len(t.phis) == 3


class TestPipelineData:
    """Test PipelineData construction, FK lookup, and serialization."""
    
    def test_get_query_for_traj(self):
        pd = _build_pipeline_data([
            ("t1", "q1", [0.5], 10.0, True),
            ("t2", "q1", [0.2], 0.0, False),
        ])
        q = pd.get_query_for_traj("t1")
        assert q.uid == "q1"
        assert q == pd.get_query_for_traj("t2")  # same query
    
    def test_get_query_for_traj_keyerror_missing_traj(self):
        pd = _build_pipeline_data([("t1", "q1", [0.5], 10.0, True)])
        with pytest.raises(KeyError):
            pd.get_query_for_traj("nonexistent")
    
    def test_get_query_for_traj_keyerror_dangling_fk(self):
        """If trajectory points to a query that doesn't exist, KeyError."""
        pd = PipelineData(
            queries={},
            trajectories={"t1": TrajectoryRecord(
                traj_uid="t1", uid="missing_query",
                policy_trajectory={}, episode_reward=0.0, success=False,
            )},
        )
        with pytest.raises(KeyError):
            pd.get_query_for_traj("t1")
    
    def test_serialization_roundtrip(self):
        pd = _build_pipeline_data([
            ("t1", "q1", [0.1, 0.5], 10.0, True),
            ("t2", "q2", [0.2, 0.3], 0.0, False),
        ])
        data = pickle.dumps(pd)
        pd2 = pickle.loads(data)
        assert pd2.trajectories["t1"].episode_reward == 10.0
        assert pd2.get_query_for_traj("t2").uid == "q2"
    
    def test_empty_pipeline_data(self):
        pd = PipelineData()
        assert len(pd.queries) == 0
        assert len(pd.trajectories) == 0


# ==================== Priority 1+: Core GAE (updated for PipelineData) ====================
class TestComputeMilestoneGAE:
    """Test single trajectory GAE computation."""
    
    def test_basic_gae_computation(self):
        phis = [0.0, 0.2, 0.5, 0.8, 1.0]
        rewards = [0.0, 0.0, 0.0, 0.0, 10.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis, rewards=rewards, done=True, success=True,
            gamma=0.99, lam=0.95, cost=0.05,
        )
        
        assert len(adv) == 5
        assert len(ret) == 5
        assert adv[-1] > 0
    
    def test_failure_trajectory(self):
        phis = [0.0, 0.1, 0.2, 0.3]
        rewards = [0.0, 0.0, 0.0, 0.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis, rewards=rewards, done=True, success=False,
            gamma=0.99, lam=0.95, cost=0.05,
        )
        
        assert len(adv) == 4
        assert all(a < 0.5 for a in adv)
    
    def test_empty_trajectory(self):
        adv, ret = compute_milestone_gae(
            phis=[], rewards=[], done=True, success=False,
        )
        assert adv == []
        assert ret == []
    
    def test_single_step_success(self):
        phis = [0.5]
        rewards = [10.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis, rewards=rewards, done=True, success=True,
            gamma=0.99, lam=0.95, cost=0.05,
        )
        
        assert len(adv) == 1
        expected_delta = (10.0 - 0.05) + 0.99 * 1.0 - 0.5
        assert abs(adv[0] - expected_delta) < 0.01


# ==================== Priority 2: Batch GAE with PipelineData ====================
class TestComputeMilestoneGAEFromBatch:
    """Test batch GAE computation using PipelineData."""
    
    def test_batch_with_uuid_traj_ids(self):
        """Test that UUID string traj_ids work correctly with PipelineData."""
        traj_ids = [
            "003ba7cd-e869-4389-902c-a06ad7f3ec19",
            "003ba7cd-e869-4389-902c-a06ad7f3ec19",
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
        ]
        traj_index = np.array(traj_ids)
        
        pipeline_data = _build_pipeline_data([
            ("003ba7cd-e869-4389-902c-a06ad7f3ec19", "q1", [0.3, 0.6], 10.0, True),
            ("11111111-2222-3333-4444-555555555555", "q1", [0.2, 0.5, 0.8], 0.0, False),
        ])
        
        response_mask = torch.ones(5, 10)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            traj_index=traj_index,
            response_mask=response_mask,
            pipeline_data=pipeline_data,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
        )
        
        assert adv.shape == (5, 10)
        assert ret.shape == (5, 10)
        assert details["num_trajectories"] == 2
    
    def test_missing_phis_fallback(self):
        """Test fallback when PipelineData has no phis for a trajectory."""
        traj_ids = ["traj-1", "traj-1", "traj-2", "traj-2"]
        traj_index = np.array(traj_ids)
        
        # traj-1 has phis, traj-2 does not (None)
        pipeline_data = _build_pipeline_data([
            ("traj-1", "q1", [0.3, 0.7], 10.0, True),
        ])
        # Add traj-2 without phis
        pipeline_data.trajectories["traj-2"] = TrajectoryRecord(
            traj_uid="traj-2", uid="q1",
            policy_trajectory={}, episode_reward=0.0, success=False,
            phis=None,
        )
        
        response_mask = torch.ones(4, 5)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            traj_index=traj_index,
            response_mask=response_mask,
            pipeline_data=pipeline_data,
        )
        
        assert adv.shape == (4, 5)
        assert details["num_trajectories"] == 2
    
    def test_phis_length_mismatch(self):
        """Test handling when phis length doesn't match trajectory steps."""
        traj_ids = ["traj-1"] * 5  # 5 steps
        traj_index = np.array(traj_ids)
        
        # Only 3 phi values for a 5-step trajectory
        pipeline_data = _build_pipeline_data([
            ("traj-1", "q1", [0.2, 0.5, 0.8], 0.0, False),
        ])
        
        response_mask = torch.ones(5, 10)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            traj_index=traj_index,
            response_mask=response_mask,
            pipeline_data=pipeline_data,
        )
        
        assert adv.shape == (5, 10)


class TestPhiDictOrdering:
    """Test that phis in PipelineData maintain correct mapping."""
    
    def test_phis_match_trajectory(self):
        """Verify phis are correctly matched to their trajectories."""
        traj_ids = [
            "alpha", "alpha",
            "beta", "beta", "beta",
            "gamma",
        ]
        traj_index = np.array(traj_ids)
        
        pipeline_data = _build_pipeline_data([
            ("alpha", "q1", [0.1, 0.2], 0.0, False),
            ("beta", "q1", [0.9, 0.8, 0.7], 0.0, False),
            ("gamma", "q2", [0.5], 0.0, False),
        ])
        
        response_mask = torch.ones(6, 1)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            traj_index=traj_index,
            response_mask=response_mask,
            pipeline_data=pipeline_data,
            norm_adv_by_std=False,
        )
        
        # Index 0,1 -> alpha, 2,3,4 -> beta, 5 -> gamma
        assert ret[0, 0].item() != ret[2, 0].item()  # alpha != beta
        assert ret[2, 0].item() != ret[5, 0].item()  # beta != gamma


class TestEndToEndPipeline:
    """Test the complete pipeline from mock PipelineData to GAE."""
    
    def test_full_pipeline_simulation(self):
        """Simulate complete pipeline: queries + trajectories -> GAE."""
        # Build PipelineData with 2 queries, 4 trajectories
        pd = PipelineData()
        pd.queries["query-1"] = QueryRecord(
            uid="query-1",
            task="Put a hot apple in the fridge",
            expert_trajectory=[{"action": "go to kitchen"}],
            milestones=[
                {"id": "M1", "name": "Found object", "phi": 0.3},
                {"id": "M2", "name": "Picked up", "phi": 0.6},
                {"id": "M3", "name": "Placed", "phi": 1.0},
            ],
        )
        pd.queries["query-2"] = QueryRecord(
            uid="query-2",
            task="Clean a cup",
            expert_trajectory=[{"action": "go to sink"}],
            milestones=[
                {"id": "M1", "name": "Located", "phi": 0.5},
                {"id": "M2", "name": "Completed", "phi": 1.0},
            ],
        )
        
        pd.trajectories["traj-q1-0"] = TrajectoryRecord(
            traj_uid="traj-q1-0", uid="query-1",
            policy_trajectory={"traj": [{"action": "a", "obs": "o"}] * 4},
            episode_reward=10.0, success=True,
            phis=[0.0, 0.3, 0.6, 1.0],
        )
        pd.trajectories["traj-q1-1"] = TrajectoryRecord(
            traj_uid="traj-q1-1", uid="query-1",
            policy_trajectory={"traj": [{"action": "a", "obs": "o"}] * 4},
            episode_reward=0.0, success=False,
            phis=[0.0, 0.3, 0.3, 0.3],
        )
        pd.trajectories["traj-q2-0"] = TrajectoryRecord(
            traj_uid="traj-q2-0", uid="query-2",
            policy_trajectory={"traj": [{"action": "a", "obs": "o"}] * 3},
            episode_reward=10.0, success=True,
            phis=[0.0, 0.5, 1.0],
        )
        pd.trajectories["traj-q2-1"] = TrajectoryRecord(
            traj_uid="traj-q2-1", uid="query-2",
            policy_trajectory={"traj": [{"action": "a", "obs": "o"}] * 3},
            episode_reward=0.0, success=False,
            phis=[0.0, 0.0, 0.0],
        )
        
        # Build traj_index (per-step)
        traj_ids = (
            ["traj-q1-0"] * 4 +
            ["traj-q1-1"] * 4 +
            ["traj-q2-0"] * 3 +
            ["traj-q2-1"] * 3
        )
        traj_index = np.array(traj_ids)
        response_mask = torch.ones(len(traj_ids), 1)
        
        # Verify FK lookups work
        assert pd.get_query_for_traj("traj-q1-0").task == "Put a hot apple in the fridge"
        assert pd.get_query_for_traj("traj-q2-1").task == "Clean a cup"
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            traj_index=traj_index,
            response_mask=response_mask,
            pipeline_data=pd,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
        )
        
        assert adv.shape == (14, 1)
        assert ret.shape == (14, 1)
        assert details["num_trajectories"] == 4
        
        print(f"Pipeline test passed: {details['num_trajectories']} trajectories processed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
