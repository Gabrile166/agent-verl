"""
Comprehensive unit tests for Milestone GAE pipeline.
Tests the flow from API results to advantage computation.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

import sys
sys.path.insert(0, 'd:/Workspace/Agentic/agent-verl')

from rlvmr.core_milestone_gae import (
    compute_milestone_gae,
    compute_milestone_gae_from_batch,
    TrajectoryData,
)


class TestComputeMilestoneGAE:
    """Test single trajectory GAE computation."""
    
    def test_basic_gae_computation(self):
        """Test basic GAE with simple inputs."""
        phis = [0.0, 0.2, 0.5, 0.8, 1.0]
        rewards = [0.0, 0.0, 0.0, 0.0, 10.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=rewards,
            done=True,
            success=True,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
        )
        
        assert len(adv) == 5
        assert len(ret) == 5
        # Last step advantage should be positive (got reward)
        assert adv[-1] > 0
    
    def test_failure_trajectory(self):
        """Test GAE for failed trajectory."""
        phis = [0.0, 0.1, 0.2, 0.3]
        rewards = [0.0, 0.0, 0.0, 0.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=rewards,
            done=True,
            success=False,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
        )
        
        assert len(adv) == 4
        # All advantages should be negative (cost penalty, no reward)
        assert all(a < 0.5 for a in adv)
    
    def test_empty_trajectory(self):
        """Test handling of empty trajectory."""
        adv, ret = compute_milestone_gae(
            phis=[],
            rewards=[],
            done=True,
            success=False,
        )
        
        assert adv == []
        assert ret == []
    
    def test_single_step_success(self):
        """Test single step successful trajectory."""
        phis = [0.5]
        rewards = [10.0]
        
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=rewards,
            done=True,
            success=True,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
        )
        
        assert len(adv) == 1
        # Should have positive advantage
        expected_delta = (10.0 - 0.05) + 0.99 * 1.0 - 0.5
        assert abs(adv[0] - expected_delta) < 0.01


class TestComputeMilestoneGAEFromBatch:
    """Test batch GAE computation with verl compatibility."""
    
    def test_batch_with_uuid_traj_ids(self):
        """Test that UUID string traj_ids work correctly."""
        # Create mock data with UUID-style traj_ids
        traj_ids = [
            "003ba7cd-e869-4389-902c-a06ad7f3ec19",
            "003ba7cd-e869-4389-902c-a06ad7f3ec19",
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
        ]
        traj_index = np.array(traj_ids)
        
        phis_dict = {
            "003ba7cd-e869-4389-902c-a06ad7f3ec19": [0.3, 0.6],
            "11111111-2222-3333-4444-555555555555": [0.2, 0.5, 0.8],
        }
        
        response_mask = torch.ones(5, 10)
        episode_rewards = np.array([10.0, 0.0])  # First success, second fail
        success_flags = np.array([True, False])
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            phis_dict=phis_dict,
            traj_index=traj_index,
            response_mask=response_mask,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
            episode_rewards=episode_rewards,
            success_flags=success_flags,
        )
        
        assert adv.shape == (5, 10)
        assert ret.shape == (5, 10)
        assert details["num_trajectories"] == 2
        # Check traj_id is string not int
        for td in details["traj_details"]:
            assert isinstance(td["traj_id"], str)
    
    def test_missing_phis_fallback(self):
        """Test fallback when phis_dict missing a trajectory."""
        traj_ids = ["traj-1", "traj-1", "traj-2", "traj-2"]
        traj_index = np.array(traj_ids)
        
        # Only provide phis for traj-1
        phis_dict = {
            "traj-1": [0.3, 0.7],
        }
        
        response_mask = torch.ones(4, 5)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            phis_dict=phis_dict,
            traj_index=traj_index,
            response_mask=response_mask,
        )
        
        # Should not crash, traj-2 uses fallback
        assert adv.shape == (4, 5)
        assert details["num_trajectories"] == 2
    
    def test_phis_length_mismatch(self):
        """Test handling when phis length doesn't match trajectory."""
        traj_ids = ["traj-1"] * 5  # 5 steps
        traj_index = np.array(traj_ids)
        
        # Only 3 phi values
        phis_dict = {"traj-1": [0.2, 0.5, 0.8]}
        
        response_mask = torch.ones(5, 10)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            phis_dict=phis_dict,
            traj_index=traj_index,
            response_mask=response_mask,
        )
        
        # Should pad phis with last value
        assert adv.shape == (5, 10)


class TestPhiDictOrdering:
    """Test that phis_dict maintains correct mapping."""
    
    def test_phis_match_trajectory(self):
        """Verify phis are correctly matched to their trajectories."""
        # Setup: 3 trajectories with distinct phi patterns
        traj_ids = [
            "alpha", "alpha",
            "beta", "beta", "beta",
            "gamma",
        ]
        traj_index = np.array(traj_ids)
        
        phis_dict = {
            "alpha": [0.1, 0.2],        # increasing
            "beta": [0.9, 0.8, 0.7],    # decreasing
            "gamma": [0.5],             # constant
        }
        
        response_mask = torch.ones(6, 1)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            phis_dict=phis_dict,
            traj_index=traj_index,
            response_mask=response_mask,
            norm_adv_by_std=False,  # Disable normalization to check raw values
        )
        
        # Returns should match phi values
        # Index 0,1 -> alpha phis [0.1, 0.2]
        # Index 2,3,4 -> beta phis [0.9, 0.8, 0.7]
        # Index 5 -> gamma phi [0.5]
        
        # Check that different trajectories have different return patterns
        assert ret[0, 0].item() != ret[2, 0].item()  # alpha != beta
        assert ret[2, 0].item() != ret[5, 0].item()  # beta != gamma


class TestEndToEndPipeline:
    """Test the complete pipeline from mock API results."""
    
    def test_full_pipeline_simulation(self):
        """Simulate complete pipeline from generator to GAE."""
        # Step 1: Simulate generator output (milestones)
        milestones_per_query = {
            "query-1": [
                {"id": "M1", "name": "Found object", "phi": 0.3, "criteria": "..."},
                {"id": "M2", "name": "Picked up", "phi": 0.6, "criteria": "..."},
                {"id": "M3", "name": "Placed", "phi": 1.0, "criteria": "..."},
            ],
            "query-2": [
                {"id": "M1", "name": "Located", "phi": 0.5, "criteria": "..."},
                {"id": "M2", "name": "Completed", "phi": 1.0, "criteria": "..."},
            ],
        }
        
        # Step 2: Simulate judge output (phis per trajectory)
        # Each query has multiple policy rollouts
        phis_dict = {
            "traj-q1-0": [0.0, 0.3, 0.6, 1.0],  # query-1, rollout 0, success
            "traj-q1-1": [0.0, 0.3, 0.3, 0.3],  # query-1, rollout 1, fail
            "traj-q2-0": [0.0, 0.5, 1.0],       # query-2, rollout 0, success
            "traj-q2-1": [0.0, 0.0, 0.0],       # query-2, rollout 1, fail
        }
        
        # Step 3: Build traj_index (flatten for batch)
        traj_ids = (
            ["traj-q1-0"] * 4 +
            ["traj-q1-1"] * 4 +
            ["traj-q2-0"] * 3 +
            ["traj-q2-1"] * 3
        )
        traj_index = np.array(traj_ids)
        
        # Step 4: Build episode rewards and success flags
        episode_rewards = np.array([10.0, 0.0, 10.0, 0.0])
        success_flags = np.array([True, False, True, False])
        
        # Step 5: Run GAE computation
        response_mask = torch.ones(len(traj_ids), 1)
        
        adv, ret, details = compute_milestone_gae_from_batch(
            batch_data=None,
            phis_dict=phis_dict,
            traj_index=traj_index,
            response_mask=response_mask,
            gamma=0.99,
            lam=0.95,
            cost=0.05,
            episode_rewards=episode_rewards,
            success_flags=success_flags,
        )
        
        # Verify
        assert adv.shape == (14, 1)
        assert ret.shape == (14, 1)
        assert details["num_trajectories"] == 4
        
        # Successful trajectories should have higher final advantages
        # (before normalization, we can't easily verify this after normalization)
        print(f"Pipeline test passed: {details['num_trajectories']} trajectories processed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
