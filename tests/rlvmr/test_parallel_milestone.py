"""
Unit tests for parallel milestone generation and judging.
Verifies that parallel execution maintains correct ordering.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Mock imports for testing
import sys
sys.path.insert(0, 'd:/Workspace/Agentic/agent-verl')


class TestParallelOrdering:
    """Test that parallel execution maintains correct ordering."""
    
    def test_batch_generate_ordering(self):
        """Test that batch_generate returns results in correct order."""
        from rlvmr.milestone.generator import MilestoneGenerator, GeneratedMilestones
        
        # Create mock clients
        mock_client = MagicMock()
        
        # Simulate varying response times to test ordering
        def mock_create(*args, **kwargs):
            content = kwargs.get('messages', [{}])[0].get('content', '')
            # Extract task index from content
            import re
            match = re.search(r'Task (\d+)', content)
            task_idx = int(match.group(1)) if match else 0
            
            # Simulate varying delays (task 0 takes longest)
            delay = 0.1 * (5 - task_idx)
            time.sleep(delay)
            
            # Return milestone with task index embedded
            result = MagicMock()
            result.choices = [MagicMock()]
            result.choices[0].message.content = f'''{{
                "milestones": [
                    {{"id": "M1", "name": "Task{task_idx}_M1", "phi": 0.5, "criteria": "test"}},
                    {{"id": "M2", "name": "Task{task_idx}_M2", "phi": 1.0, "criteria": "test"}}
                ],
                "reasoning": "Task {task_idx}"
            }}'''
            return result
        
        mock_client.chat.completions.create = mock_create
        
        with patch('rlvmr.milestone.generator.OpenAI', return_value=mock_client):
            generator = MilestoneGenerator(
                base_urls=["http://url1", "http://url2"],
                model="test-model",
                num_milestones=2,
            )
            # Inject mock clients
            generator.clients = [mock_client, mock_client]
            
            # Prepare test data
            task_descriptions = [f"Task {i}" for i in range(5)]
            expert_trajectories = [
                [{"observation": f"obs_{i}", "action": f"action_{i}"}]
                for i in range(5)
            ]
            
            # Run batch generate
            results = generator.batch_generate(task_descriptions, expert_trajectories)
            
            # Verify ordering
            assert len(results) == 5, f"Expected 5 results, got {len(results)}"
            
            for i, result in enumerate(results):
                # Check that result i corresponds to Task i
                assert result.success, f"Result {i} should be successful"
                milestone_name = result.milestones[0]['name']
                assert f"Task{i}" in milestone_name, \
                    f"Result {i} has wrong task: {milestone_name}"
    
    def test_batch_judge_ordering(self):
        """Test that batch_judge returns results in correct order."""
        from rlvmr.milestone.judge import MilestoneJudge, JudgmentResult
        
        mock_client = MagicMock()
        
        def mock_create(*args, **kwargs):
            content = kwargs.get('messages', [{}])[0].get('content', '')
            import re
            match = re.search(r'Traj_(\d+)', content)
            traj_idx = int(match.group(1)) if match else 0
            
            # Vary delay to test ordering
            delay = 0.05 * (8 - traj_idx)
            time.sleep(delay)
            
            # Return phi values encoding the trajectory index
            result = MagicMock()
            result.choices = [MagicMock()]
            result.choices[0].message.content = f'''{{
                "judgments": [
                    {{"step": 1, "highest_milestone": "M{traj_idx}", "phi": 0.{traj_idx}}}
                ],
                "final_success": true,
                "reasoning": "Traj {traj_idx}"
            }}'''
            return result
        
        mock_client.chat.completions.create = mock_create
        
        with patch('rlvmr.milestone.judge.OpenAI', return_value=mock_client):
            judge = MilestoneJudge(
                base_urls=["http://url1", "http://url2"],
                model="test-model",
                milestones=[{"id": "M1", "name": "test", "phi": 1.0, "criteria": "test"}],
            )
            judge.clients = [mock_client, mock_client]
            
            # Prepare test data
            task_descriptions = [f"Traj_{i} description" for i in range(8)]
            trajectories = [
                [{"action": f"action_{i}", "observation": f"obs_{i}"}]
                for i in range(8)
            ]
            
            # Run batch judge
            results = judge.batch_judge(task_descriptions, trajectories)
            
            # Verify ordering
            assert len(results) == 8, f"Expected 8 results, got {len(results)}"
            
            for i, result in enumerate(results):
                expected_phi = float(f"0.{i}")
                actual_phi = result.step_phis[0]
                assert actual_phi == expected_phi, \
                    f"Result {i} has wrong phi: {actual_phi}, expected {expected_phi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
