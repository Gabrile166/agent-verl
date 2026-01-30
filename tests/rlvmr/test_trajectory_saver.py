"""
单元测试：TrajectorySaver

测试轨迹数据保存功能，验证 JSONL 格式正确性。
"""

import pytest
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from rlvmr.trajectory_saver import TrajectorySaver


class TestTrajectorySaver:
    """测试 TrajectorySaver 类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def mock_trajectory_data(self):
        """模拟轨迹数据"""
        return [
            # Episode 1
            [
                {'active_masks': True, 'anchor_obs': 'You are in the kitchen.', 'full_output': '<think>I need to find an apple.</think><action>look</action>'},
                {'active_masks': True, 'anchor_obs': 'You see a countertop.', 'full_output': '<action>go to countertop 1</action>'},
                {'active_masks': True, 'anchor_obs': 'You are at countertop 1.', 'full_output': '<action>take apple 1</action>'},
                {'active_masks': False, 'anchor_obs': '', 'full_output': ''},  # Inactive step
            ],
            # Episode 2
            [
                {'active_masks': True, 'anchor_obs': 'You are in the bedroom.', 'full_output': '<think>Look around first.</think><action>look</action>'},
                {'active_masks': True, 'anchor_obs': 'You see a bed.', 'full_output': '<action>go to dresser 1</action>'},
            ],
        ]
    
    @pytest.fixture
    def mock_expert_trajectories(self):
        """模拟专家轨迹"""
        return [
            [
                {'observation': 'You are in the kitchen.', 'action': 'go to countertop 1'},
                {'observation': 'You are at countertop 1.', 'action': 'take apple 1'},
                {'observation': 'You have the apple.', 'action': 'go to fridge 1'},
            ],
        ]
    
    def test_initialization_creates_directory(self, temp_dir):
        """测试初始化时创建目录"""
        output_path = os.path.join(temp_dir, "nested", "trajectory_data")
        saver = TrajectorySaver(output_dir=output_path, enabled=True)
        
        assert os.path.exists(output_path)
        assert saver.enabled is True
    
    def test_disabled_saver_returns_none(self, temp_dir):
        """测试禁用状态不保存文件"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=False)
        
        result = saver.save_batch(
            batch_id=0,
            trajectory_list=[],
            episode_rewards=np.array([]),
            expert_trajectories=[],
            tasks=[],
            traj_uids=np.array([]),
        )
        
        assert result is None
        assert len(os.listdir(temp_dir)) == 0  # 不应创建任何文件
    
    def test_save_batch_creates_jsonl_file(self, temp_dir, mock_trajectory_data, mock_expert_trajectories):
        """测试保存批次创建 JSONL 文件"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        tasks = ["Put apple in fridge.", "Find the key."]
        episode_rewards = np.array([10.0, 0.0])
        traj_uids = np.array(["uid-001", "uid-002"])
        
        filepath = saver.save_batch(
            batch_id=0,
            trajectory_list=mock_trajectory_data,
            episode_rewards=episode_rewards,
            expert_trajectories=mock_expert_trajectories,
            tasks=tasks,
            traj_uids=traj_uids,
            rollout_n=2,
            global_step=100,
        )
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith(".jsonl")
        assert "batch_000000_step_100" in filepath
    
    def test_jsonl_format_correctness(self, temp_dir, mock_trajectory_data, mock_expert_trajectories):
        """测试 JSONL 格式正确性"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        tasks = ["Put apple in fridge.", "Find the key."]
        episode_rewards = np.array([10.0, 0.0])
        traj_uids = np.array(["uid-001", "uid-002"])
        
        filepath = saver.save_batch(
            batch_id=0,
            trajectory_list=mock_trajectory_data,
            episode_rewards=episode_rewards,
            expert_trajectories=mock_expert_trajectories,
            tasks=tasks,
            traj_uids=traj_uids,
            rollout_n=2,
            global_step=0,
        )
        
        # 读取并解析 JSONL（注意：使用 indent=2，所以需要合并多行）
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析 JSON 对象（pretty-printed，每个对象以 "{\n" 开头）
        import re
        # 查找所有顶级 JSON 对象
        samples = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            content_stripped = content[idx:].lstrip()
            if not content_stripped:
                break
            try:
                obj, end = decoder.raw_decode(content_stripped)
                samples.append(obj)
                idx += len(content) - len(content_stripped) + end
            except json.JSONDecodeError:
                break
        
        assert len(samples) == 1  # 2 episodes / 2 rollout_n = 1 sample
        
        sample = samples[0]
        
        # 验证 sample 结构
        assert "sample_index" in sample
        assert "prompt_index" in sample
        assert "expert_trajectory" in sample
        assert "episodes" in sample
        
        # 验证 episodes
        assert len(sample["episodes"]) == 2  # 2 episodes in this sample
        
        # 验证第一个 episode (成功)
        ep0 = sample["episodes"][0]
        assert ep0["episode_reward"] == 10.0
        assert ep0["is_success"] is True
        assert ep0["task"] == "Put apple in fridge."
        assert ep0["traj_uid"] == "uid-001"
        assert ep0["episode_length"] == 3  # 只有 3 个 active steps
        
        # 验证第一个 episode 的轨迹
        assert len(ep0["traj"]) == 3
        assert ep0["traj"][0]["action"] == "look"
        assert ep0["traj"][1]["action"] == "go to countertop 1"
        assert ep0["traj"][2]["action"] == "take apple 1"
        
        # 验证第二个 episode (失败)
        ep1 = sample["episodes"][1]
        assert ep1["episode_reward"] == 0.0
        assert ep1["is_success"] is False
    
    def test_extract_action_from_tags(self, temp_dir):
        """测试从标签中提取动作"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        # 有 action 标签
        assert saver._extract_action('<action>go to fridge 1</action>') == 'go to fridge 1'
        
        # 有 think 和 action 标签
        assert saver._extract_action('<think>I need...</think><action>look</action>') == 'look'
        
        # 无标签（返回原文）
        assert saver._extract_action('just plain text') == 'just plain text'
    
    def test_extract_reason_from_tags(self, temp_dir):
        """测试从标签中提取推理内容"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        # think 标签
        result = saver._extract_reason('<think>I need to find an apple</think><action>look</action>')
        assert '<think>I need to find an apple</think>' == result
        
        # planning 标签
        result = saver._extract_reason('<planning>Step 1: Find apple</planning>')
        assert '<planning>Step 1: Find apple</planning>' == result
        
        # 无标签
        assert saver._extract_reason('no tags here') == ''
    
    def test_expert_trajectory_formatting(self, temp_dir, mock_expert_trajectories):
        """测试专家轨迹格式化"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        formatted = saver._format_expert_trajectory(mock_expert_trajectories[0])
        
        assert len(formatted) == 3
        assert formatted[0]["step_index"] == 0
        assert formatted[0]["obs"] == "You are in the kitchen."
        assert formatted[0]["action"] == "go to countertop 1"
    
    def test_multiple_batches_increment_count(self, temp_dir, mock_trajectory_data, mock_expert_trajectories):
        """测试多批次保存计数递增"""
        saver = TrajectorySaver(output_dir=temp_dir, enabled=True)
        
        tasks = ["Task 1", "Task 2"]
        episode_rewards = np.array([10.0, 0.0])
        traj_uids = np.array(["uid-001", "uid-002"])
        
        # 保存两批
        saver.save_batch(batch_id=0, trajectory_list=mock_trajectory_data, 
                        episode_rewards=episode_rewards, expert_trajectories=mock_expert_trajectories,
                        tasks=tasks, traj_uids=traj_uids, rollout_n=2, global_step=0)
        
        saver.save_batch(batch_id=1, trajectory_list=mock_trajectory_data,
                        episode_rewards=episode_rewards, expert_trajectories=mock_expert_trajectories,
                        tasks=tasks, traj_uids=traj_uids, rollout_n=2, global_step=100)
        
        # 应该有两个文件
        files = os.listdir(temp_dir)
        assert len(files) == 2
        assert saver._batch_count == 2


class TestFormatEpisode:
    """测试 _format_episode 方法"""
    
    @pytest.fixture
    def saver(self):
        return TrajectorySaver(output_dir="/tmp/test", enabled=False)
    
    def test_format_episode_with_discriminator_scores(self, saver):
        """测试带有 discriminator 评分的 episode 格式化"""
        trajectory = [
            {'active_masks': True, 'anchor_obs': 'Obs 1', 'full_output': '<action>action 1</action>'},
            {'active_masks': True, 'anchor_obs': 'Obs 2', 'full_output': '<action>action 2</action>'},
        ]
        
        episode = saver._format_episode(
            episode_index=0,
            trajectory=trajectory,
            task="Test task",
            traj_uid="test-uid",
            episode_reward=10.0,
            disc_episode_score=0.8,
            disc_step_scores=[0.5, 1.0],
        )
        
        assert episode["episode_success_score"] == 0.8
        assert episode["traj"][0]["step_score"] == 0.5
        assert episode["traj"][0]["is_critical"] is True  # 0.5 >= 0.5
        assert episode["traj"][1]["step_score"] == 1.0
        assert episode["traj"][1]["is_critical"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
