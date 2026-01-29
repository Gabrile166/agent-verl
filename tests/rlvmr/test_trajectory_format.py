"""
单元测试：轨迹格式化函数

测试内容：
1. format_policy_trajectory: 从 <action> 标签提取动作，去除 <think> 等标签
2. format_expert_trajectory: Expert 轨迹格式化
"""

import pytest
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from rlvmr.discriminator_reward import (
    format_policy_trajectory,
    format_expert_trajectory,
)


class TestFormatPolicyTrajectory:
    """测试 format_policy_trajectory 函数"""
    
    def test_extract_action_from_tags(self):
        """测试从 <action> 标签提取动作"""
        trajectory = [
            {
                'active_masks': True,
                'full_output': '<think>I should go to fridge to get the apple</think><action>go to fridge 1</action>',
                'anchor_obs': 'You are in kitchen'
            }
        ]
        result = format_policy_trajectory(trajectory, task="get apple")
        
        assert result['task'] == 'get apple'
        assert len(result['traj']) == 1
        assert result['traj'][0]['action'] == 'go to fridge 1'
        assert result['traj'][0]['obs'] == 'You are in kitchen'
    
    def test_extract_action_with_planning_tag(self):
        """测试包含 <planning> 标签的情况"""
        trajectory = [
            {
                'active_masks': True,
                'full_output': '<planning>Need to open fridge first</planning><action>open fridge 1</action>',
                'anchor_obs': 'You are at fridge 1'
            }
        ]
        result = format_policy_trajectory(trajectory)
        
        assert len(result['traj']) == 1
        assert result['traj'][0]['action'] == 'open fridge 1'
    
    def test_skip_inactive_steps(self):
        """测试跳过 active_masks=False 的步骤"""
        trajectory = [
            {'active_masks': False, 'full_output': 'padding step', 'anchor_obs': 'padding'},
            {'active_masks': True, 'full_output': '<action>look</action>', 'anchor_obs': 'You see a room'}
        ]
        result = format_policy_trajectory(trajectory)
        
        assert len(result['traj']) == 1
        assert result['traj'][0]['action'] == 'look'
    
    def test_fallback_when_no_action_tag(self):
        """测试没有 <action> 标签时的回退处理"""
        trajectory = [
            {
                'active_masks': True,
                'full_output': '<think>thinking...</think>look around',
                'anchor_obs': 'You are in room'
            }
        ]
        result = format_policy_trajectory(trajectory)
        
        assert len(result['traj']) == 1
        # 应该移除 <think> 标签，保留 "look around"
        assert 'thinking' not in result['traj'][0]['action']
        assert 'look around' in result['traj'][0]['action']
    
    def test_handle_list_observation(self):
        """测试观测是列表的情况"""
        trajectory = [
            {
                'active_masks': True,
                'full_output': '<action>examine table</action>',
                'anchor_obs': ['You see a table with items on it']
            }
        ]
        result = format_policy_trajectory(trajectory)
        
        assert result['traj'][0]['obs'] == 'You see a table with items on it'
    
    def test_multiple_steps(self):
        """测试多步轨迹"""
        trajectory = [
            {
                'active_masks': True,
                'full_output': '<think>Look around first</think><action>look</action>',
                'anchor_obs': 'Initial observation'
            },
            {
                'active_masks': True,
                'full_output': '<planning>Go to counter</planning><action>go to countertop 1</action>',
                'anchor_obs': 'You see a countertop'
            },
            {
                'active_masks': True,
                'full_output': '<action>take knife 1</action>',
                'anchor_obs': 'You are at countertop 1'
            }
        ]
        result = format_policy_trajectory(trajectory, task="prepare food")
        
        assert result['task'] == 'prepare food'
        assert len(result['traj']) == 3
        assert result['traj'][0]['action'] == 'look'
        assert result['traj'][1]['action'] == 'go to countertop 1'
        assert result['traj'][2]['action'] == 'take knife 1'
    
    def test_empty_trajectory(self):
        """测试空轨迹"""
        result = format_policy_trajectory([], task="empty")
        
        assert result['task'] == 'empty'
        assert len(result['traj']) == 0


class TestFormatExpertTrajectory:
    """测试 format_expert_trajectory 函数"""
    
    def test_basic_format(self):
        """测试基本格式化"""
        expert_traj = [
            {'observation': 'You are in kitchen', 'action': 'go to fridge 1'},
            {'observation': 'You are at fridge 1', 'action': 'open fridge 1'},
        ]
        result = format_expert_trajectory(expert_traj)
        
        assert len(result) == 2
        assert result[0]['observation'] == 'You are in kitchen'
        assert result[0]['action'] == 'go to fridge 1'
    
    def test_alternative_field_names(self):
        """测试兼容不同的字段名 (obs vs observation)"""
        expert_traj = [
            {'obs': 'You are in room', 'action': 'look'},
        ]
        result = format_expert_trajectory(expert_traj)
        
        assert len(result) == 1
        assert result[0]['observation'] == 'You are in room'
        assert result[0]['action'] == 'look'
    
    def test_empty_trajectory(self):
        """测试空轨迹"""
        result = format_expert_trajectory([])
        assert result == []
    
    def test_none_trajectory(self):
        """测试 None 输入"""
        result = format_expert_trajectory(None)
        assert result == []


class TestIntegration:
    """集成测试：模拟完整的 Discriminator 输入构建流程"""
    
    def test_full_trajectory_processing(self):
        """测试完整的轨迹处理流程"""
        # 模拟 Policy 轨迹（包含各种标签）
        policy_traj = [
            {
                'active_masks': True,
                'full_output': '''<think>
I need to find an apple. Let me check the fridge.
</think><planning>
1. Go to fridge
2. Open fridge
3. Take apple
</planning><action>go to fridge 1</action>''',
                'anchor_obs': 'You are in the kitchen. You see a fridge.'
            },
            {
                'active_masks': True,
                'full_output': '<think>The fridge is closed</think><action>open fridge 1</action>',
                'anchor_obs': 'You are at fridge 1. The fridge is closed.'
            },
        ]
        
        # 模拟 Expert 轨迹
        expert_traj = [
            {'observation': 'You are in the kitchen', 'action': 'go to fridge 1'},
            {'observation': 'You are at fridge 1', 'action': 'open fridge 1'},
            {'observation': 'The fridge is open. You see an apple.', 'action': 'take apple 1 from fridge 1'},
        ]
        
        # 格式化
        formatted_policy = format_policy_trajectory(policy_traj, task="Get apple from fridge")
        formatted_expert = format_expert_trajectory(expert_traj)
        
        # 验证 Policy 轨迹
        assert formatted_policy['task'] == "Get apple from fridge"
        assert len(formatted_policy['traj']) == 2
        # 确保 <think> 和 <planning> 内容被移除
        assert 'think' not in str(formatted_policy['traj']).lower() or 'think' in formatted_policy['traj'][0].get('action', '').lower() == False
        assert formatted_policy['traj'][0]['action'] == 'go to fridge 1'
        assert formatted_policy['traj'][1]['action'] == 'open fridge 1'
        
        # 验证 Expert 轨迹
        assert len(formatted_expert) == 3
        assert formatted_expert[2]['action'] == 'take apple 1 from fridge 1'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
