"""
单元测试：Hybrid Advantage Computation

测试内容：
1. compute_hybrid_outcome_advantage 三种模式
2. _grpo_normalize group normalization
3. compute_hybrid_step_advantage step-level 优势
"""

import pytest
import numpy as np
import torch
import sys
import os

# 添加项目根目录到 path（向上两级：tests/rlvmr -> tests -> agent-verl）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from rlvmr.core_hybrid import (
    compute_hybrid_outcome_advantage,
    compute_hybrid_step_advantage,
    _grpo_normalize,
)


class TestComputeHybridOutcomeAdvantage:
    """测试 compute_hybrid_outcome_advantage"""
    
    def setup_method(self):
        self.batch_size = 4
        self.response_length = 10
        self.token_level_rewards = torch.zeros(self.batch_size, self.response_length)
        self.token_level_rewards[:, -1] = torch.tensor([1.0, 0.5, 0.8, 0.2])
        self.response_mask = torch.ones(self.batch_size, self.response_length)
        self.index = np.array(["g1", "g1", "g2", "g2"])
        self.traj_index = np.array(["t1", "t2", "t3", "t4"])
    
    def test_grpo_mode(self):
        """测试 GRPO 模式（仅环境奖励）"""
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=self.token_level_rewards,
            response_mask=self.response_mask,
            index=self.index,
            traj_index=self.traj_index,
            reward_mode="grpo"
        )
        
        assert advantages.shape == (self.batch_size, self.response_length)
        assert returns.shape == (self.batch_size, self.response_length)
        # 同组内 mean=0 after normalization
        group1_mean = advantages[:2, 0].mean()
        assert abs(group1_mean.item()) < 0.1
    
    def test_discriminator_mode(self):
        """测试 Discriminator 模式"""
        disc_episode_rewards = torch.tensor([0.8, 0.6, 0.9, 0.3])
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=self.token_level_rewards,
            response_mask=self.response_mask,
            index=self.index,
            traj_index=self.traj_index,
            disc_episode_rewards=disc_episode_rewards,
            reward_mode="discriminator"
        )
        
        assert advantages.shape == (self.batch_size, self.response_length)
    
    def test_discriminator_mode_missing_rewards(self):
        """测试 Discriminator 模式缺少奖励时报错"""
        with pytest.raises(ValueError, match="disc_episode_rewards is required"):
            compute_hybrid_outcome_advantage(
                token_level_rewards=self.token_level_rewards,
                response_mask=self.response_mask,
                index=self.index,
                traj_index=self.traj_index,
                reward_mode="discriminator"
            )
    
    def test_hybrid_mode(self):
        """测试 Hybrid 模式"""
        disc_episode_rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=self.token_level_rewards,
            response_mask=self.response_mask,
            index=self.index,
            traj_index=self.traj_index,
            disc_episode_rewards=disc_episode_rewards,
            episode_reward_weight=1.0,
            step_reward_weight=0.5,
            reward_mode="hybrid"
        )
        
        assert advantages.shape == (self.batch_size, self.response_length)
    
    def test_invalid_mode(self):
        """测试无效模式报错"""
        with pytest.raises(ValueError, match="Unknown reward_mode"):
            compute_hybrid_outcome_advantage(
                token_level_rewards=self.token_level_rewards,
                response_mask=self.response_mask,
                index=self.index,
                traj_index=self.traj_index,
                reward_mode="invalid"
            )


class TestGrpoNormalize:
    """测试 _grpo_normalize"""
    
    def test_single_group(self):
        """测试单组归一化"""
        scores = torch.tensor([1.0, 2.0, 3.0])
        response_mask = torch.ones(3, 5)
        index = np.array(["g1", "g1", "g1"])
        traj_index = np.array(["t1", "t2", "t3"])
        
        advantages = _grpo_normalize(scores, response_mask, index, traj_index)
        
        # 归一化后 group 内均值应接近 0
        mean_adv = advantages[:, 0].mean()
        assert abs(mean_adv.item()) < 0.1
    
    def test_multiple_groups(self):
        """测试多组归一化"""
        scores = torch.tensor([1.0, 5.0, 2.0, 4.0])
        response_mask = torch.ones(4, 5)
        index = np.array(["g1", "g1", "g2", "g2"])
        traj_index = np.array(["t1", "t2", "t3", "t4"])
        
        advantages = _grpo_normalize(scores, response_mask, index, traj_index)
        
        # 每组内均值接近 0
        g1_mean = advantages[:2, 0].mean()
        g2_mean = advantages[2:, 0].mean()
        assert abs(g1_mean.item()) < 0.1
        assert abs(g2_mean.item()) < 0.1


class TestComputeHybridStepAdvantage:
    """测试 compute_hybrid_step_advantage"""
    
    def setup_method(self):
        self.batch_size = 4
        self.response_length = 10
        self.env_step_rewards = torch.tensor([0.5, 0.3, 0.8, 0.2])
        self.disc_step_rewards = torch.tensor([0.6, 0.4, 0.7, 0.3])
        self.response_mask = torch.ones(self.batch_size, self.response_length)
        self.step_group_uids = np.array(["s1", "s1", "s2", "s2"])
    
    def test_grpo_mode(self):
        """测试 GRPO 模式"""
        step_adv = compute_hybrid_step_advantage(
            env_step_rewards=self.env_step_rewards,
            disc_step_rewards=None,
            response_mask=self.response_mask,
            step_group_uids=self.step_group_uids,
            reward_mode="grpo"
        )
        
        assert step_adv.shape == (self.batch_size, self.response_length)
    
    def test_hybrid_mode(self):
        """测试 Hybrid 模式"""
        step_adv = compute_hybrid_step_advantage(
            env_step_rewards=self.env_step_rewards,
            disc_step_rewards=self.disc_step_rewards,
            response_mask=self.response_mask,
            step_group_uids=self.step_group_uids,
            step_reward_weight=0.5,
            reward_mode="hybrid"
        )
        
        assert step_adv.shape == (self.batch_size, self.response_length)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
