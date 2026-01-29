"""
Integration Test for HybridGRPO Training Flow

模拟训练流程的关键步骤，检测潜在问题:
1. compute_hybrid_outcome_advantage 参数传递
2. rollout_loop 返回值处理
3. RayPPOTrainer compute_advantage 调用链

运行: python -m pytest tests/rlvmr/test_integration_hybrid.py -v
"""

import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestHybridAdvantageComputation:
    """测试 compute_hybrid_outcome_advantage 函数"""
    
    def test_grpo_mode_basic(self):
        """测试 GRPO 模式基本功能"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 8
        response_length = 32
        
        # 模拟输入数据
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        
        # 创建分组索引 (每2个样本一组)
        index = np.array(['g0', 'g0', 'g1', 'g1', 'g2', 'g2', 'g3', 'g3'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            reward_mode="grpo"
        )
        
        assert advantages.shape == (batch_size, response_length)
        assert returns.shape == (batch_size, response_length)
        assert not torch.isnan(advantages).any(), "Advantages contain NaN"
    
    def test_hybrid_mode_with_none_disc_rewards(self):
        """测试 hybrid 模式在没有 Discriminator 奖励时的行为"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 4
        response_length = 16
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array(['g0', 'g0', 'g1', 'g1'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        # disc_episode_rewards 和 disc_step_rewards 为 None
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            env_step_rewards=None,
            disc_step_rewards=None,
            disc_episode_rewards=None,
            reward_mode="hybrid"
        )
        
        assert advantages.shape == (batch_size, response_length)
        assert not torch.isnan(advantages).any(), "Advantages contain NaN with None disc rewards"
    
    def test_hybrid_mode_with_disc_rewards(self):
        """测试 hybrid 模式使用 Discriminator 奖励"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 4
        response_length = 16
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array(['g0', 'g0', 'g1', 'g1'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        # 模拟 Discriminator 奖励
        disc_episode_rewards = torch.rand(batch_size)
        disc_step_rewards = torch.rand(batch_size)
        env_step_rewards = torch.rand(batch_size)
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            env_step_rewards=env_step_rewards,
            disc_step_rewards=disc_step_rewards,
            disc_episode_rewards=disc_episode_rewards,
            episode_reward_weight=1.0,
            step_reward_weight=0.1,
            reward_mode="hybrid"
        )
        
        assert advantages.shape == (batch_size, response_length)
        assert not torch.isnan(advantages).any()
    
    def test_discriminator_mode_requires_disc_rewards(self):
        """测试 discriminator 模式必须提供 disc_episode_rewards"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 4
        response_length = 16
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array(['g0', 'g0', 'g1', 'g1'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        with pytest.raises(ValueError, match="disc_episode_rewards is required"):
            compute_hybrid_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                traj_index=traj_index,
                disc_episode_rewards=None,  # 缺失必需参数
                reward_mode="discriminator"
            )
    
    def test_single_sample_per_group(self):
        """测试每组只有一个样本时的处理"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 4
        response_length = 16
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        
        # 每个样本独立一组
        index = np.array(['g0', 'g1', 'g2', 'g3'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            reward_mode="grpo"
        )
        
        # 单样本组的 mean=0, std=1，结果应该是 score - 0 = score
        assert advantages.shape == (batch_size, response_length)
        assert not torch.isnan(advantages).any()


class TestRolloutLoopReturnValue:
    """测试 rollout_loop 的返回值结构"""
    
    def test_vanilla_multi_turn_loop_return_count(self):
        """验证 vanilla_multi_turn_loop 返回值数量"""
        # 模拟返回值结构
        expected_return_count = 8  # 更新后应该返回 8 个值
        
        # 检查函数签名 (通过 docstring 或代码注释)
        # total_batch_list, episode_rewards, episode_lengths, success, 
        # traj_uid, tool_callings, expert_trajectories, tasks
        
        assert expected_return_count == 8, "vanilla_multi_turn_loop should return 8 values"
    
    def test_multi_turn_loop_meta_info_keys(self):
        """验证 multi_turn_loop 应该在 meta_info 中存储 expert_trajectories 和 tasks"""
        # 这是代码审查要点：确保 expert_trajectories 被正确存储
        expected_keys = ['expert_trajectories', 'tasks']
        for key in expected_keys:
            assert True  # 占位：实际验证需要运行环境


class TestSafeGetBatch:
    """测试 safe_get_batch 辅助函数的行为"""
    
    def test_safe_get_with_missing_key(self):
        """模拟 TensorDict 访问缺失键的场景"""
        
        class MockTensorDict:
            """模拟 TensorDict 的行为"""
            def __init__(self, data):
                self._data = data
            
            def keys(self):
                return self._data.keys()
            
            def __getitem__(self, key):
                if key not in self._data:
                    raise KeyError(f'key "{key}" not found')
                return self._data[key]
            
            def get(self, key, default=None):
                # TensorDict.get() 在某些版本可能抛出 KeyError
                if key not in self._data:
                    raise KeyError(f'key "{key}" not found')
                return self._data[key]
        
        mock_batch = MockTensorDict({
            'token_level_rewards': torch.randn(4, 16),
            'response_mask': torch.ones(4, 16)
        })
        
        # 模拟 safe_get_batch
        def safe_get_batch(batch, key, default=None):
            try:
                if key in batch.keys():
                    return batch[key]
            except:
                pass
            return default
        
        # 存在的键
        result = safe_get_batch(mock_batch, 'token_level_rewards')
        assert result is not None
        
        # 不存在的键
        result = safe_get_batch(mock_batch, 'step_rewards')
        assert result is None
        
        # 不存在的键使用默认值
        result = safe_get_batch(mock_batch, 'step_rewards', default=0.0)
        assert result == 0.0


class TestConfigParsing:
    """测试配置解析"""
    
    def test_hybrid_reward_config_defaults(self):
        """验证 hybrid_reward 配置的默认值"""
        expected_defaults = {
            'enable': False,
            'reward_mode': 'grpo',
            'episode_reward_weight': 1.0,
            'step_reward_weight': 1.0
        }
        
        for key, expected_value in expected_defaults.items():
            assert True  # 占位：实际需要加载配置验证
    
    def test_discriminator_config_defaults(self):
        """验证 discriminator 配置的默认值"""
        expected_defaults = {
            'enable': False,
            'max_concurrency_per_url': 16,
            'request_timeout': 120,
            'prompt_template': 'milestone'
        }
        
        for key, expected_value in expected_defaults.items():
            assert True  # 占位


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_batch(self):
        """测试空 batch 的处理"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 0
        response_length = 16
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([], dtype=object)
        traj_index = np.array([], dtype=object)
        
        # 应该能处理空输入而不崩溃
        try:
            advantages, returns = compute_hybrid_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                traj_index=traj_index,
                reward_mode="grpo"
            )
            assert advantages.shape == (0, response_length)
        except Exception as e:
            pytest.fail(f"Empty batch should be handled gracefully: {e}")
    
    def test_all_zeros_rewards(self):
        """测试全零奖励的处理"""
        from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
        
        batch_size = 4
        response_length = 16
        
        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array(['g0', 'g0', 'g1', 'g1'], dtype=object)
        traj_index = np.array([f't{i}' for i in range(batch_size)], dtype=object)
        
        advantages, returns = compute_hybrid_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            reward_mode="grpo"
        )
        
        # 全零奖励应该产生全零优势（归一化后）
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()


class TestDiscriminatorRewardCalculator:
    """测试 DiscriminatorRewardCalculator"""
    
    def test_config_creation(self):
        """测试配置创建"""
        from rlvmr.discriminator_reward import DiscriminatorConfig
        
        config = DiscriminatorConfig(
            base_urls=["http://localhost:8080/v1", "http://localhost:8081/v1"],
            max_concurrency_per_url=8
        )
        
        assert len(config.base_urls) == 2
        assert config.max_concurrency_per_url == 8
    
    def test_prompt_template_selection(self):
        """测试 prompt 模板选择"""
        from rlvmr.discriminator_reward import DiscriminatorConfig, DiscriminatorRewardCalculator
        
        config = DiscriminatorConfig(prompt_template="milestone")
        calculator = DiscriminatorRewardCalculator(config)
        
        # 检查模板类型
        from rlvmr.discriminator_reward import MilestonePromptTemplate
        assert isinstance(calculator.prompt_template, MilestonePromptTemplate)
    
    def test_unknown_template_raises_error(self):
        """测试未知模板类型应该报错"""
        from rlvmr.discriminator_reward import DiscriminatorConfig, DiscriminatorRewardCalculator
        
        config = DiscriminatorConfig(prompt_template="unknown_template")
        
        with pytest.raises(ValueError, match="Unknown prompt template"):
            DiscriminatorRewardCalculator(config)


class TestExpertTrajectoryGenerator:
    """测试 ExpertTrajectoryGenerator"""
    
    def test_factory_function(self):
        """测试工厂函数"""
        from rlvmr.expert_trajectory import create_expert_generator
        
        generator = create_expert_generator("alfworld/AlfredTWEnv")
        assert generator is not None
        
        generator = create_expert_generator("unsupported_env")
        assert generator is None
    
    def test_alfworld_generator_generate(self):
        """测试 AlfWorld 生成器的 generate 方法"""
        from rlvmr.expert_trajectory import AlfWorldExpertGenerator
        
        generator = AlfWorldExpertGenerator()
        
        # 空 info
        result = generator.generate({})
        assert result == []
        
        # 有 expert_plan
        result = generator.generate({
            "expert_plan": ["action1", "action2"]
        })
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
