"""
单元测试：Discriminator Reward Calculator

测试内容：
1. DiscriminatorConfig 配置类
2. MilestonePromptTemplate prompt 构建和解析
3. DiscriminatorRewardCalculator 初始化（不需要真实 API）
"""

import pytest
import numpy as np
import json
import sys
import os

# 添加项目根目录到 path（向上两级：tests/rlvmr -> tests -> agent-verl）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from rlvmr.discriminator_reward import (
    DiscriminatorConfig,
    MilestonePromptTemplate,
    DiscriminatorRewardCalculator,
    create_discriminator_from_config,
)


class TestDiscriminatorConfig:
    """测试 DiscriminatorConfig 配置类"""
    
    def test_default_config(self):
        config = DiscriminatorConfig()
        assert config.base_urls == ["http://127.0.0.1:8080/v1"]
        assert config.api_key == "EMPTY"
        assert config.max_concurrency_per_url == 16
        assert config.request_timeout == 120
        assert config.prompt_template == "milestone"
    
    def test_custom_config(self):
        config = DiscriminatorConfig(
            base_urls=["http://localhost:8000/v1", "http://localhost:8001/v1"],
            api_key="test_key",
            max_concurrency_per_url=32,
            request_timeout=60
        )
        assert len(config.base_urls) == 2
        assert config.api_key == "test_key"
        assert config.max_concurrency_per_url == 32


class TestMilestonePromptTemplate:
    """测试 MilestonePromptTemplate"""
    
    def setup_method(self):
        self.template = MilestonePromptTemplate()
    
    def test_build_prompt_with_expert(self):
        policy_traj = [
            {"observation": "You are in the kitchen", "action": "go to fridge"},
            {"observation": "You are at the fridge", "action": "open fridge"},
        ]
        expert_traj = [
            {"observation": "You are in the kitchen", "action": "go to fridge"},
            {"observation": "You are at the fridge", "action": "open fridge"},
            {"observation": "The fridge is open", "action": "take apple"},
        ]
        
        prompt = self.template.build_prompt(policy_traj, expert_traj)
        
        assert "Policy Trajectory" in prompt
        assert "Expert Trajectory" in prompt
        assert "go to fridge" in prompt
        assert "episode_score" in prompt
    
    def test_build_prompt_without_expert(self):
        policy_traj = [
            {"observation": "You are in the kitchen", "action": "look around"},
        ]
        
        prompt = self.template.build_prompt(policy_traj, None)
        
        assert "Policy Trajectory" in prompt
        assert "Not provided" in prompt
    
    def test_parse_valid_response(self):
        response = '''```json
{
  "episode_score": 7,
  "step_scores": [1, 0, 1, 1]
}
```'''
        
        episode_score, step_scores = self.template.parse_response(response)
        
        assert episode_score == 7.0
        assert step_scores == [1.0, 0.0, 1.0, 1.0]
    
    def test_parse_response_without_markdown(self):
        response = '{"episode_score": 5, "step_scores": [0, 1]}'
        
        episode_score, step_scores = self.template.parse_response(response)
        
        assert episode_score == 5.0
        assert step_scores == [0.0, 1.0]
    
    def test_parse_invalid_response(self):
        response = "This is not valid JSON"
        
        episode_score, step_scores = self.template.parse_response(response)
        
        assert episode_score == 0.0
        assert step_scores == []


class TestDiscriminatorRewardCalculator:
    """测试 DiscriminatorRewardCalculator（不需要真实 API）"""
    
    def test_init_with_default_config(self):
        config = DiscriminatorConfig()
        calculator = DiscriminatorRewardCalculator(config)
        
        assert calculator.config == config
        assert not calculator._initialized  # 延迟初始化
        assert isinstance(calculator.prompt_template, MilestonePromptTemplate)
    
    def test_init_with_invalid_template(self):
        config = DiscriminatorConfig(prompt_template="invalid")
        
        with pytest.raises(ValueError, match="Unknown prompt template"):
            DiscriminatorRewardCalculator(config)


class TestCreateDiscriminatorFromConfig:
    """测试 create_discriminator_from_config 工厂函数"""
    
    def test_disabled_returns_none(self):
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            "algorithm": {
                "discriminator": {
                    "enable": False,
                    "base_urls": ["http://localhost:8080/v1"],
                    "api_key": "EMPTY",
                    "model_name": "discriminator",
                    "max_concurrency_per_url": 16,
                    "request_timeout": 120,
                    "prompt_template": "milestone",
                    "use_expert": False
                }
            }
        })
        
        result = create_discriminator_from_config(config)
        assert result is None
    
    def test_enabled_returns_calculator(self):
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            "algorithm": {
                "discriminator": {
                    "enable": True,
                    "base_urls": ["http://localhost:8080/v1"],
                    "api_key": "test_key",
                    "model_name": "test_model",
                    "max_concurrency_per_url": 8,
                    "request_timeout": 60,
                    "prompt_template": "milestone",
                    "use_expert": True
                }
            }
        })
        
        result = create_discriminator_from_config(config)
        
        assert result is not None
        assert isinstance(result, DiscriminatorRewardCalculator)
        assert result.config.api_key == "test_key"
        assert result.config.max_concurrency_per_url == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
