"""
Tests for MilestoneGenerator

测试里程碑动态生成器的功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Test data
SAMPLE_EXPERT_TRAJECTORY = [
    {"observation": "You are in the kitchen. You see a cabinet.", "action": "go to cabinet 1"},
    {"observation": "The cabinet 1 is open. You see a pan.", "action": "take pan 1 from cabinet 1"},
    {"observation": "You pick up the pan 1 from the cabinet 1.", "action": "go to stoveburner 1"},
    {"observation": "You arrive at stoveburner 1.", "action": "put pan 1 in/on stoveburner 1"},
    {"observation": "You put the pan 1 in/on the stoveburner 1.", "action": ""},
]

SAMPLE_TASK = "put a pan in stoveburner"

MOCK_LLM_RESPONSE = """
{
  "milestones": [
    {"id": "M1", "name": "找到容器位置", "phi": 0.2, "criteria": "到达柜子位置"},
    {"id": "M2", "name": "获取目标物品", "phi": 0.4, "criteria": "从柜子中取出锅"},
    {"id": "M3", "name": "找到目标位置", "phi": 0.6, "criteria": "到达炉灶位置"},
    {"id": "M4", "name": "放置物品", "phi": 0.8, "criteria": "将锅放在炉灶上"},
    {"id": "M5", "name": "任务完成", "phi": 1.0, "criteria": "锅已成功放置在炉灶上"}
  ],
  "reasoning": "基于专家轨迹的关键步骤划分"
}
"""


class TestMilestoneGenerator:
    """测试 MilestoneGenerator 类"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_LLM_RESPONSE
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_format_trajectory(self):
        """测试轨迹格式化"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            
            formatted = generator._format_trajectory(SAMPLE_EXPERT_TRAJECTORY)
            
            assert "Step 1:" in formatted
            assert "kitchen" in formatted
            assert "go to cabinet 1" in formatted
    
    def test_format_empty_trajectory(self):
        """测试空轨迹格式化"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            
            formatted = generator._format_trajectory([])
            assert formatted == "[]"
    
    def test_parse_valid_response(self):
        """测试解析有效的 LLM 响应"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            
            result = generator._parse_response(MOCK_LLM_RESPONSE)
            
            assert result.success
            assert len(result.milestones) == 5
            assert result.milestones[0]["id"] == "M1"
            assert result.milestones[-1]["phi"] == 1.0
    
    def test_parse_invalid_response(self):
        """测试解析无效的 LLM 响应"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            
            result = generator._parse_response("This is not valid JSON")
            
            assert not result.success
            assert len(result.milestones) == 0
    
    def test_get_default_milestones(self):
        """测试默认里程碑生成"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
                num_milestones=5,
            )
            
            defaults = generator._get_default_milestones()
            
            assert len(defaults) == 5
            assert defaults[0]["id"] == "M1"
            assert defaults[-1]["phi"] == 1.0
    
    def test_generate_with_mock_client(self, mock_openai_client):
        """测试使用 mock client 生成里程碑"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI', return_value=mock_openai_client):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            generator.client = mock_openai_client
            
            result = generator.generate(SAMPLE_TASK, SAMPLE_EXPERT_TRAJECTORY)
            
            assert result.success
            assert len(result.milestones) == 5
    
    def test_generate_without_expert_trajectory(self):
        """测试没有专家轨迹时返回默认里程碑"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
            )
            
            result = generator.generate(SAMPLE_TASK, [])
            
            assert not result.success
            assert "No expert trajectory" in result.reasoning
            assert len(result.milestones) == 5  # 默认里程碑
    
    def test_build_prompt(self):
        """测试 prompt 构建"""
        from rlvmr.milestone.generator import MilestoneGenerator
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = MilestoneGenerator(
                base_url="http://test",
                model="test-model",
                num_milestones=5,
            )
            
            prompt = generator._build_prompt(SAMPLE_TASK, SAMPLE_EXPERT_TRAJECTORY)
            
            assert SAMPLE_TASK in prompt
            assert "Step 1:" in prompt
            assert "5" in prompt  # num_milestones


class TestGeneratedMilestones:
    """测试 GeneratedMilestones 数据类"""
    
    def test_success_result(self):
        """测试成功结果"""
        from rlvmr.milestone.generator import GeneratedMilestones
        
        result = GeneratedMilestones(
            milestones=[{"id": "M1", "name": "test", "phi": 0.5, "criteria": "test"}],
            reasoning="test",
            success=True,
        )
        
        assert result.success
        assert len(result.milestones) == 1
    
    def test_failure_result(self):
        """测试失败结果"""
        from rlvmr.milestone.generator import GeneratedMilestones
        
        result = GeneratedMilestones(
            milestones=[],
            reasoning="error",
            success=False,
        )
        
        assert not result.success
        assert len(result.milestones) == 0


class TestCreateGeneratorFromConfig:
    """测试配置创建工厂函数"""
    
    def test_create_with_valid_config(self):
        """测试使用有效配置创建生成器"""
        from omegaconf import OmegaConf
        from rlvmr.milestone.generator import create_milestone_generator_from_config
        
        config = OmegaConf.create({
            "algorithm": {
                "milestone_gae": {
                    "generator": {
                        "enable": True,
                        "num_milestones": 5,
                        "llm": {
                            "base_url": "http://test",
                            "model": "test-model",
                            "api_key": "test-key",
                            "temperature": 0.3,
                        }
                    }
                }
            }
        })
        
        with patch('rlvmr.milestone.generator.OpenAI'):
            generator = create_milestone_generator_from_config(config)
            
            assert generator is not None
            assert generator.num_milestones == 5
    
    def test_create_with_disabled_config(self):
        """测试禁用配置时返回 None"""
        from omegaconf import OmegaConf
        from rlvmr.milestone.generator import create_milestone_generator_from_config
        
        config = OmegaConf.create({
            "algorithm": {
                "milestone_gae": {
                    "generator": {
                        "enable": False,
                        "llm": {
                            "base_url": "http://test",
                            "model": "test-model",
                        }
                    }
                }
            }
        })
        
        generator = create_milestone_generator_from_config(config)
        assert generator is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
