"""
Unit tests for Expert Trajectory Generator module
"""

import os
import sys
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rlvmr.expert_trajectory import (
    ExpertTrajectoryGeneratorBase,
    AlfWorldExpertGenerator,
    NullExpertGenerator,
    create_expert_generator,
    register_expert_generator,
)


class TestAlfWorldExpertGenerator:
    """Tests for AlfWorldExpertGenerator"""
    
    def test_is_supported(self):
        """Test environment support detection"""
        generator = AlfWorldExpertGenerator()
        
        assert generator.is_supported("alfworld/AlfredTWEnv") == True
        assert generator.is_supported("ALFWorld") == True
        assert generator.is_supported("webshop") == False
        assert generator.is_supported("sokoban") == False
    
    def test_generate_empty_info(self):
        """Test generation with empty info"""
        generator = AlfWorldExpertGenerator()
        
        result = generator.generate({})
        assert result == []
    
    def test_generate_with_expert_plan(self):
        """Test generation with expert_plan actions"""
        generator = AlfWorldExpertGenerator()
        
        env_info = {
            "expert_plan": ["go to desk 1", "take lamp 1 from desk 1", "turn on lamp 1"]
        }
        
        result = generator.generate(env_info)
        
        assert len(result) == 3
        assert result[0]["action"] == "go to desk 1"
        assert result[1]["action"] == "take lamp 1 from desk 1"
        assert result[2]["action"] == "turn on lamp 1"
    
    def test_generate_with_pregenerated_trajectory(self):
        """Test that pre-generated trajectories are returned as-is"""
        generator = AlfWorldExpertGenerator()
        
        pre_generated = [
            {"observation": "obs1", "action": "action1"},
            {"observation": "obs2", "action": "action2"},
        ]
        
        env_info = {"expert_plan": pre_generated}
        
        result = generator.generate(env_info)
        
        assert result == pre_generated
    
    def test_format_trajectory(self):
        """Test trajectory formatting"""
        generator = AlfWorldExpertGenerator()
        
        trajectory = [
            {"observation": "You are in room 1", "action": "go to desk"},
            {"observation": "You see a desk", "action": "take lamp"},
        ]
        
        formatted = generator.format_trajectory(trajectory)
        
        assert "Step 0" in formatted
        assert "Step 1" in formatted
        assert "go to desk" in formatted
        assert "take lamp" in formatted


class TestNullExpertGenerator:
    """Tests for NullExpertGenerator (fallback)"""
    
    def test_is_supported_always_true(self):
        """Test that NullExpertGenerator supports any environment"""
        generator = NullExpertGenerator()
        
        assert generator.is_supported("any_env") == True
        assert generator.is_supported("random_name") == True
    
    def test_generate_returns_empty(self):
        """Test that generation always returns empty list"""
        generator = NullExpertGenerator()
        
        assert generator.generate({}) == []
        assert generator.generate({"expert_plan": ["action1"]}) == []


class TestFactoryFunction:
    """Tests for create_expert_generator factory function"""
    
    def test_create_alfworld_generator(self):
        """Test factory creates AlfWorldExpertGenerator for alfworld"""
        generator = create_expert_generator("alfworld/AlfredTWEnv")
        
        assert generator is not None
        assert isinstance(generator, AlfWorldExpertGenerator)
    
    def test_create_with_custom_params(self):
        """Test factory passes parameters to generator"""
        generator = create_expert_generator("alfworld", max_steps=100, debug=True)
        
        assert generator is not None
        assert generator.max_steps == 100
        assert generator.debug == True
    
    def test_create_unsupported_returns_none(self):
        """Test factory returns None for unsupported environments"""
        generator = create_expert_generator("unsupported_env_name")
        
        assert generator is None


class TestRegisterGenerator:
    """Tests for register_expert_generator function"""
    
    def test_register_custom_generator(self):
        """Test registering a custom generator"""
        
        class CustomGenerator(ExpertTrajectoryGeneratorBase):
            def is_supported(self, env_name: str) -> bool:
                return "custom" in env_name.lower()
            
            def generate(self, env_info):
                return [{"observation": "custom", "action": "custom_action"}]
        
        register_expert_generator("custom", CustomGenerator)
        
        generator = create_expert_generator("custom_env")
        
        assert generator is not None
        assert isinstance(generator, CustomGenerator)
        
        result = generator.generate({})
        assert len(result) == 1
        assert result[0]["action"] == "custom_action"


class TestExpertTrajectoryGeneratorBase:
    """Tests for base class interface"""
    
    def test_abstract_methods(self):
        """Test that abstract methods raise errors"""
        
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            ExpertTrajectoryGeneratorBase()
