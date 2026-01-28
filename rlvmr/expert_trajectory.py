"""
Expert Trajectory Generator Module

提供模块化的专家轨迹生成接口，支持不同环境的扩展。

Usage:
    from rlvmr.expert_trajectory import create_expert_generator
    
    generator = create_expert_generator("alfworld")
    if generator:
        trajectories = generator.generate(env_info)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ExpertStep:
    """专家轨迹中的单步"""
    observation: str
    action: str
    reward: float = 0.0
    done: bool = False
    info: Optional[Dict] = None


class ExpertTrajectoryGeneratorBase(ABC):
    """
    专家轨迹生成器基类（策略模式）
    
    不同环境需要实现自己的子类来生成专家轨迹。
    """
    
    @abstractmethod
    def generate(self, env_info: Dict) -> List[Dict[str, Any]]:
        """
        生成专家轨迹
        
        Args:
            env_info: 环境信息，包含生成专家轨迹所需的数据
                     不同环境可能需要不同的信息
        
        Returns:
            专家轨迹，格式为 [{"observation": str, "action": str, ...}, ...]
        """
        pass
    
    @abstractmethod
    def is_supported(self, env_name: str) -> bool:
        """
        检查是否支持该环境
        
        Args:
            env_name: 环境名称
        
        Returns:
            True if supported, False otherwise
        """
        pass
    
    def format_trajectory(self, trajectory: List[Dict]) -> str:
        """
        将轨迹格式化为可读字符串
        
        Args:
            trajectory: 专家轨迹
        
        Returns:
            格式化的字符串
        """
        lines = []
        for i, step in enumerate(trajectory):
            obs = step.get("observation", "")[:100]
            action = step.get("action", "")
            lines.append(f"Step {i}: Action='{action}', Obs='{obs}...'")
        return "\n".join(lines)


class AlfWorldExpertGenerator(ExpertTrajectoryGeneratorBase):
    """
    ALFWorld 环境的专家轨迹生成器
    
    使用 TextWorld 的 expert_plan 信息生成专家轨迹。
    """
    
    def __init__(self, max_steps: int = 50, debug: bool = False):
        self.max_steps = max_steps
        self.debug = debug
    
    def is_supported(self, env_name: str) -> bool:
        return "alfworld" in env_name.lower()
    
    def generate(self, env_info: Dict) -> List[Dict[str, Any]]:
        """
        使用 TextWorld 的 expert_plan 生成专家轨迹
        
        Args:
            env_info: 需要包含以下键:
                - 'expert_plan': List[str] - 当前的专家计划
                - 'env' (可选): 环境实例，用于执行动作
                - 'admissible_commands' (可选): 当前可用命令
        
        Returns:
            专家轨迹
        """
        trajectory = []
        
        # 从 info 中获取预计算的 expert_plan
        expert_plan = env_info.get('expert_plan', [])
        
        if not expert_plan:
            if self.debug:
                print("[AlfWorldExpert] No expert_plan provided")
            return trajectory
        
        # 如果提供了完整的预生成轨迹，直接返回
        if isinstance(expert_plan, list) and len(expert_plan) > 0:
            if isinstance(expert_plan[0], dict) and 'observation' in expert_plan[0]:
                return expert_plan
        
        # 如果只有动作列表，转换为轨迹格式
        for i, action in enumerate(expert_plan):
            trajectory.append({
                "observation": f"Step {i}",
                "action": action if isinstance(action, str) else str(action),
            })
        
        return trajectory
    
    def generate_from_env(
        self, 
        env: Any,
        game_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        通过实际执行环境来生成专家轨迹（更完整但更慢）
        
        Args:
            env: TextWorld 环境实例
            game_file: 游戏文件路径（用于 reset）
        
        Returns:
            专家轨迹
        """
        trajectory = []
        
        try:
            # Reset 环境
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
            
            done = False
            steps = 0
            
            while not done and steps < self.max_steps:
                # 获取 expert action
                expert_plan = info.get('extra.expert_plan', [])
                if not expert_plan:
                    break
                
                # 处理 batch 维度
                if isinstance(expert_plan, list) and len(expert_plan) > 0:
                    if isinstance(expert_plan[0], list):
                        current_plan = expert_plan[0]
                    else:
                        current_plan = expert_plan
                else:
                    break
                
                if not current_plan:
                    break
                
                expert_action = current_plan[0]
                
                # 记录轨迹
                current_obs = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
                trajectory.append({
                    "observation": current_obs,
                    "action": expert_action,
                })
                
                # 执行动作
                step_result = env.step([expert_action] if isinstance(obs, list) else expert_action)
                
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                
                if isinstance(done, (list, tuple)):
                    done = done[0]
                
                # 检查胜利
                won = info.get('won', [False])
                if isinstance(won, list):
                    won = won[0]
                if won:
                    break
                
                steps += 1
                
        except Exception as e:
            if self.debug:
                print(f"[AlfWorldExpert] Error generating trajectory: {e}")
        
        return trajectory


class NullExpertGenerator(ExpertTrajectoryGeneratorBase):
    """
    空实现，用于不支持专家轨迹的环境
    """
    
    def is_supported(self, env_name: str) -> bool:
        return True  # 作为 fallback
    
    def generate(self, env_info: Dict) -> List[Dict[str, Any]]:
        return []


# 注册表：环境名 -> 生成器类
_EXPERT_GENERATORS = {
    "alfworld": AlfWorldExpertGenerator,
}


def register_expert_generator(env_pattern: str, generator_cls: type):
    """
    注册新的专家轨迹生成器
    
    Args:
        env_pattern: 环境名称模式（用于匹配）
        generator_cls: 生成器类
    """
    _EXPERT_GENERATORS[env_pattern] = generator_cls


def create_expert_generator(
    env_name: str,
    **kwargs
) -> Optional[ExpertTrajectoryGeneratorBase]:
    """
    工厂函数：根据环境名称创建专家轨迹生成器
    
    Args:
        env_name: 环境名称
        **kwargs: 传递给生成器构造函数的参数
    
    Returns:
        ExpertTrajectoryGeneratorBase 实例，如果不支持则返回 None
    """
    env_name_lower = env_name.lower()
    
    for pattern, generator_cls in _EXPERT_GENERATORS.items():
        if pattern in env_name_lower:
            return generator_cls(**kwargs)
    
    return None


def create_expert_generator_from_config(config) -> Optional[ExpertTrajectoryGeneratorBase]:
    """
    从配置创建专家轨迹生成器
    
    Args:
        config: OmegaConf 配置对象
    
    Returns:
        ExpertTrajectoryGeneratorBase 实例
    """
    expert_cfg = config.algorithm.expert
    
    if not expert_cfg.enable:
        return None
    
    env_name = config.env.env_name
    max_steps = config.env.max_steps
    
    return create_expert_generator(env_name, max_steps=max_steps)
