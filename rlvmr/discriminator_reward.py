"""
Discriminator Reward Calculator

使用 OpenAI 兼容 API 调用外部 Discriminator 模型，评估策略轨迹质量。
支持多并发调用和多 URL 负载均衡。

Usage:
    from rlvmr.discriminator_reward import DiscriminatorRewardCalculator, DiscriminatorConfig
    
    config = DiscriminatorConfig(
        base_urls=['http://127.0.0.1:8080/v1'],
        max_concurrency_per_url=16
    )
    calculator = DiscriminatorRewardCalculator(config)
    
    episode_scores, step_scores = await calculator.compute_rewards(policy_trajectories)
"""

import asyncio
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


@dataclass
class DiscriminatorConfig:
    """Discriminator 配置类"""
    base_urls: List[str] = field(default_factory=lambda: ["http://127.0.0.1:8080/v1"])
    api_key: str = "EMPTY"
    model_name: str = "discriminator"
    max_concurrency_per_url: int = 16
    request_timeout: int = 120
    prompt_template: str = "milestone"  # "milestone" | "custom"
    use_expert: bool = False


class PromptTemplateBase(ABC):
    """Prompt 模板基类，便于扩展新的评估方式"""
    
    @abstractmethod
    def build_prompt(
        self, 
        policy_trajectory: List[Dict], 
        expert_trajectory: Optional[List[Dict]] = None
    ) -> str:
        """构建评估 prompt"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Tuple[float, List[float]]:
        """
        解析模型响应
        Returns:
            episode_score: float, 整体评分
            step_scores: List[float], 每步评分
        """
        pass


class MilestonePromptTemplate(PromptTemplateBase):
    """Milestone 评估模板：判断每步是否为里程碑状态变化"""
    
    PROMPT_TEMPLATE = """### Role Definition
You are the **Environment itself** (a State Machine). Your role is to objectively judge whether each step in the Policy Trajectory triggers a **valid state transition** towards the goal.

### Core Standard: What is a "Milestone"?
A **Milestone State** is: An irreversible or necessary change in the environment state that aligns with the key nodes in the Expert Trajectory.
Examples:
- Change in location (e.g., "go to kitchen")
- Change in inventory (e.g., "pick up apple")
- State change of objects (e.g., "open fridge", "heat apple")

### Policy Trajectory
{policy_trajectory}

### Expert Trajectory (Reference)
{expert_trajectory}

### Scoring Criteria
1. **Episode Score (0-10)**: Overall progress towards the goal
   - 0-3: Failed or made no progress
   - 4-6: Partial progress with some correct steps
   - 7-9: Mostly correct with minor inefficiencies
   - 10: Perfect execution matching expert
   
2. **Step Scores**: For each step, output 1 if it triggers a milestone state change, 0 otherwise

### Output Format (JSON only, no explanation)
```json
{{
  "episode_score": <0-10>,
  "step_scores": [<0 or 1>, ...]
}}
```"""

    def build_prompt(
        self, 
        policy_trajectory: List[Dict], 
        expert_trajectory: Optional[List[Dict]] = None
    ) -> str:
        policy_str = self._format_trajectory(policy_trajectory)
        expert_str = self._format_trajectory(expert_trajectory) if expert_trajectory else "Not provided"
        
        return self.PROMPT_TEMPLATE.format(
            policy_trajectory=policy_str,
            expert_trajectory=expert_str
        )
    
    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """格式化轨迹为可读字符串"""
        if not trajectory:
            return "Empty trajectory"
        
        lines = []
        for i, step in enumerate(trajectory):
            obs = step.get("observation", "")
            action = step.get("action", "")
            lines.append(f"Step {i+1}: Action='{action}', Obs='{obs[:100]}...'")
        return "\n".join(lines)
    
    def parse_response(self, response: str) -> Tuple[float, List[float]]:
        """解析 JSON 响应"""
        try:
            # 尝试从响应中提取 JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                episode_score = float(data.get("episode_score", 0))
                step_scores = [float(s) for s in data.get("step_scores", [])]
                return episode_score, step_scores
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Discriminator] Failed to parse response: {e}")
        
        # 解析失败返回默认值
        return 0.0, []


class DiscriminatorRewardCalculator:
    """
    Discriminator 奖励计算器
    
    通过 OpenAI 兼容 API 调用外部 Discriminator 模型，
    评估策略轨迹与专家轨迹的差异，生成 episode 和 step 级别的奖励。
    """
    
    def __init__(self, config: DiscriminatorConfig):
        self.config = config
        self.clients = []
        self._initialized = False
        
        # 选择 prompt 模板
        if config.prompt_template == "milestone":
            self.prompt_template = MilestonePromptTemplate()
        else:
            raise ValueError(f"Unknown prompt template: {config.prompt_template}")
    
    def _ensure_initialized(self):
        """延迟初始化 OpenAI 客户端（避免在导入时就需要 openai 包）"""
        if self._initialized:
            return
        
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for DiscriminatorRewardCalculator. "
                "Install it with: pip install openai"
            )
        
        for url in self.config.base_urls:
            client = AsyncOpenAI(
                base_url=url,
                api_key=self.config.api_key,
                timeout=self.config.request_timeout
            )
            self.clients.append(client)
            # 注意：不在这里创建 Semaphore，避免事件循环绑定问题
        
        self._initialized = True
    
    async def compute_rewards(
        self,
        policy_trajectories: List[List[Dict]],
        expert_trajectories: Optional[List[List[Dict]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 Discriminator 奖励
        
        Args:
            policy_trajectories: List of policy trajectories, each is a list of step dicts
            expert_trajectories: Optional list of expert trajectories for reference
        
        Returns:
            episode_scores: (batch_size,) Episode-level scores
            step_scores: (batch_size,) Aggregated step-level scores
        """
        self._ensure_initialized()
        
        batch_size = len(policy_trajectories)
        
        # 每次调用时创建新的 semaphores（绑定到当前事件循环，避免跨循环使用）
        semaphores = [
            asyncio.Semaphore(self.config.max_concurrency_per_url) 
            for _ in self.clients
        ]
        
        # 准备专家轨迹
        if expert_trajectories is None:
            expert_trajectories = [None] * batch_size
        
        # 构建所有请求
        tasks = []
        for i, (policy_traj, expert_traj) in enumerate(zip(policy_trajectories, expert_trajectories)):
            tasks.append(self._compute_single_reward(i, policy_traj, expert_traj, semaphores))
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        episode_scores = np.zeros(batch_size, dtype=np.float32)
        step_scores = np.zeros(batch_size, dtype=np.float32)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[Discriminator] Request {i} failed: {result}")
                continue
            
            ep_score, st_scores = result
            episode_scores[i] = ep_score / 10.0  # 归一化到 [0, 1]
            step_scores[i] = np.mean(st_scores) if st_scores else 0.0
        
        return episode_scores, step_scores
    
    async def _compute_single_reward(
        self,
        index: int,
        policy_trajectory: List[Dict],
        expert_trajectory: Optional[List[Dict]],
        semaphores: List[asyncio.Semaphore]
    ) -> Tuple[float, List[float]]:
        """计算单个轨迹的奖励"""
        # 选择客户端（负载均衡）
        client_idx = index % len(self.clients)
        client = self.clients[client_idx]
        semaphore = semaphores[client_idx]
        
        # 构建 prompt
        # 确保 expert trajectory 格式正确
        formatted_expert = format_expert_trajectory(expert_trajectory) if expert_trajectory else None
        prompt = self.prompt_template.build_prompt(policy_trajectory, formatted_expert)
        
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512
                )
                content = response.choices[0].message.content
                return self.prompt_template.parse_response(content)
            except Exception as e:
                print(f"[Discriminator] API call failed for index {index}: {e}")
                return 0.0, []
    
    def compute_rewards_sync(
        self,
        policy_trajectories: List[List[Dict]],
        expert_trajectories: Optional[List[List[Dict]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """同步版本的奖励计算（便于非异步环境调用）"""
        return asyncio.run(self.compute_rewards(policy_trajectories, expert_trajectories))


def create_discriminator_from_config(config) -> Optional[DiscriminatorRewardCalculator]:
    """
    从 OmegaConf 配置创建 Discriminator 实例
    
    Args:
        config: OmegaConf config object with algorithm.discriminator section
    
    Returns:
        DiscriminatorRewardCalculator instance or None if disabled
    """
    disc_cfg = config.algorithm.discriminator
    
    # 打印配置用于调试
    print(f"[Discriminator] Config received:")
    print(f"  enable: {disc_cfg.enable} (type: {type(disc_cfg.enable).__name__})")
    print(f"  model_name: {disc_cfg.model_name}")
    print(f"  base_urls: {disc_cfg.base_urls} (type: {type(disc_cfg.base_urls).__name__})")
    
    # 检查 enable 是否为有效布尔值
    enable = disc_cfg.enable
    if isinstance(enable, str):
        enable = enable.lower() in ('true', '1', 'yes')
        print(f"  [WARN] enable was string, converted to: {enable}")
    
    if not enable:
        print(f"[Discriminator] Disabled, skipping initialization")
        return None
    
    # 处理 base_urls - 可能是列表、ListConfig 或字符串
    base_urls = disc_cfg.base_urls
    if isinstance(base_urls, str):
        # 尝试解析 JSON 格式 '["url1","url2"]'
        if base_urls.startswith('[') and base_urls.endswith(']'):
            try:
                base_urls = json.loads(base_urls)
                print(f"  [INFO] base_urls parsed from JSON string: {base_urls}")
            except json.JSONDecodeError:
                # 尝试简单分割 '[url1,url2]'
                base_urls = base_urls[1:-1].split(',')
                base_urls = [url.strip().strip('"').strip("'") for url in base_urls]
                print(f"  [INFO] base_urls parsed from bracket string: {base_urls}")
        else:
            # 单个 URL 字符串
            base_urls = [base_urls]
            print(f"  [INFO] base_urls was single string, wrapped: {base_urls}")
    else:
        # ListConfig 或 list
        base_urls = list(base_urls)
    
    print(f"[Discriminator] Creating calculator with:")
    print(f"  model_name: {disc_cfg.model_name}")
    print(f"  base_urls: {base_urls}")
    
    return DiscriminatorRewardCalculator(
        DiscriminatorConfig(
            base_urls=base_urls,
            api_key=disc_cfg.api_key,
            model_name=disc_cfg.model_name,
            max_concurrency_per_url=disc_cfg.max_concurrency_per_url,
            request_timeout=disc_cfg.request_timeout,
            prompt_template=disc_cfg.prompt_template,
            use_expert=disc_cfg.use_expert
        )
    )


# ==================== Trajectory Formatting Functions ====================

def format_policy_trajectory(
    trajectory: List[Dict],
    task: str = ""
) -> Dict:
    """
    将原始策略轨迹格式化为 Discriminator 输入格式。
    
    关键功能：
    1. 仅提取 <action> 标签内容，去除 <think>, <planning> 等推理标签
    2. 从 anchor_obs 字段获取观测信息
    3. 跳过 active_masks=False 的填充步骤
    
    Args:
        trajectory: 原始轨迹数据列表，每个 step 包含:
            - full_output: str, 完整模型输出（包含各种标签）
            - active_masks: bool, 是否为有效步骤
            - anchor_obs: str/List[str], 当前步骤的观测
        task: 任务描述字符串
    
    Returns:
        格式化的轨迹字典: {"task": str, "traj": List[Dict]}
        其中 traj 每个元素为 {"obs": str, "action": str}
    """
    formatted_steps = []
    
    for step in trajectory:
        # 跳过非活跃步骤（填充步骤）
        if not step.get('active_masks', False):
            continue
        
        full_output = step.get('full_output', '')
        
        # 提取 action: 优先从 <action> 标签提取
        action = ""
        action_match = re.search(r'<action>(.*?)</action>', full_output, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # 如果没有 action 标签，尝试清理其他标签后使用
            # 移除 <think>...</think>, <planning>...</planning> 等
            cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL)
            cleaned = re.sub(r'<planning>.*?</planning>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<explore>.*?</explore>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<reflection>.*?</reflection>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<monitor>.*?</monitor>', '', cleaned, flags=re.DOTALL)
            action = cleaned.strip()
        
        # 获取观测
        obs = step.get('anchor_obs', '')
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            obs = obs[0] if isinstance(obs[0], str) else str(obs[0])
        elif not isinstance(obs, str):
            obs = str(obs) if obs else ""
        
        formatted_steps.append({
            'obs': obs,
            'action': action
        })
    
    return {
        'task': task,
        'traj': formatted_steps
    }


def format_expert_trajectory(expert_traj: List[Dict]) -> List[Dict]:
    """
    格式化 Expert 轨迹为标准格式。
    
    确保每个步骤包含 observation 和 action 字段。
    
    Args:
        expert_traj: Expert 轨迹，每个步骤可能包含不同字段名
    
    Returns:
        格式化的 Expert 轨迹列表，每个元素为:
        {"observation": str, "action": str}
    """
    if not expert_traj:
        return []
    
    formatted = []
    for step in expert_traj:
        # 兼容不同的字段名
        obs = step.get('observation', step.get('obs', ''))
        action = step.get('action', '')
        
        formatted.append({
            'observation': obs if isinstance(obs, str) else str(obs),
            'action': action if isinstance(action, str) else str(action)
        })
    
    return formatted
