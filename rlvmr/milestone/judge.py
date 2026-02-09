"""
Milestone Judge - LLM-based trajectory milestone evaluation

Uses an LLM to evaluate which milestones have been achieved at each step
of a trajectory. This replaces traditional Critic networks with
LLM-as-Critic approach.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class JudgmentResult:
    """单条轨迹的判定结果"""
    step_phis: List[float]  # 每步的势能值
    highest_milestones: List[str]  # 每步达到的最高里程碑
    final_success: bool
    reasoning: str


class MilestoneJudge:
    """
    基于 LLM 的里程碑判定器
    
    对整条轨迹进行一次性判定，输出每个步骤的势能值 Φ(s)。
    """
    
    def __init__(
        self,
        base_url: str,
        model: str,
        milestones: List[Dict[str, Any]],
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """
        初始化 MilestoneJudge
        
        Args:
            base_url: LLM API 地址
            model: 模型名称
            milestones: 里程碑列表，每个元素包含 id, name, phi, criteria
            api_key: API 密钥
            temperature: 采样温度
            max_retries: 最大重试次数
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.milestones = milestones
        self.temperature = temperature
        self.max_retries = max_retries
        
        # 构建里程碑 ID 到 phi 的映射
        self.milestone_to_phi = {"M0": 0.0}
        for m in milestones:
            self.milestone_to_phi[m["id"]] = m["phi"]
    
    def set_milestones(self, milestones: List[Dict[str, Any]]):
        """
        动态设置里程碑列表（用于支持 LLM 生成的里程碑）
        
        Args:
            milestones: 里程碑列表，每个元素包含 id, name, phi, criteria
        """
        self.milestones = milestones
        
        # 重建映射
        self.milestone_to_phi = {"M0": 0.0}
        for m in milestones:
            self.milestone_to_phi[m["id"]] = m["phi"]
    
    def _build_prompt(
        self,
        task_description: str,
        trajectory: List[Dict[str, str]],
    ) -> str:
        """构建 Judge 的 Prompt"""
        
        # 里程碑清单
        milestone_list = "\n".join([
            f'{m["id"]} (Φ={m["phi"]}): {m["name"]} — 判定标准：{m["criteria"]}'
            for m in self.milestones
        ])
        
        # 轨迹步骤
        steps_str = ""
        for i, step in enumerate(trajectory, 1):
            action = step.get("action", "N/A")
            observation = step.get("observation", "N/A")
            steps_str += f"\nStep {i}:\n  Action: {action}\n  Observation: {observation}\n"
        
        prompt = f"""你是一个任务进度评估器。

## 任务描述
{task_description}

## 里程碑清单
M0 (Φ=0.0): 尚未开始 — 判定标准：未达成任何里程碑
{milestone_list}

## Agent 执行轨迹
{steps_str}

## 任务

请对每个步骤判断已达成的最高里程碑。

输出格式 (严格 JSON):
{{
  "judgments": [
    {{"step": 1, "highest_milestone": "M0", "phi": 0.0}},
    {{"step": 2, "highest_milestone": "M1", "phi": 0.15}},
    ...
  ],
  "final_success": true/false,
  "reasoning": "简要说明判断依据"
}}

注意：
1. M0 表示尚未达成任何里程碑，phi=0.0
2. 里程碑通常是单调递增的（偶尔可能因错误动作回退）
3. 只有最终确认任务成功才能达到最高里程碑 (phi=1.0)
4. 必须输出有效的 JSON 格式"""

        return prompt
    
    def _parse_response(self, response_text: str, num_steps: int) -> JudgmentResult:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        judgments = data.get("judgments", [])
        
        # 提取每步的 phi 值
        step_phis = []
        highest_milestones = []
        
        for j in judgments:
            milestone_id = j.get("highest_milestone", "M0")
            phi = j.get("phi", self.milestone_to_phi.get(milestone_id, 0.0))
            step_phis.append(phi)
            highest_milestones.append(milestone_id)
        
        # 如果返回的步数不够，用最后一个值填充
        while len(step_phis) < num_steps:
            step_phis.append(step_phis[-1] if step_phis else 0.0)
            highest_milestones.append(highest_milestones[-1] if highest_milestones else "M0")
        
        return JudgmentResult(
            step_phis=step_phis[:num_steps],
            highest_milestones=highest_milestones[:num_steps],
            final_success=data.get("final_success", False),
            reasoning=data.get("reasoning", ""),
        )
    
    def judge_trajectory(
        self,
        task_description: str,
        trajectory: List[Dict[str, str]],
    ) -> JudgmentResult:
        """
        对单条轨迹进行里程碑判定
        
        Args:
            task_description: 任务描述
            trajectory: 轨迹步骤列表，每个元素包含 action 和 observation
        
        Returns:
            JudgmentResult 包含每步的势能值
        """
        prompt = self._build_prompt(task_description, trajectory)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content
                return self._parse_response(response_text, len(trajectory))
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 最后一次尝试失败，返回默认值
                    print(f"[MilestoneJudge] All retries failed: {e}")
                    return JudgmentResult(
                        step_phis=[0.0] * len(trajectory),
                        highest_milestones=["M0"] * len(trajectory),
                        final_success=False,
                        reasoning=f"Judge failed: {e}",
                    )
                print(f"[MilestoneJudge] Retry {attempt + 1}: {e}")
    
    def batch_judge(
        self,
        task_descriptions: List[str],
        trajectories: List[List[Dict[str, str]]],
    ) -> List[JudgmentResult]:
        """
        批量判定多条轨迹
        
        Args:
            task_descriptions: 任务描述列表
            trajectories: 轨迹列表
        
        Returns:
            判定结果列表
        """
        results = []
        for task_desc, traj in zip(task_descriptions, trajectories):
            result = self.judge_trajectory(task_desc, traj)
            results.append(result)
        return results


def create_milestone_judge_from_config(config) -> Optional[MilestoneJudge]:
    """
    从配置创建 MilestoneJudge 实例
    
    Args:
        config: Hydra 配置对象，需包含 algorithm.milestone_gae
    
    Returns:
        MilestoneJudge 实例，如果配置不完整则返回 None
    """
    try:
        milestone_cfg = config.algorithm.milestone_gae
        judge_cfg = milestone_cfg.judge_llm
        
        # 加载里程碑模板（支持 fallback_template 或旧的 milestone_template）
        from .templates import load_milestone_template
        template_name = milestone_cfg.get("fallback_template", 
                                          milestone_cfg.get("milestone_template", "alfworld"))
        template = load_milestone_template(template_name)
        
        # 获取默认里程碑（当启用动态生成时，这些会被覆盖）
        milestones = template.get("default_milestones", [])
        
        return MilestoneJudge(
            base_url=judge_cfg.base_url,
            model=judge_cfg.model,
            milestones=milestones,
            api_key=judge_cfg.get("api_key", "EMPTY"),
            temperature=judge_cfg.get("temperature", 0.1),
        )
    except Exception as e:
        print(f"[MilestoneJudge] Failed to create from config: {e}")
        return None
