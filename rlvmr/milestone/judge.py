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
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
    支持多 URL 负载均衡。
    """
    
    def __init__(
        self,
        base_urls: List[str],
        model: str,
        milestones: List[Dict[str, Any]],
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """
        初始化 MilestoneJudge
        
        Args:
            base_urls: LLM API 地址列表 (支持多 URL 负载均衡)
            model: 模型名称
            milestones: 里程碑列表，每个元素包含 id, name, phi, criteria
            api_key: API 密钥
            temperature: 采样温度
            max_retries: 最大重试次数
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # 支持多 URL 负载均衡
        self.clients = []
        for url in base_urls:
            self.clients.append(OpenAI(base_url=url, api_key=api_key))
        
        self.model = model
        self.milestones = milestones
        self.temperature = temperature
        self.max_retries = max_retries
        self._call_index = 0  # 用于轮询负载均衡
        
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
        milestones: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建 Judge 的 Prompt
        
        Args:
            task_description: 任务描述
            trajectory: 轨迹
            milestones: 里程碑列表（可选，默认使用 self.milestones）
        """
        # 使用传入的里程碑或默认的实例里程碑
        ms = milestones if milestones is not None else self.milestones
        
        # 里程碑清单
        milestone_list = "\n".join([
            f'{m["id"]} (Φ={m["phi"]}): {m["name"]} — Criteria: {m["criteria"]}'
            for m in ms
        ])
        
        # 轨迹步骤
        steps_str = ""
        for i, step in enumerate(trajectory, 1):
            action = step.get("action", "N/A")
            observation = step.get("observation", "N/A")
            steps_str += f"\nStep {i}:\n  Environment State: {observation}\n  Agent Action: {action}\n"
        
        prompt = f"""You are a task progress evaluator.

## Task Description
{task_description}

## Milestone Checklist
M0 (Φ=0.0): Not started — Criteria: No milestone has been achieved
{milestone_list}

## Agent Execution Trajectory

Note: Each step shows the environment state (what the agent observes before acting) followed by the agent's action.
{steps_str}

## Instructions

Evaluate the highest milestone achieved at each step.

Output format (strict JSON):
{{
  "judgments": [
    {{"step": 1, "highest_milestone": "M0", "phi": 0.0}},
    {{"step": 2, "highest_milestone": "M1", "phi": 0.15}},
    ...
  ],
  "final_success": true/false,
  "reasoning": "Brief explanation of your judgment"
}}

Notes:
1. M0 means no milestone has been achieved yet, phi=0.0
2. Milestones are generally monotonically increasing (may occasionally regress due to wrong actions)
3. The highest milestone (phi=1.0) should only be reached when the task is confirmed successful
4. You must output valid JSON"""

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
    
    def _parse_response_with_phi_map(
        self, response_text: str, num_steps: int, phi_map: Dict[str, float]
    ) -> JudgmentResult:
        """解析 LLM 响应（使用传入的 phi 映射，线程安全）"""
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")
        
        data = json.loads(json_match.group())
        judgments = data.get("judgments", [])
        
        step_phis = []
        highest_milestones = []
        
        for j in judgments:
            milestone_id = j.get("highest_milestone", "M0")
            phi = j.get("phi", phi_map.get(milestone_id, 0.0))
            step_phis.append(phi)
            highest_milestones.append(milestone_id)
        
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
        
        # 轮询选择 client
        client_idx = self._call_index % len(self.clients)
        self._call_index += 1
        
        last_error = None
        for attempt in range(self.max_retries):
            # 在不同 URL 之间轮换尝试
            current_client_idx = (client_idx + attempt) % len(self.clients)
            client = self.clients[current_client_idx]
            
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content
                return self._parse_response(response_text, len(trajectory))
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"[MilestoneJudge] Retry {attempt + 1}: {e}")
        
        # 所有尝试失败，返回默认值
        print(f"[MilestoneJudge] All retries failed: {last_error}")
        return JudgmentResult(
            step_phis=[0.0] * len(trajectory),
            highest_milestones=["M0"] * len(trajectory),
            final_success=False,
            reasoning=f"Judge failed: {last_error}",
        )
    
    def judge_trajectory_with_milestones(
        self,
        task_description: str,
        trajectory: List[Dict[str, str]],
        milestones: List[Dict[str, Any]],
    ) -> JudgmentResult:
        """
        线程安全版本：对单条轨迹进行里程碑判定（使用传入的里程碑）
        
        Args:
            task_description: 任务描述
            trajectory: 轨迹步骤列表
            milestones: 里程碑列表
        
        Returns:
            JudgmentResult 包含每步的势能值
        """
        prompt = self._build_prompt(task_description, trajectory, milestones)
        
        # 构建局部 phi 映射
        local_phi_map = {"M0": 0.0}
        for m in milestones:
            local_phi_map[m["id"]] = m["phi"]
        
        # 轮询选择 client
        client_idx = self._call_index % len(self.clients)
        self._call_index += 1
        
        last_error = None
        for attempt in range(self.max_retries):
            current_client_idx = (client_idx + attempt) % len(self.clients)
            client = self.clients[current_client_idx]
            
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content
                return self._parse_response_with_phi_map(response_text, len(trajectory), local_phi_map)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"[MilestoneJudge] Retry {attempt + 1}: {e}")
        
        print(f"[MilestoneJudge] All retries failed: {last_error}")
        return JudgmentResult(
            step_phis=[0.0] * len(trajectory),
            highest_milestones=["M0"] * len(trajectory),
            final_success=False,
            reasoning=f"Judge failed: {last_error}",
        )
    
    def batch_judge(
        self,
        task_descriptions: List[str],
        trajectories: List[List[Dict[str, str]]],
        max_workers: Optional[int] = None,
    ) -> List[JudgmentResult]:
        """
        并行批量判定多条轨迹
        
        Args:
            task_descriptions: 任务描述列表
            trajectories: 轨迹列表
            max_workers: 最大并行数，默认为 len(clients) * 4
        
        Returns:
            判定结果列表
        """
        if not task_descriptions:
            return []
        
        n = len(task_descriptions)
        if max_workers is None:
            max_workers = len(self.clients) * 4
        
        results = [None] * n
        
        def _judge_one(idx: int) -> Tuple[int, JudgmentResult]:
            """判定单条轨迹（带索引返回）"""
            task_desc = task_descriptions[idx]
            traj = trajectories[idx] if idx < len(trajectories) else []
            result = self.judge_trajectory(task_desc, traj)
            return idx, result
        
        # 使用 ThreadPoolExecutor 并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_judge_one, i) for i in range(n)]
            
            # 进度条
            if tqdm is not None:
                pbar = tqdm(total=n, desc="[Judge] Trajectories", unit="traj")
            
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"[MilestoneJudge] batch_judge error: {e}")
                finally:
                    if tqdm is not None:
                        pbar.update(1)
            
            if tqdm is not None:
                pbar.close()
        
        # 填充失败的结果
        for i in range(n):
            if results[i] is None:
                traj_len = len(trajectories[i]) if i < len(trajectories) else 1
                results[i] = JudgmentResult(
                    step_phis=[0.0] * traj_len,
                    highest_milestones=["M0"] * traj_len,
                    final_success=False,
                    reasoning="Parallel judging failed",
                )
        
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
        import json as json_module
        
        milestone_cfg = config.algorithm.milestone_gae
        judge_cfg = milestone_cfg.judge_llm
        
        # 加载里程碑模板（支持 fallback_template 或旧的 milestone_template）
        from .templates import load_milestone_template
        template_name = milestone_cfg.get("fallback_template", 
                                          milestone_cfg.get("milestone_template", "alfworld"))
        template = load_milestone_template(template_name)
        
        # 获取默认里程碑（当启用动态生成时，这些会被覆盖）
        milestones = template.get("default_milestones", [])
        
        # 处理 base_urls - 支持列表、JSON 字符串、逗号分隔字符串
        base_urls = judge_cfg.get("base_urls", judge_cfg.get("base_url", "http://127.0.0.1:8080/v1"))
        
        if isinstance(base_urls, str):
            # 去除可能的外层引号
            base_urls = base_urls.strip("'\"")
            
            if base_urls.startswith('[') and base_urls.endswith(']'):
                # JSON 数组格式: ["url1", "url2"]
                try:
                    base_urls = json_module.loads(base_urls)
                    print(f"  [MilestoneJudge] base_urls parsed from JSON: {base_urls}")
                except json_module.JSONDecodeError:
                    # 非标准格式，尝试手动解析
                    base_urls = base_urls[1:-1].split(',')
                    base_urls = [url.strip().strip('"').strip("'") for url in base_urls]
                    print(f"  [MilestoneJudge] base_urls parsed from bracket string: {base_urls}")
            elif ',' in base_urls:
                # 逗号分隔格式: url1,url2
                base_urls = [url.strip().strip('"').strip("'") for url in base_urls.split(',')]
                print(f"  [MilestoneJudge] base_urls parsed from comma-separated: {base_urls}")
            else:
                # 单个 URL
                base_urls = [base_urls]
                print(f"  [MilestoneJudge] base_urls single URL: {base_urls}")
        else:
            # 已经是列表
            base_urls = list(base_urls)
        
        print(f"[MilestoneJudge] Creating with {len(base_urls)} URLs: {base_urls}")
        
        return MilestoneJudge(
            base_urls=base_urls,
            model=judge_cfg.model,
            milestones=milestones,
            api_key=judge_cfg.get("api_key", "EMPTY"),
            temperature=judge_cfg.get("temperature", 0.1),
        )
    except Exception as e:
        print(f"[MilestoneJudge] Failed to create from config: {e}")
        import traceback
        traceback.print_exc()
        return None
