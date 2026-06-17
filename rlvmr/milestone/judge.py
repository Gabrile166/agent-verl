"""
Milestone Judge - LLM-based trajectory milestone evaluation

Uses an LLM to evaluate which milestones have been achieved at each step
of a trajectory. This replaces traditional Critic networks with
LLM-as-Critic approach.
"""

import json
import re
import threading
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
    chunk_stats: Optional[Dict[str, float]] = None


@dataclass
class SegmentJudgeState:
    """Chunked Judge 的跨段状态"""
    previous_highest_milestone: str = "M0"
    previous_phi: float = 0.0
    previous_summary: str = ""


@dataclass
class SegmentJudgmentResult:
    """单个 chunk 的判定结果"""
    step_phis: List[float]
    highest_milestones: List[str]
    ending_highest_milestone: str
    ending_phi: float
    segment_summary: str
    final_success: bool = False
    failed: bool = False
    reasoning: str = ""


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
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False,
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
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._call_index = 0  # 用于轮询负载均衡
        self._call_index_lock = threading.Lock()
        
        # 构建里程碑 ID 到 phi 的映射
        self.milestone_to_phi = {"M0": 0.0}
        for m in milestones:
            self.milestone_to_phi[m["id"]] = m["phi"]

    def _next_client_index(self) -> int:
        """Thread-safe round-robin client selection."""
        with self._call_index_lock:
            client_idx = self._call_index % len(self.clients)
            self._call_index += 1
        return client_idx

    def _chat_completion(self, client, prompt: str):
        request_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.disable_thinking:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }
        return client.chat.completions.create(**request_kwargs)
    
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

    @staticmethod
    def _build_phi_map(milestones: List[Dict[str, Any]]) -> Dict[str, float]:
        phi_map = {"M0": 0.0}
        for m in milestones:
            if "id" not in m:
                continue
            try:
                phi_map[str(m["id"])] = float(m.get("phi", 0.0))
            except (TypeError, ValueError):
                phi_map[str(m["id"])] = 0.0
        return phi_map

    @staticmethod
    def _normalize_chunk_config(chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
        try:
            chunk_size = int(chunk_size)
        except (TypeError, ValueError):
            chunk_size = 0
        try:
            chunk_overlap = int(chunk_overlap)
        except (TypeError, ValueError):
            chunk_overlap = 0
        chunk_overlap = max(0, chunk_overlap)
        if chunk_size > 0:
            chunk_overlap = min(chunk_overlap, max(chunk_size - 1, 0))
        return chunk_size, chunk_overlap

    @staticmethod
    def _iter_judge_chunks(total_steps: int, chunk_size: int, chunk_overlap: int):
        """Yield balanced Judge chunks with context-only overlap."""
        if total_steps <= 0:
            return
        chunk_size, chunk_overlap = MilestoneJudge._normalize_chunk_config(chunk_size, chunk_overlap)
        if chunk_size <= 0 or total_steps <= chunk_size:
            yield {
                "judge_start": 0,
                "judge_end": total_steps,
                "context_start": 0,
                "context_end": 0,
            }
            return

        num_chunks = (total_steps + chunk_size - 1) // chunk_size
        base_size = total_steps // num_chunks
        remainder = total_steps % num_chunks

        judge_start = 0
        for chunk_idx in range(num_chunks):
            chunk_len = base_size + (1 if chunk_idx < remainder else 0)
            judge_end = judge_start + chunk_len
            yield {
                "judge_start": judge_start,
                "judge_end": judge_end,
                "context_start": max(0, judge_start - chunk_overlap),
                "context_end": judge_start,
            }
            judge_start = judge_end

    @staticmethod
    def _format_score(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "N/A"

    @classmethod
    def _format_steps(
        cls,
        trajectory: List[Dict[str, Any]],
        start_step_number: int = 1,
        include_task_score: bool = False,
    ) -> str:
        steps_str = ""
        for offset, step in enumerate(trajectory):
            step_number = start_step_number + offset
            action = step.get("action", "N/A")
            observation = step.get("observation", "N/A")
            steps_str += f"\nStep {step_number}:\n  Environment State: {observation}\n  Agent Action: {action}\n"
            if include_task_score:
                score_before = cls._format_score(step.get("task_score_before"))
                score_after = cls._format_score(step.get("task_score_after"))
                score_delta = cls._format_score(step.get("task_score_delta"))
                steps_str += (
                    f"  Task Score Before: {score_before}\n"
                    f"  Task Score After: {score_after}\n"
                    f"  Task Score Delta: {score_delta}\n"
                )
        return steps_str

    @staticmethod
    def _milestone_list(milestones: List[Dict[str, Any]]) -> str:
        return "\n".join([
            f'{m["id"]} (Φ={m["phi"]}): {m["name"]} — Criteria: {m["criteria"]}'
            for m in milestones
        ])

    @staticmethod
    def _task_score_prompt_parts(include_task_score: bool) -> Tuple[str, str]:
        if not include_task_score:
            return "", ""
        context = """
When present, Task Score fields are SciWorld's cumulative environment progress scores:
- Task Score Before: score before the action at this step
- Task Score After: score after the action at this step
- Task Score Delta: after minus before
"""
        notes = """
5. Use Task Score Delta as strong evidence of objective progress when it is positive
6. Task Score Delta = 0 does not necessarily mean the step is useless; navigation, preparation, reading, or inspection may be necessary
7. If Task Score Delta = 0 and the observation shows no clear task-relevant state change, keep the previous milestone
8. If Task Score Delta < 0, treat the step as regression or a likely mistake unless the observation clearly proves otherwise"""
        return context, notes

    def _build_prompt(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        milestones: Optional[List[Dict[str, Any]]] = None,
        include_task_score: bool = False,
    ) -> str:
        """构建 Judge 的 Prompt

        Args:
            task_description: 任务描述
            trajectory: 轨迹
            milestones: 里程碑列表（可选，默认使用 self.milestones）
            include_task_score: 是否展示 SciWorld 环境累计分变化
        """
        # 使用传入的里程碑或默认的实例里程碑
        ms = milestones if milestones is not None else self.milestones

        milestone_list = self._milestone_list(ms)
        steps_str = self._format_steps(trajectory, start_step_number=1, include_task_score=include_task_score)
        task_score_context, task_score_notes = self._task_score_prompt_parts(include_task_score)
        
        prompt = f"""You are a task progress evaluator.

## Task Description
{task_description}

## Milestone Checklist
M0 (Φ=0.0): Not started — Criteria: No milestone has been achieved
{milestone_list}

## Agent Execution Trajectory

Note: Each step shows the environment state (what the agent observes before acting) followed by the agent's action.
{task_score_context}
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
4. You must output valid JSON{task_score_notes}"""

        return prompt

    def _build_chunk_prompt(
        self,
        task_description: str,
        context_steps: List[Dict[str, Any]],
        steps_to_judge: List[Dict[str, Any]],
        milestones: List[Dict[str, Any]],
        previous_state: SegmentJudgeState,
        judge_start: int,
        context_start: int,
        include_task_score: bool = False,
    ) -> str:
        """Build a chunked Judge prompt with previous-state carryover."""
        milestone_list = self._milestone_list(milestones)
        task_score_context, task_score_notes = self._task_score_prompt_parts(include_task_score)

        context_block = ""
        if context_steps:
            context_block = f"""
## Context-Only Previous Steps
These steps are provided only to recover local context. Do not output judgments for them.
{self._format_steps(context_steps, start_step_number=context_start + 1, include_task_score=include_task_score)}
"""

        steps_block = self._format_steps(
            steps_to_judge,
            start_step_number=judge_start + 1,
            include_task_score=include_task_score,
        )

        previous_summary = previous_state.previous_summary or "No previous segment summary."
        expected_count = len(steps_to_judge)
        first_step = judge_start + 1
        last_step = judge_start + expected_count

        return f"""You are a task progress evaluator judging one segment of a longer trajectory.

## Task Description
{task_description}

## Milestone Checklist
M0 (Φ=0.0): Not started — Criteria: No milestone has been achieved
{milestone_list}

## Previous Segment State
Previous highest milestone: {previous_state.previous_highest_milestone}
Previous phi: {previous_state.previous_phi:.4f}
Previous summary: {previous_summary}

{task_score_context}
{context_block}
## Steps To Judge
Output exactly one judgment for each of these {expected_count} steps: Step {first_step} through Step {last_step}.
{steps_block}

## Instructions
Start from the previous highest milestone and previous phi. Do not restart from M0.
Judge only the steps in "Steps To Judge"; context-only steps are reference material.
Only downgrade if this segment clearly shows the agent lost prior progress, such as destroying, dropping, disconnecting, or invalidating a previously achieved required state.
Use milestone IDs from the checklist. The phi value will be derived from the milestone ID, so choose the milestone ID carefully.

Output format (strict JSON):
{{
  "judgments": [
    {{"step": {first_step}, "highest_milestone": "{previous_state.previous_highest_milestone}"}},
    ...
  ],
  "ending_highest_milestone": "<milestone id after the final judged step>",
  "final_success": true/false,
  "segment_summary": "Brief summary of task-relevant progress in this segment"
}}

Notes:
1. Output exactly {expected_count} judgments, no more and no fewer
2. Do not output judgments for context-only steps
3. Use global step numbers as shown above, not segment-local step numbers
4. The highest milestone (phi=1.0) should only be reached when the task is confirmed successful
5. You must output valid JSON{task_score_notes}"""
    
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
            milestone_id = self._resolve_milestone_id(j.get("highest_milestone", "M0"), phi_map, "M0")
            phi = phi_map.get(milestone_id, 0.0)
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

    @staticmethod
    def _resolve_milestone_id(milestone_id: Any, phi_map: Dict[str, float], fallback_id: str) -> str:
        milestone_id = str(milestone_id).strip() if milestone_id is not None else fallback_id
        if milestone_id in phi_map:
            return milestone_id
        return fallback_id if fallback_id in phi_map else "M0"

    @staticmethod
    def _fallback_segment_result(
        expected_steps: int,
        previous_state: SegmentJudgeState,
        reason: str,
    ) -> SegmentJudgmentResult:
        milestone_id = previous_state.previous_highest_milestone or "M0"
        phi = float(previous_state.previous_phi)
        summary = f"[Previous segment failed, context may be incomplete] {reason}"
        return SegmentJudgmentResult(
            step_phis=[phi] * expected_steps,
            highest_milestones=[milestone_id] * expected_steps,
            ending_highest_milestone=milestone_id,
            ending_phi=phi,
            segment_summary=summary,
            final_success=False,
            failed=True,
            reasoning=summary,
        )

    def _parse_chunk_response_with_phi_map(
        self,
        response_text: str,
        expected_steps: int,
        phi_map: Dict[str, float],
        previous_state: SegmentJudgeState,
    ) -> SegmentJudgmentResult:
        """Parse a chunk response. Phi values are always derived from phi_map."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError(f"No JSON found in response: {response_text[:200]}")

            data = json.loads(json_match.group())
            judgments = data.get("judgments", [])

            step_phis = []
            highest_milestones = []
            fallback_mid = previous_state.previous_highest_milestone or "M0"

            for j in judgments:
                milestone_id = self._resolve_milestone_id(j.get("highest_milestone", fallback_mid), phi_map, fallback_mid)
                step_phis.append(phi_map.get(milestone_id, previous_state.previous_phi))
                highest_milestones.append(milestone_id)
                fallback_mid = milestone_id

            while len(step_phis) < expected_steps:
                step_phis.append(step_phis[-1] if step_phis else previous_state.previous_phi)
                highest_milestones.append(highest_milestones[-1] if highest_milestones else previous_state.previous_highest_milestone)

            step_phis = step_phis[:expected_steps]
            highest_milestones = highest_milestones[:expected_steps]

            last_mid = highest_milestones[-1] if highest_milestones else previous_state.previous_highest_milestone
            ending_mid = self._resolve_milestone_id(last_mid, phi_map, previous_state.previous_highest_milestone)
            ending_phi = phi_map.get(ending_mid, step_phis[-1] if step_phis else previous_state.previous_phi)
            summary = data.get("segment_summary") or data.get("reasoning") or f"Segment ended at {ending_mid} (phi={ending_phi:.4f})"

            return SegmentJudgmentResult(
                step_phis=step_phis,
                highest_milestones=highest_milestones,
                ending_highest_milestone=ending_mid,
                ending_phi=ending_phi,
                segment_summary=summary,
                final_success=bool(data.get("final_success", False)),
                failed=False,
                reasoning=data.get("reasoning", summary),
            )
        except Exception as e:
            return self._fallback_segment_result(expected_steps, previous_state, f"Parse failed: {e}")
    
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
        client_idx = self._next_client_index()
        
        last_error = None
        for attempt in range(self.max_retries):
            # 在不同 URL 之间轮换尝试
            current_client_idx = (client_idx + attempt) % len(self.clients)
            client = self.clients[current_client_idx]
            
            try:
                response = self._chat_completion(client, prompt)
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
    
    def _judge_trajectory_single_with_milestones(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        milestones: List[Dict[str, Any]],
        include_task_score: bool = False,
    ) -> JudgmentResult:
        """
        线程安全版本：对单条轨迹进行里程碑判定（使用传入的里程碑）
        
        Args:
            task_description: 任务描述
            trajectory: 轨迹步骤列表
            milestones: 里程碑列表
            include_task_score: 是否在 prompt 中展示 SciWorld 环境累计分变化
        
        Returns:
            JudgmentResult 包含每步的势能值
        """
        prompt = self._build_prompt(
            task_description,
            trajectory,
            milestones,
            include_task_score=include_task_score,
        )
        
        local_phi_map = self._build_phi_map(milestones)
        
        # 轮询选择 client
        client_idx = self._next_client_index()
        
        last_error = None
        for attempt in range(self.max_retries):
            current_client_idx = (client_idx + attempt) % len(self.clients)
            client = self.clients[current_client_idx]
            
            try:
                response = self._chat_completion(client, prompt)
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

    def judge_trajectory_with_milestones(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        milestones: List[Dict[str, Any]],
        include_task_score: bool = False,
        chunk_size: int = 0,
        chunk_overlap: int = 1,
    ) -> JudgmentResult:
        """
        对单条轨迹进行里程碑判定。默认整条轨迹一次判定；chunk_size>0 时按段顺序判定。
        """
        chunk_size, chunk_overlap = self._normalize_chunk_config(chunk_size, chunk_overlap)
        if chunk_size <= 0 or len(trajectory) <= chunk_size:
            result = self._judge_trajectory_single_with_milestones(
                task_description=task_description,
                trajectory=trajectory,
                milestones=milestones,
                include_task_score=include_task_score,
            )
            result.chunk_stats = {
                "chunk_enabled": 0.0,
                "chunk_size": float(chunk_size),
                "chunk_overlap": float(chunk_overlap),
                "chunk_count": 1.0 if trajectory else 0.0,
                "chunk_failures": 0.0,
            }
            return result

        return self._judge_trajectory_chunked_with_milestones(
            task_description=task_description,
            trajectory=trajectory,
            milestones=milestones,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_task_score=include_task_score,
        )

    def _judge_trajectory_chunked_with_milestones(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        milestones: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int = 1,
        include_task_score: bool = False,
    ) -> JudgmentResult:
        """Judge one trajectory in sequential chunks while preserving trajectory-level continuity."""
        total_steps = len(trajectory)
        phi_map = self._build_phi_map(milestones)
        chunks = list(self._iter_judge_chunks(total_steps, chunk_size, chunk_overlap))

        state = SegmentJudgeState()
        all_phis: List[float] = []
        all_milestones: List[str] = []
        summaries: List[str] = []
        failure_count = 0
        final_success = False

        for chunk in chunks:
            judge_start = chunk["judge_start"]
            judge_end = chunk["judge_end"]
            context_start = chunk["context_start"]
            context_end = chunk["context_end"]
            context_steps = trajectory[context_start:context_end]
            steps_to_judge = trajectory[judge_start:judge_end]

            prompt = self._build_chunk_prompt(
                task_description=task_description,
                context_steps=context_steps,
                steps_to_judge=steps_to_judge,
                milestones=milestones,
                previous_state=state,
                judge_start=judge_start,
                context_start=context_start,
                include_task_score=include_task_score,
            )

            segment_result = None
            last_error = None
            client_idx = self._next_client_index()
            for attempt in range(self.max_retries):
                current_client_idx = (client_idx + attempt) % len(self.clients)
                client = self.clients[current_client_idx]
                try:
                    response = self._chat_completion(client, prompt)
                    response_text = response.choices[0].message.content
                    parsed = self._parse_chunk_response_with_phi_map(
                        response_text=response_text,
                        expected_steps=len(steps_to_judge),
                        phi_map=phi_map,
                        previous_state=state,
                    )
                    if not parsed.failed:
                        segment_result = parsed
                        break
                    last_error = parsed.reasoning
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        print(f"[MilestoneJudge:chunk] Retry {attempt + 1}: {e}")

            if segment_result is None:
                segment_result = self._fallback_segment_result(
                    expected_steps=len(steps_to_judge),
                    previous_state=state,
                    reason=f"Judge failed: {last_error}",
                )

            all_phis.extend(segment_result.step_phis)
            all_milestones.extend(segment_result.highest_milestones)
            summaries.append(segment_result.segment_summary)
            final_success = segment_result.final_success

            if segment_result.failed:
                failure_count += 1
                state.previous_summary = segment_result.segment_summary
            else:
                state = SegmentJudgeState(
                    previous_highest_milestone=segment_result.ending_highest_milestone,
                    previous_phi=segment_result.ending_phi,
                    previous_summary=segment_result.segment_summary,
                )

        if len(all_phis) < total_steps:
            fill_phi = all_phis[-1] if all_phis else 0.0
            fill_mid = all_milestones[-1] if all_milestones else "M0"
            missing = total_steps - len(all_phis)
            all_phis.extend([fill_phi] * missing)
            all_milestones.extend([fill_mid] * missing)
        all_phis = all_phis[:total_steps]
        all_milestones = all_milestones[:total_steps]

        reasoning = " | ".join(summaries)
        if len(reasoning) > 1000:
            reasoning = reasoning[:997] + "..."

        return JudgmentResult(
            step_phis=all_phis,
            highest_milestones=all_milestones,
            final_success=final_success if failure_count < len(chunks) else False,
            reasoning=reasoning,
            chunk_stats={
                "chunk_enabled": 1.0,
                "chunk_size": float(chunk_size),
                "chunk_overlap": float(chunk_overlap),
                "chunk_count": float(len(chunks)),
                "chunk_failures": float(failure_count),
            },
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
            # max_workers = len(self.clients) * 4
            max_workers = 128
        
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
            max_tokens=judge_cfg.get("max_tokens", None),
            disable_thinking=judge_cfg.get("disable_thinking", False),
        )
    except Exception as e:
        print(f"[MilestoneJudge] Failed to create from config: {e}")
        import traceback
        traceback.print_exc()
        return None
