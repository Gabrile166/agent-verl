"""
Milestone Generator - LLM-based dynamic milestone generation

Uses an LLM to automatically generate milestones from expert trajectories.
This replaces static JSON templates with dynamic, task-specific milestones.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class GeneratedMilestones:
    """生成的里程碑结果"""
    milestones: List[Dict[str, Any]]  # [{id, name, phi, criteria}, ...]
    reasoning: str
    success: bool


class MilestoneGenerator:
    """
    基于 LLM 的动态里程碑生成器
    
    根据专家轨迹自动识别关键阶段并生成里程碑定义。
    """
    
    PROMPT_TEMPLATE = """你是一个任务分解专家。

## 任务描述
{task_description}

## 专家成功轨迹
{expert_trajectory}

## 任务
请分析这条成功的专家轨迹，识别任务完成过程中的关键阶段/里程碑。

要求：
1. 提取 {num_milestones} 个关键里程碑，从初始状态到任务完成
2. 每个里程碑应该是任务进展的明确标志
3. 里程碑应该是可观测的（能从观察中判断是否达成）
4. phi 值应该均匀分布，从 0.0 递增到 1.0（最后一个必须是 1.0）
5. 判定标准要具体、可操作

输出格式 (严格 JSON):
{{
  "milestones": [
    {{"id": "M1", "name": "里程碑名称", "phi": 0.2, "criteria": "判定标准：具体的观察条件"}},
    {{"id": "M2", "name": "里程碑名称", "phi": 0.4, "criteria": "判定标准：具体的观察条件"}},
    ...
    {{"id": "M{num_milestones}", "name": "任务完成", "phi": 1.0, "criteria": "判定标准：任务成功完成"}}
  ],
  "reasoning": "简要说明里程碑划分依据"
}}"""

    def __init__(
        self,
        base_urls: List[str],
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.3,
        max_retries: int = 3,
        num_milestones: int = 5,
    ):
        """
        初始化 MilestoneGenerator
        
        Args:
            base_urls: LLM API 地址列表 (支持多 URL 负载均衡)
            model: 模型名称
            api_key: API 密钥
            temperature: 采样温度（生成时稍高一些）
            max_retries: 最大重试次数
            num_milestones: 期望的里程碑数量
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # 支持多 URL 负载均衡
        self.clients = []
        for url in base_urls:
            self.clients.append(OpenAI(base_url=url, api_key=api_key))
        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.num_milestones = num_milestones
        self._call_index = 0  # 用于轮询负载均衡
    
    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """格式化轨迹为可读字符串"""
        if not trajectory:
            return "[]"
        
        lines = []
        for i, step in enumerate(trajectory, 1):
            obs = step.get('observation', step.get('obs', ''))
            action = step.get('action', '')
            lines.append(f"Step {i}:\n  Observation: {obs}\n  Action: {action}")
        
        return "\n".join(lines)
    
    def _build_prompt(
        self,
        task_description: str,
        expert_trajectory: List[Dict],
    ) -> str:
        """构建生成 Prompt"""
        traj_str = self._format_trajectory(expert_trajectory)
        
        return self.PROMPT_TEMPLATE.format(
            task_description=task_description,
            expert_trajectory=traj_str,
            num_milestones=self.num_milestones,
        )
    
    def _parse_response(self, response_text: str) -> GeneratedMilestones:
        """解析 LLM 响应"""
        try:
            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError(f"No JSON found in response")
            
            data = json.loads(json_match.group())
            milestones = data.get("milestones", [])
            reasoning = data.get("reasoning", "")
            
            # 验证里程碑格式
            for m in milestones:
                if not all(k in m for k in ["id", "name", "phi", "criteria"]):
                    raise ValueError(f"Invalid milestone format: {m}")
            
            return GeneratedMilestones(
                milestones=milestones,
                reasoning=reasoning,
                success=True,
            )
        except Exception as e:
            return GeneratedMilestones(
                milestones=[],
                reasoning=f"Parse error: {e}",
                success=False,
            )
    
    def generate(
        self,
        task_description: str,
        expert_trajectory: List[Dict],
    ) -> GeneratedMilestones:
        """
        根据专家轨迹生成里程碑清单
        
        Args:
            task_description: 任务描述
            expert_trajectory: 专家轨迹
            
        Returns:
            GeneratedMilestones 包含里程碑列表
        """
        if not expert_trajectory:
            return GeneratedMilestones(
                milestones=self._get_default_milestones(),
                reasoning="No expert trajectory provided, using defaults",
                success=False,
            )
        
        prompt = self._build_prompt(task_description, expert_trajectory)
        
        # 轮询选择 client
        client_idx = self._call_index % len(self.clients)
        self._call_index += 1
        
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
                result = self._parse_response(response_text)
                
                if result.success and result.milestones:
                    return result
                    
            except Exception as e:
                print(f"[MilestoneGenerator] Retry {attempt + 1}: {e}")
        
        # 所有重试失败，返回默认里程碑
        return GeneratedMilestones(
            milestones=self._get_default_milestones(),
            reasoning="All retries failed, using defaults",
            success=False,
        )
    
    def batch_generate(
        self,
        task_descriptions: List[str],
        expert_trajectories: List[List[Dict]],
        max_workers: Optional[int] = None,
    ) -> List[GeneratedMilestones]:
        """
        并行生成多个里程碑清单
        
        Args:
            task_descriptions: 任务描述列表
            expert_trajectories: 专家轨迹列表
            max_workers: 最大并行数，默认为 len(clients) * 4
            
        Returns:
            GeneratedMilestones 列表
        """
        if not task_descriptions:
            return []
        
        n = len(task_descriptions)
        if max_workers is None:
            max_workers = len(self.clients) * 4
        
        results = [None] * n
        
        def _generate_one(idx: int) -> Tuple[int, GeneratedMilestones]:
            """生成单个里程碑（带索引返回）"""
            task_desc = task_descriptions[idx]
            expert_traj = expert_trajectories[idx] if idx < len(expert_trajectories) else []
            result = self.generate(task_desc, expert_traj)
            return idx, result
        
        # 使用 ThreadPoolExecutor 并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_one, i) for i in range(n)]
            
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"[MilestoneGenerator] batch_generate error: {e}")
        
        # 填充失败的结果
        for i in range(n):
            if results[i] is None:
                results[i] = GeneratedMilestones(
                    milestones=self._get_default_milestones(),
                    reasoning="Parallel generation failed",
                    success=False,
                )
        
        return results

    def _get_default_milestones(self) -> List[Dict[str, Any]]:
        """获取默认里程碑（fallback）"""
        n = self.num_milestones
        return [
            {
                "id": f"M{i+1}",
                "name": f"阶段 {i+1}",
                "phi": round((i + 1) / n, 2),
                "criteria": f"完成约 {int((i+1)/n*100)}% 的任务进度",
            }
            for i in range(n)
        ]


def create_milestone_generator_from_config(config) -> Optional[MilestoneGenerator]:
    """
    从配置创建 MilestoneGenerator 实例
    
    Args:
        config: Hydra 配置对象，需包含 algorithm.milestone_gae.generator
    
    Returns:
        MilestoneGenerator 实例，如果配置不完整则返回 None
    """
    try:
        import json as json_module
        
        milestone_cfg = config.algorithm.milestone_gae
        gen_cfg = milestone_cfg.generator
        
        if not gen_cfg.enable:
            return None
        
        # 处理 base_urls - 支持列表、JSON 字符串、逗号分隔字符串
        base_urls = gen_cfg.llm.get("base_urls", gen_cfg.llm.get("base_url", "http://127.0.0.1:8080/v1"))
        
        if isinstance(base_urls, str):
            # 去除可能的外层引号
            base_urls = base_urls.strip("'\"")
            
            if base_urls.startswith('[') and base_urls.endswith(']'):
                # JSON 数组格式
                try:
                    base_urls = json_module.loads(base_urls)
                except json_module.JSONDecodeError:
                    base_urls = base_urls[1:-1].split(',')
                    base_urls = [url.strip().strip('"').strip("'") for url in base_urls]
            elif ',' in base_urls:
                # 逗号分隔格式
                base_urls = [url.strip().strip('"').strip("'") for url in base_urls.split(',')]
            else:
                # 单个 URL
                base_urls = [base_urls]
        else:
            # 已经是列表
            base_urls = list(base_urls)
        
        print(f"[MilestoneGenerator] Creating with {len(base_urls)} URLs: {base_urls}")
        
        return MilestoneGenerator(
            base_urls=base_urls,
            model=gen_cfg.llm.model,
            api_key=gen_cfg.llm.get("api_key", "EMPTY"),
            temperature=gen_cfg.llm.get("temperature", 0.3),
            num_milestones=gen_cfg.get("num_milestones", 5),
        )
    except Exception as e:
        print(f"[MilestoneGenerator] Failed to create from config: {e}")
        import traceback
        traceback.print_exc()
        return None
