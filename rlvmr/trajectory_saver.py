"""
Trajectory Saver Module

将 rollout 轨迹数据保存为 JSONL 格式，用于后续分析和调试。

Usage:
    from rlvmr.trajectory_saver import TrajectorySaver
    
    saver = TrajectorySaver(output_dir="outputs/trajectory_data", enabled=True)
    saver.save_batch(batch_id=0, trajectory_list=..., ...)
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import numpy as np
import torch


class TrajectorySaver:
    """
    JSONL 格式的轨迹数据保存器。
    
    每个 JSONL 文件保存一个 batch 的数据，每行是一个样本（包含多个 episodes）。
    
    Attributes:
        output_dir: JSONL 文件保存目录
        enabled: 是否启用保存功能
    """
    
    def __init__(self, output_dir: str, enabled: bool = True):
        """
        初始化轨迹保存器。
        
        Args:
            output_dir: 保存目录路径
            enabled: 是否启用保存功能，默认 True
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._batch_count = 0
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_batch(
        self,
        batch_id: int,
        trajectory_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        expert_trajectories: List[List[Dict]],
        tasks: List[str],
        traj_uids: np.ndarray,
        gamefiles: List[str] = None,
        disc_episode_rewards: Optional[torch.Tensor] = None,
        disc_step_rewards: Optional[List[List[float]]] = None,
        disc_analyses: Optional[List[str]] = None,
        rollout_n: int = 8,
        global_step: int = 0,
    ) -> Optional[str]:
        """
        保存一个 batch 的数据到 JSONL 文件。
        
        Args:
            batch_id: Batch 编号
            trajectory_list: 轨迹列表，List[List[Dict]]，外层是 episode，内层是 steps
            episode_rewards: 每个 episode 的奖励
            expert_trajectories: Expert 轨迹列表
            tasks: 任务描述列表
            traj_uids: 轨迹唯一标识符
            disc_episode_rewards: Discriminator episode 奖励 (可选)
            disc_step_rewards: Discriminator step 奖励列表 (可选)
            disc_analyses: Discriminator 分析文本列表 (可选)
            gamefiles: 游戏文件路径列表，用于标识具体游戏环境 (可选)
            rollout_n: 每个样本的 rollout 数量
            global_step: 全局训练步数
        
        Returns:
            保存的文件路径，如果未启用则返回 None
        """
        if not self.enabled:
            return None
        
        # 文件命名
        filename = f"batch_{batch_id:06d}_step_{global_step}.jsonl"
        filepath = self.output_dir / filename
        
        # 按 prompt_index 分组样本
        num_episodes = len(trajectory_list)
        if rollout_n <= 0:
            rollout_n = 1
        
        num_samples = num_episodes // rollout_n
        
        samples = []
        for sample_idx in range(num_samples):
            start_idx = sample_idx * rollout_n
            end_idx = start_idx + rollout_n
            
            episodes = []
            for ep_idx in range(start_idx, end_idx):
                if ep_idx >= num_episodes:
                    break
                
                # 获取 discriminator scores
                disc_ep_score = None
                disc_step_scores_for_ep = None
                
                if disc_episode_rewards is not None:
                    disc_ep_score = float(disc_episode_rewards[ep_idx].item() 
                                         if torch.is_tensor(disc_episode_rewards[ep_idx]) 
                                         else disc_episode_rewards[ep_idx])
                
                if disc_step_rewards is not None and ep_idx < len(disc_step_rewards):
                    disc_step_scores_for_ep = disc_step_rewards[ep_idx]
                
                # 获取 discriminator analysis
                disc_analysis = None
                if disc_analyses is not None and ep_idx < len(disc_analyses):
                    disc_analysis = disc_analyses[ep_idx]
                
                # 格式化 episode
                episode = self._format_episode(
                    episode_index=ep_idx - start_idx,
                    trajectory=trajectory_list[ep_idx],
                    task=tasks[ep_idx] if ep_idx < len(tasks) else "",
                    traj_uid=str(traj_uids[ep_idx]) if ep_idx < len(traj_uids) else "",
                    episode_reward=float(episode_rewards[ep_idx]),
                    disc_episode_score=disc_ep_score,
                    disc_step_scores=disc_step_scores_for_ep,
                    disc_analysis=disc_analysis,
                )
                episodes.append(episode)
            
            # 获取该 sample 对应的 expert 轨迹
            expert_traj = []
            if sample_idx < len(expert_trajectories) and expert_trajectories[sample_idx]:
                expert_traj = self._format_expert_trajectory(expert_trajectories[sample_idx])
            
            # 获取该 sample 对应的 gamefile（同一 sample 的所有 episodes 共享同一个 gamefile）
            gamefile = ""
            if gamefiles and start_idx < len(gamefiles):
                gamefile = gamefiles[start_idx] if gamefiles[start_idx] else ""
            
            sample = {
                "sample_index": sample_idx,
                "prompt_index": sample_idx,
                "gamefile": gamefile,
                "expert_trajectory": expert_traj,
                "episodes": episodes
            }
            samples.append(sample)
        
        # 写入 JSONL 文件
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False, indent=2) + '\n')
        
        self._batch_count += 1
        print(f"[TrajectorySaver] Saved batch {batch_id} to {filepath}")
        
        return str(filepath)
    
    def _format_episode(
        self,
        episode_index: int,
        trajectory: List[Dict],
        task: str,
        traj_uid: str,
        episode_reward: float,
        disc_episode_score: Optional[float] = None,
        disc_step_scores: Optional[List[float]] = None,
        disc_analysis: Optional[str] = None,
    ) -> Dict:
        """
        格式化单个 episode 数据。
        
        Args:
            episode_index: Episode 在样本内的索引
            trajectory: 轨迹数据（step 列表）
            task: 任务描述
            traj_uid: 轨迹唯一标识符
            episode_reward: Episode 总奖励
            disc_episode_score: Discriminator episode 评分
            disc_step_scores: Discriminator step 评分列表
            disc_analysis: Discriminator 分析文本
        
        Returns:
            格式化的 episode 字典
        """
        traj = []
        active_step_idx = 0
        
        for step_data in trajectory:
            # 只处理有效的 step
            if not step_data.get('active_masks', False):
                continue
            
            full_output = step_data.get('full_output', '')
            
            step = {
                "step_index": active_step_idx,
                "obs": self._get_obs(step_data),
                "reason": self._extract_reason(full_output),
                "action": self._extract_action(full_output),
            }
            
            # 合并 discriminator scores 到 step 中
            if disc_step_scores is not None and active_step_idx < len(disc_step_scores):
                score = disc_step_scores[active_step_idx]
                step["step_score"] = score
                step["is_critical"] = score >= 0.5
            else:
                step["step_score"] = None
                step["is_critical"] = None
            
            traj.append(step)
            active_step_idx += 1
        
        return {
            "episode_index": episode_index,
            "traj_uid": traj_uid,
            "task": task,
            "episode_success_score": disc_episode_score,
            "episode_reward": episode_reward,
            "episode_length": len(traj),
            "is_success": episode_reward > 0,
            "discriminator_analysis": disc_analysis,
            "traj": traj
        }
    
    def _get_obs(self, step_data: Dict) -> str:
        """从 step 数据中提取观测文本。"""
        obs = step_data.get('anchor_obs', '')
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            obs = obs[0] if isinstance(obs[0], str) else str(obs[0])
        elif not isinstance(obs, str):
            obs = str(obs) if obs else ""
        return obs
    
    def _extract_reason(self, full_output: str) -> str:
        """从完整输出中提取推理内容（planning/explore/reflection/monitor/think 标签）。"""
        for tag in ['planning', 'explore', 'reflection', 'monitor', 'think']:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, full_output, re.DOTALL)
            if match:
                return f"<{tag}>{match.group(1)}</{tag}>"
        return ""
    
    def _extract_action(self, full_output: str) -> str:
        """从完整输出中提取 action。"""
        action_match = re.search(r'<action>(.*?)</action>', full_output, re.DOTALL)
        if action_match:
            return action_match.group(1).strip()
        return full_output.strip()
    
    def _format_expert_trajectory(self, expert_traj: List[Dict]) -> List[Dict]:
        """
        格式化 expert 轨迹。
        
        支持两种格式：
        - 旧格式 AlfWorld: {observation, action}
        - 新格式 SciWorld: {step_index, obs, action, obs_after}
        """
        formatted = []
        for idx, step in enumerate(expert_traj):
            # 优先使用新格式的 "obs" 字段（obs_before），其次是旧格式的 "observation"
            obs = step.get("obs", step.get("observation", ""))
            
            formatted_step = {
                "step_index": step.get("step_index", idx),
                "obs": obs,
                "action": step.get("action", ""),
            }
            
            # 如果有 obs_after，也保留它
            if "obs_after" in step:
                formatted_step["obs_after"] = step["obs_after"]
            
            formatted.append(formatted_step)
        return formatted
