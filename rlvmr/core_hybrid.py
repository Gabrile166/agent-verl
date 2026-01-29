"""
Hybrid Advantage Computation Module

提供基于 GRPO 的 Hybrid 优势函数计算，支持融合环境奖励和 Discriminator 奖励。

正确逻辑：分别归一化 Episode 和 Step 优势，再加权合并。
（参考 Agent-PRM 的 core_rlvmr.py 实现）

Usage:
    from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
    
    advantages, returns, details = compute_hybrid_outcome_advantage(
        token_level_rewards=data.batch['token_level_rewards'],
        response_mask=data.batch['response_mask'],
        index=data.non_tensor_batch['uid'],
        traj_index=data.non_tensor_batch['traj_uid'],
        disc_episode_rewards=hybrid_reward_config.get('disc_episode_rewards'),
        disc_step_rewards=hybrid_reward_config.get('disc_step_rewards'),
        reward_mode="hybrid"
    )
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Tuple, Optional, Dict


def compute_hybrid_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    env_step_rewards: Optional[torch.Tensor] = None,
    disc_step_rewards: Optional[torch.Tensor] = None,
    disc_episode_rewards: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    episode_reward_weight: float = 1.0,
    step_reward_weight: float = 1.0,
    reward_mode: str = "grpo",
    norm_adv_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    计算 Hybrid 优势函数，融合环境奖励和 Discriminator 奖励。
    
    正确逻辑：
    1. Episode 优势：基于 trajectory 级别奖励，单独归一化
    2. Step 优势：基于 step 级别奖励，单独归一化
    3. 加权合并：total_adv = w1 × episode_adv + w2 × step_adv
    
    支持三种模式：
    - "grpo": 仅使用环境 episode 奖励（原始 GRPO）
    - "discriminator": Episode 和 Step 奖励都来自 Discriminator
    - "hybrid": Episode 来自环境，Step 来自 Discriminator
    
    Args:
        token_level_rewards: (num_steps, response_length) Token-level rewards
        response_mask: (num_steps, response_length) Mask for valid tokens
        index: (num_steps,) Group indices for normalization (uid)
        traj_index: (num_steps,) Trajectory indices (traj_uid)
        env_step_rewards: (num_steps,) Step-level rewards from environment (optional)
        disc_step_rewards: (num_trajectories,) Discriminator step scores
        disc_episode_rewards: (num_trajectories,) Discriminator episode scores
        epsilon: Small value to avoid division by zero
        episode_reward_weight: Weight for episode-level advantages
        step_reward_weight: Weight for step-level advantages
        reward_mode: "grpo" | "discriminator" | "hybrid"
        norm_adv_by_std: Whether to normalize by standard deviation
    
    Returns:
        advantages: (num_steps, response_length) Total advantages
        returns: (num_steps, response_length) Same as advantages
        details: Dict containing episode_advantages and step_advantages
    """
    response_length = token_level_rewards.shape[-1]
    device = token_level_rewards.device
    num_steps = token_level_rewards.shape[0]
    
    # 计算环境 episode scores（每个 step 的 token 奖励求和）
    env_episode_scores = token_level_rewards.sum(dim=-1)  # (num_steps,)
    
    if reward_mode == "grpo":
        # =========================
        # GRPO 模式：仅使用环境 episode 奖励
        # =========================
        episode_advantages = _compute_episode_advantage(
            episode_rewards=env_episode_scores,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std=norm_adv_by_std,
        )
        step_advantages = torch.zeros_like(episode_advantages)
        total_advantages = episode_advantages
        
    elif reward_mode == "discriminator":
        # =========================
        # Discriminator 模式：Episode 和 Step 都来自 Discriminator
        # =========================
        if disc_episode_rewards is None:
            raise ValueError("disc_episode_rewards is required for 'discriminator' mode")
        
        # 将 trajectory-level 奖励扩展到 step-level
        disc_ep_expanded = _expand_traj_to_step(disc_episode_rewards, traj_index, device)
        
        # Episode 优势（单独归一化）
        episode_advantages = _compute_episode_advantage(
            episode_rewards=disc_ep_expanded,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std=norm_adv_by_std,
        )
        
        # Step 优势（单独归一化）
        if disc_step_rewards is not None:
            disc_st_expanded = _expand_traj_to_step(disc_step_rewards, traj_index, device)
            step_advantages = _compute_step_advantage(
                step_rewards=disc_st_expanded,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                norm_adv_by_std=norm_adv_by_std,
            )
        else:
            step_advantages = torch.zeros_like(episode_advantages)
        
        # 加权合并优势（不是分数！）
        total_advantages = (
            episode_reward_weight * episode_advantages + 
            step_reward_weight * step_advantages
        )
        
    elif reward_mode == "hybrid":
        # =========================
        # Hybrid 模式：Episode 来自环境，Step 来自 Discriminator
        # =========================
        
        # Episode 优势：使用环境 episode 奖励（单独归一化）
        episode_advantages = _compute_episode_advantage(
            episode_rewards=env_episode_scores,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std=norm_adv_by_std,
        )
        
        # Step 优势：使用 Discriminator step 奖励（单独归一化）
        if disc_step_rewards is not None:
            disc_st_expanded = _expand_traj_to_step(disc_step_rewards, traj_index, device)
            step_advantages = _compute_step_advantage(
                step_rewards=disc_st_expanded,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                norm_adv_by_std=norm_adv_by_std,
            )
        else:
            step_advantages = torch.zeros_like(episode_advantages)
        
        # 加权合并优势
        total_advantages = (
            episode_reward_weight * episode_advantages + 
            step_reward_weight * step_advantages
        )
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    
    details = {
        "episode_advantages": episode_advantages,
        "step_advantages": step_advantages,
        "weighted_episode_advantages": episode_reward_weight * episode_advantages,
        "weighted_step_advantages": step_reward_weight * step_advantages,
    }
    
    return total_advantages, total_advantages, details


def _compute_episode_advantage(
    episode_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
) -> torch.Tensor:
    """
    计算 Episode 级别优势。
    
    使用 GRPO 风格的组归一化：advantage = (reward - group_mean) / group_std
    
    注意：同一个 trajectory 的多个 steps 共享同一个 episode 优势。
    去重逻辑：按 (index, traj_index) 组合去重后再计算组均值/标准差。
    
    Args:
        episode_rewards: (num_steps,) Episode rewards (可能是扩展后的)
        response_mask: (num_steps, response_length) Mask for valid tokens
        index: (num_steps,) Group indices (uid)
        traj_index: (num_steps,) Trajectory indices (traj_uid)
        epsilon: Numerical stability
        norm_adv_by_std: Whether to divide by std
    
    Returns:
        episode_advantages: (num_steps, response_length)
    """
    response_length = response_mask.shape[-1]
    device = episode_rewards.device
    scores = episode_rewards.clone()
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 按 (index, traj_index) 去重，收集每个 group 的 scores
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add(pair)
        
        # 计算每个 group 的均值和标准差
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
                id2std[idx] = torch.std(torch.stack(group_scores))
        
        # 归一化
        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        # 扩展到 response_length
        episode_advantages = scores.unsqueeze(-1).expand(-1, response_length) * response_mask
    
    return episode_advantages


def _compute_step_advantage(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
) -> torch.Tensor:
    """
    计算 Step 级别优势。
    
    对同一 group (index) 内的所有 steps 进行归一化。
    不需要按 traj_index 去重，因为每个 step 都有独立的 step 奖励。
    
    Args:
        step_rewards: (num_steps,) Step rewards
        response_mask: (num_steps, response_length) Mask for valid tokens
        index: (num_steps,) Group indices (uid)
        epsilon: Numerical stability
        norm_adv_by_std: Whether to divide by std
    
    Returns:
        step_advantages: (num_steps, response_length)
    """
    response_length = response_mask.shape[-1]
    device = step_rewards.device
    scores = step_rewards.clone()
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 收集每个 group 的所有 step scores（不去重）
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每个 group 的均值和标准差
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
                id2std[idx] = torch.std(torch.stack(group_scores))
        
        # 归一化
        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        # 扩展到 response_length
        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * response_mask
    
    return step_advantages


def _expand_traj_to_step(
    traj_rewards: torch.Tensor,
    traj_index: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    将 trajectory-level 奖励扩展到 step-level。
    
    通过 traj_index 将每个 trajectory 的奖励复制到该 trajectory 的所有 steps。
    
    Args:
        traj_rewards: (num_unique_trajs,) Trajectory-level rewards
        traj_index: (num_steps,) Trajectory index for each step
        device: Target device
    
    Returns:
        step_rewards: (num_steps,) Expanded step-level rewards
    """
    num_steps = len(traj_index)
    step_rewards = torch.zeros(num_steps, device=device, dtype=traj_rewards.dtype)
    
    # 获取唯一的 traj_uid 及其映射
    unique_trajs = list(set(traj_index))
    unique_trajs.sort()  # 保持顺序一致性
    traj_to_idx = {t: i for i, t in enumerate(unique_trajs)}
    
    # 维度检查
    if len(traj_rewards) != len(unique_trajs):
        print(f"[Hybrid] Warning: traj_rewards={len(traj_rewards)}, unique_trajs={len(unique_trajs)}")
    
    for step_i in range(num_steps):
        traj_uid = traj_index[step_i]
        traj_idx = traj_to_idx.get(traj_uid, 0)
        if traj_idx < len(traj_rewards):
            step_rewards[step_i] = traj_rewards[traj_idx]
    
    return step_rewards
