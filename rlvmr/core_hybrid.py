"""
Hybrid Advantage Computation Module

提供基于 GRPO 的 Hybrid 优势函数计算，支持融合环境奖励和 Discriminator 奖励。

Usage:
    from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
    
    advantages, returns = compute_hybrid_outcome_advantage(
        token_level_rewards=data.batch['token_level_rewards'],
        env_step_rewards=data.batch['step_rewards'],
        disc_step_rewards=data.batch.get('disc_step_rewards'),
        disc_episode_rewards=data.batch.get('disc_episode_rewards'),
        response_mask=data.batch['response_mask'],
        index=data.non_tensor_batch['uid'],
        traj_index=data.non_tensor_batch['traj_uid'],
        reward_mode="hybrid"
    )
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Tuple, Optional


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Hybrid 优势函数，融合环境奖励和 Discriminator 奖励。
    
    支持 Multi-turn Agent 的数据结构：
    - token_level_rewards: (num_steps, response_length) - step-level 数据
    - disc_episode_rewards: (num_trajectories,) - trajectory-level 数据
    
    通过 traj_index 将 trajectory-level 奖励映射到 step-level。
    
    Args:
        token_level_rewards: (num_steps, response_length) Token-level rewards
        response_mask: (num_steps, response_length) Mask for valid tokens
        index: (num_steps,) Group indices for normalization (uid)
        traj_index: (num_steps,) Trajectory indices (traj_uid)
        env_step_rewards: (num_steps,) Step-level rewards from environment (optional)
        disc_step_rewards: (num_trajectories,) Discriminator step scores, indexed by traj_uid
        disc_episode_rewards: (num_trajectories,) Discriminator episode scores, indexed by traj_uid
        epsilon: Small value to avoid division by zero
        episode_reward_weight: Weight for episode-level rewards
        step_reward_weight: Weight for step-level rewards
        reward_mode: "grpo" | "discriminator" | "hybrid"
        norm_adv_by_std: Whether to normalize by standard deviation
    
    Returns:
        advantages: (num_steps, response_length) Computed advantages
        returns: (num_steps, response_length) Computed returns
    """
    response_length = token_level_rewards.shape[-1]
    device = token_level_rewards.device
    num_steps = token_level_rewards.shape[0]
    
    # 计算 step-level episode scores（环境奖励）
    env_episode_scores = token_level_rewards.sum(dim=-1)  # (num_steps,)
    
    if reward_mode == "grpo":
        # 原始 GRPO：仅使用环境奖励
        combined_scores = env_episode_scores
        
    elif reward_mode == "discriminator":
        # 仅使用 Discriminator 奖励
        if disc_episode_rewards is None:
            raise ValueError("disc_episode_rewards is required for 'discriminator' mode")
        
        # 将 trajectory-level 奖励映射到 step-level
        disc_ep_step = _expand_traj_to_step(disc_episode_rewards, traj_index, device)
        combined_scores = disc_ep_step
        
    elif reward_mode == "hybrid":
        # 混合模式：加权融合
        if disc_episode_rewards is not None:
            disc_ep_step = _expand_traj_to_step(disc_episode_rewards, traj_index, device)
        else:
            disc_ep_step = torch.zeros_like(env_episode_scores)
        
        # Episode-level 融合
        combined_scores = (
            episode_reward_weight * env_episode_scores + 
            step_reward_weight * disc_ep_step
        )
        
        # 如果有 step-level Discriminator 奖励，也加入融合
        if disc_step_rewards is not None:
            disc_st_step = _expand_traj_to_step(disc_step_rewards, traj_index, device)
            step_bonus = step_reward_weight * disc_st_step
            combined_scores = combined_scores + step_bonus
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    
    # GRPO-style group normalization
    advantages = _grpo_normalize(
        scores=combined_scores,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        norm_adv_by_std=norm_adv_by_std
    )
    
    return advantages, advantages


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
    
    # 如果 traj_rewards 维度与 unique_trajs 不同，可能需要处理
    if len(traj_rewards) != len(unique_trajs):
        # Discriminator 返回的是 per-trajectory 奖励，可能需要取均值
        # 这里假设 traj_rewards 按照 trajectory 顺序排列
        print(f"[Hybrid] Warning: traj_rewards={len(traj_rewards)}, unique_trajs={len(unique_trajs)}")
    
    for step_i in range(num_steps):
        traj_uid = traj_index[step_i]
        traj_idx = traj_to_idx.get(traj_uid, 0)
        if traj_idx < len(traj_rewards):
            step_rewards[step_i] = traj_rewards[traj_idx]
    
    return step_rewards


def _grpo_normalize(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True
) -> torch.Tensor:
    """
    GRPO-style group-relative normalization.
    
    对同一 group（相同 index）内的 scores 进行 mean-std normalization。
    
    Args:
        scores: (batch_size,) Raw scores to normalize
        response_mask: (batch_size, response_length) Mask for valid tokens
        index: (batch_size,) Group indices
        traj_index: (batch_size,) Trajectory indices
        epsilon: Small value for numerical stability
        norm_adv_by_std: Whether to normalize by std
    
    Returns:
        advantages: (batch_size, response_length) Normalized advantages
    """
    response_length = response_mask.shape[-1]
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 按 group 收集 scores
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add(pair)
        
        # 计算每个 group 的 mean 和 std
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
                id2std[idx] = torch.std(torch.stack(group_scores))
        
        # 归一化
        normalized_scores = scores.clone()
        for i in range(bsz):
            if norm_adv_by_std:
                normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                normalized_scores[i] = scores[i] - id2mean[index[i]]
        
        # 扩展到 response_length 并应用 mask
        advantages = normalized_scores.unsqueeze(-1).expand(-1, response_length) * response_mask
    
    return advantages


def compute_hybrid_step_advantage(
    env_step_rewards: torch.Tensor,
    disc_step_rewards: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    step_reward_weight: float = 1.0,
    reward_mode: str = "grpo",
    norm_adv_by_std: bool = True
) -> torch.Tensor:
    """
    计算 Step-level 优势（可选，用于 GiGPO 扩展）。
    
    Args:
        env_step_rewards: (batch_size,) Step rewards from environment
        disc_step_rewards: (batch_size,) Step rewards from Discriminator
        response_mask: (batch_size, response_length) Mask for valid tokens
        step_group_uids: (batch_size,) Step group UIDs (from anchor state grouping)
        epsilon: Small value for numerical stability
        step_reward_weight: Weight for discriminator step rewards
        reward_mode: "grpo" | "discriminator" | "hybrid"
        norm_adv_by_std: Whether to normalize by std
    
    Returns:
        step_advantages: (batch_size, response_length) Step-level advantages
    """
    response_length = response_mask.shape[-1]
    device = env_step_rewards.device
    
    # 融合 step rewards
    if reward_mode == "grpo":
        combined_step = env_step_rewards
    elif reward_mode == "discriminator":
        if disc_step_rewards is None:
            raise ValueError("disc_step_rewards required for discriminator mode")
        combined_step = disc_step_rewards.to(device)
    elif reward_mode == "hybrid":
        disc_st = disc_step_rewards.to(device) if disc_step_rewards is not None else torch.zeros_like(env_step_rewards)
        combined_step = env_step_rewards + step_reward_weight * disc_st
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")
    
    # Step group normalization
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = combined_step.shape[0]
        
        for i in range(bsz):
            id2score[step_group_uids[i]].append(combined_step[i])
        
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
                id2std[idx] = torch.std(torch.stack(group_scores))
        
        normalized = combined_step.clone()
        for i in range(bsz):
            if norm_adv_by_std:
                normalized[i] = (combined_step[i] - id2mean[step_group_uids[i]]) / (id2std[step_group_uids[i]] + epsilon)
            else:
                normalized[i] = combined_step[i] - id2mean[step_group_uids[i]]
        
        step_advantages = normalized.unsqueeze(-1).expand(-1, response_length) * response_mask
    
    return step_advantages
