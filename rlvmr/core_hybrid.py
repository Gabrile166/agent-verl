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
    
    基于 GRPO 的 group-relative normalization，支持三种模式：
    - "grpo": 仅使用环境奖励（原始 GRPO）
    - "discriminator": 仅使用 Discriminator 奖励
    - "hybrid": 加权融合两种奖励
    
    Args:
        token_level_rewards: (batch_size, response_length) Token-level rewards from environment
        response_mask: (batch_size, response_length) Mask for valid tokens
        index: (batch_size,) Group indices for normalization (uid)
        traj_index: (batch_size,) Trajectory indices (traj_uid)
        env_step_rewards: (batch_size,) Step-level rewards from environment (optional)
        disc_step_rewards: (batch_size,) Step-level rewards from Discriminator (optional)
        disc_episode_rewards: (batch_size,) Episode-level rewards from Discriminator (optional)
        epsilon: Small value to avoid division by zero
        episode_reward_weight: Weight for episode-level rewards
        step_reward_weight: Weight for step-level rewards
        reward_mode: "grpo" | "discriminator" | "hybrid"
        norm_adv_by_std: Whether to normalize by standard deviation
    
    Returns:
        advantages: (batch_size, response_length) Computed advantages
        returns: (batch_size, response_length) Computed returns (same as advantages for GRPO-style)
    """
    response_length = token_level_rewards.shape[-1]
    device = token_level_rewards.device
    
    # 计算 episode scores（环境奖励）
    env_episode_scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    if reward_mode == "grpo":
        # 原始 GRPO：仅使用环境奖励
        combined_scores = env_episode_scores
        
    elif reward_mode == "discriminator":
        # 仅使用 Discriminator 奖励
        if disc_episode_rewards is None:
            raise ValueError("disc_episode_rewards is required for 'discriminator' mode")
        combined_scores = disc_episode_rewards.to(device)
        
    elif reward_mode == "hybrid":
        # 混合模式：加权融合
        disc_ep = disc_episode_rewards.to(device) if disc_episode_rewards is not None else torch.zeros_like(env_episode_scores)
        
        # Episode-level 融合
        combined_scores = (
            episode_reward_weight * env_episode_scores + 
            step_reward_weight * disc_ep
        )
        
        # 如果有 step-level 奖励，也加入融合
        if env_step_rewards is not None and disc_step_rewards is not None:
            disc_st = disc_step_rewards.to(device)
            step_bonus = step_reward_weight * disc_st
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
