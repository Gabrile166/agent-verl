"""
Core Milestone-Guided GAE Algorithm

Implements the Generalized Advantage Estimation (GAE) using LLM-predicted
milestone potentials Φ(s) as the value function estimate.

Key Design Decisions:
- Success: Φ_next = 1.0 (ideal future value)
- Failure/Truncated: Φ_next = Φ_T (preserve current potential)
- The only distinction between success and failure is the final reward r=1
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    """单条轨迹的数据"""
    observations: List[str]
    actions: List[str]
    rewards: List[float]
    done: bool
    success: bool
    task_description: str


def compute_milestone_gae(
    phis: List[float],
    rewards: List[float],
    done: bool,
    success: bool,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """
    计算单条轨迹的 Milestone-Guided GAE 优势值
    
    核心公式:
        δ_t = r_t - c + γ × Φ_{t+1} - Φ_t
        A_t = δ_t + γλ × A_{t+1}
    
    边界条件:
        - 成功: Φ_next = 1.0
        - 失败/截断: Φ_next = Φ_T (保持当前势能)
    
    Args:
        phis: 每步的势能值 Φ(s_t), 由 Judge LLM 计算
        rewards: 每步的环境奖励 (成功=1, 其他=0)
        done: 是否结束
        success: 是否成功完成任务
        gamma: 折扣因子
        lam: GAE 系数 λ
        cost: 时间成本 c
    
    Returns:
        advantages: 每步的优势值 A_t
        returns: 每步的回报值 (用于 value loss, 这里用 phis 代替)
    """
    T = len(phis)
    if T == 0:
        return [], []
    
    # 计算 TD Error
    deltas = []
    for t in range(T):
        r_t = rewards[t] - cost
        
        # 边界处理 (关键设计)
        if t == T - 1:
            if done and success:
                phi_next = 1.0  # 成功: 未来势能为理想值
            else:
                phi_next = phis[t]  # 失败/截断: 保持当前势能
        else:
            phi_next = phis[t + 1]
        
        delta_t = r_t + gamma * phi_next - phis[t]
        deltas.append(delta_t)
    
    # GAE 递归 (从后往前)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae
    
    # Returns 使用 phis 作为 value 估计
    returns = phis.copy()
    
    return advantages, returns


def compute_milestone_gae_advantage(
    trajectories: List[TrajectoryData],
    all_phis: List[List[float]],
    gamma: float = 0.99,
    lam: float = 0.95,
    cost: float = 0.01,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    计算多条轨迹的 Milestone-Guided GAE 优势值 (全局归一化)
    
    Args:
        trajectories: 轨迹数据列表
        all_phis: 每条轨迹的势能序列 (由 Judge 预计算)
        gamma: 折扣因子
        lam: GAE 系数
        cost: 时间成本
        norm_adv_by_std: 是否按标准差归一化
        epsilon: 数值稳定性小量
    
    Returns:
        advantages: 归一化后的优势值
        returns: 回报值
        details: 详细信息字典
    """
    all_advantages = []
    all_returns = []
    traj_lengths = []
    
    # 对每条轨迹计算 GAE
    for traj, phis in zip(trajectories, all_phis):
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=traj.rewards,
            done=traj.done,
            success=traj.success,
            gamma=gamma,
            lam=lam,
            cost=cost,
        )
        all_advantages.extend(adv)
        all_returns.extend(ret)
        traj_lengths.append(len(adv))
    
    # 转为 tensor
    advantages = torch.tensor(all_advantages, dtype=torch.float32)
    returns = torch.tensor(all_returns, dtype=torch.float32)
    
    # 全局归一化
    mean_adv = advantages.mean()
    std_adv = advantages.std()
    
    if norm_adv_by_std:
        advantages = (advantages - mean_adv) / (std_adv + epsilon)
    else:
        advantages = advantages - mean_adv
    
    details = {
        "raw_mean": mean_adv.item(),
        "raw_std": std_adv.item(),
        "traj_lengths": traj_lengths,
        "num_trajectories": len(trajectories),
        "total_steps": len(all_advantages),
    }
    
    return advantages, returns, details


def compute_milestone_gae_from_batch(
    batch_data: Any,
    traj_index: np.ndarray,
    response_mask: torch.Tensor,
    pipeline_data: Any,  # PipelineData — typed as Any to avoid circular import at module load
    gamma: float = 0.99,
    lam: float = 0.95,
    cost: float = 0.05,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute Milestone-Guided GAE from batch data using PipelineData.
    
    Per-trajectory data (phis, episode_reward, success) is read directly from
    pipeline_data.trajectories[traj_uid] via type-safe field access.
    
    Args:
        batch_data: verl framework batch data
        traj_index: trajectory index array (traj_uid strings, per-step)
        response_mask: response mask
        pipeline_data: PipelineData with queries and trajectories
        gamma, lam, cost: GAE parameters
        norm_adv_by_std: whether to normalize by standard deviation
        epsilon: numerical stability
    
    Returns:
        advantages: token-level advantages (batch_size, seq_len)
        returns: token-level returns
        details: diagnostic info dict
    """
    device = response_mask.device
    batch_size, seq_len = response_mask.shape
    
    # Get unique trajectory IDs (preserve insertion order, avoid np.unique alphabetical sort)
    seen = set()
    unique_trajs = []
    for t in traj_index:
        if t not in seen:
            seen.add(t)
            unique_trajs.append(t)
    
    # Per-step storage
    step_advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)
    step_returns = torch.zeros(batch_size, dtype=torch.float32, device=device)
    assigned_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Per-trajectory GAE computation
    traj_details = []
    fallback_count = 0
    for traj_idx, traj_id in enumerate(unique_trajs):
        mask = traj_index == traj_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Unified str conversion for dict key consistency (user note #2)
        traj_id_str = str(traj_id)
        
        # Read per-trajectory data from PipelineData
        traj_record = pipeline_data.trajectories.get(traj_id_str)
        if traj_record is not None:
            phis = traj_record.phis if traj_record.phis is not None else None
            total_reward = traj_record.episode_reward
            success = traj_record.success
        else:
            phis = None
            total_reward = 0.0
            success = False
        
        # Phis fallback: linear growth if not available
        if phis is None:
            phis = [i / max(len(indices), 1) for i in range(len(indices))]
            fallback_count += 1
        
        # Ensure length matches
        T = len(indices)
        if len(phis) < T:
            phis = phis + [phis[-1] if phis else 0.0] * (T - len(phis))
        phis = phis[:T]
        
        # Sparse reward: only at last step
        rewards = [0.0] * T
        if T > 0:
            rewards[-1] = total_reward
        
        done = True  # assume complete trajectories
        
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=rewards,
            done=done,
            success=success,
            gamma=gamma,
            lam=lam,
            cost=cost,
        )
        
        # Write into per-step arrays
        for i, idx in enumerate(indices):
            if i < len(adv):
                step_advantages[idx] = adv[i]
                step_returns[idx] = ret[i]
                assigned_mask[idx] = True
        
        traj_details.append({
            "traj_id": traj_id_str,
            "length": T,
            "success": success,
            "total_reward": total_reward,
            "final_phi": phis[-1] if phis else 0.0,
        })
    
    # 全局归一化（使用 assigned_mask 而非 != 0，避免合法的 0 值被排除）
    valid_mask = assigned_mask
    if valid_mask.sum() > 0:
        mean_adv = step_advantages[valid_mask].mean()
        std_adv = step_advantages[valid_mask].std()
    else:
        mean_adv = torch.tensor(0.0, device=device)
        std_adv = torch.tensor(1.0, device=device)
    
    if norm_adv_by_std:
        step_advantages = (step_advantages - mean_adv) / (std_adv + epsilon)
    else:
        step_advantages = step_advantages - mean_adv
    
    # 扩展到 token level
    advantages = step_advantages.unsqueeze(-1).expand(-1, seq_len) * response_mask
    returns = step_returns.unsqueeze(-1).expand(-1, seq_len) * response_mask
    
    details = {
        "adv/raw_mean": mean_adv.item(),
        "adv/raw_std": std_adv.item(),
        "num_trajectories": len(unique_trajs),
        "total_steps": batch_size,
        "judge/fallback_ratio": fallback_count / max(len(unique_trajs), 1),
    }
    
    # ==================== Phi 质量诊断指标 ====================
    success_final_phis = [d["final_phi"] for d in traj_details if d["success"]]
    failure_final_phis = [d["final_phi"] for d in traj_details if not d["success"]]
    
    if success_final_phis:
        s_phis = torch.tensor(success_final_phis)
        details["phi/success_final/mean"] = s_phis.mean().item()
        details["phi/success_final/min"] = s_phis.min().item()
        details["phi/success_final/max"] = s_phis.max().item()
        details["phi/success_final/std"] = s_phis.std().item() if len(s_phis) > 1 else 0.0
        details["phi/success_final_ge_0.8_ratio"] = (s_phis >= 0.8).float().mean().item()
    
    if failure_final_phis:
        f_phis = torch.tensor(failure_final_phis)
        details["phi/failure_final/mean"] = f_phis.mean().item()
        details["phi/failure_final/min"] = f_phis.min().item()
        details["phi/failure_final/max"] = f_phis.max().item()
        details["phi/failure_final/std"] = f_phis.std().item() if len(f_phis) > 1 else 0.0
        details["phi/failure_final_ge_0.8_ratio"] = (f_phis >= 0.8).float().mean().item()
    
    # Advantage 信号方向验证
    success_advs = [step_advantages[np.where(traj_index == d["traj_id"])[0]].mean().item()
                    for d in traj_details if d["success"]]
    failure_advs = [step_advantages[np.where(traj_index == d["traj_id"])[0]].mean().item()
                    for d in traj_details if not d["success"]]
    if success_advs:
        details["adv/success_mean"] = sum(success_advs) / len(success_advs)
    if failure_advs:
        details["adv/failure_mean"] = sum(failure_advs) / len(failure_advs)
    details["adv/success_ratio"] = len(success_final_phis) / max(len(traj_details), 1)
    
    return advantages, returns, details
