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
    phis_dict: Dict[str, List[float]],
    traj_index: np.ndarray,
    response_mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost: float = 0.05,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-8,
    episode_rewards: Optional[np.ndarray] = None,
    success_flags: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    从 batch 数据计算 Milestone-Guided GAE (兼容 verl 框架)
    
    Args:
        batch_data: verl 框架的 batch 数据
        phis_dict: 势能值字典，以 traj_uid 为键，值为 [phi_0, phi_1, ...]
        traj_index: 轨迹索引数组 (traj_uid 字符串)
        response_mask: 响应掩码
        gamma, lam, cost: GAE 参数
        norm_adv_by_std: 是否按标准差归一化
        epsilon: 数值稳定性小量
        episode_rewards: 每条轨迹的总奖励 (来自环境, e.g., 10 for success)
        success_flags: 每条轨迹是否成功的标志
    
    Returns:
        advantages: token-level 优势值 (batch_size, seq_len)
        returns: token-level 回报值
        details: 详细信息
    """
    device = response_mask.device
    batch_size, seq_len = response_mask.shape
    
    # 获取唯一的轨迹索引 (traj_uid 字符串)
    unique_trajs = np.unique(traj_index)
    
    # 构建 traj_uid -> 索引的映射 (用于 episode_rewards 和 success_flags)
    unique_trajs_list = list(unique_trajs)
    
    # 存储每个 step 的优势值
    step_advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)
    step_returns = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # 对每条轨迹计算 GAE
    traj_details = []
    for traj_idx, traj_id in enumerate(unique_trajs):
        mask = traj_index == traj_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # 获取该轨迹的 phis (使用字典查找)
        if traj_id in phis_dict:
            phis = phis_dict[traj_id]
        else:
            # Fallback: 使用线性增长
            phis = [i / len(indices) for i in range(len(indices))]
        
        # 确保长度匹配
        T = len(indices)
        if len(phis) < T:
            phis = phis + [phis[-1] if phis else 0.0] * (T - len(phis))
        phis = phis[:T]
        
        # ==================== 真实奖励提取 ====================
        # 获取该轨迹的总奖励（来自环境）
        # 使用 traj_idx (整数索引) 访问 episode_rewards 而非 traj_id (UUID 字符串)
        if episode_rewards is not None and traj_idx < len(episode_rewards):
            total_reward = float(episode_rewards[traj_idx])
        else:
            # Fallback: 如果没有提供，根据最终 phi 判断
            total_reward = 10.0 if (phis and phis[-1] >= 0.99) else 0.0
        
        # 获取该轨迹是否成功
        if success_flags is not None and traj_idx < len(success_flags):
            success = bool(success_flags[traj_idx])
        else:
            # Fallback: 根据奖励判断（ALFWorld: r=10 表示成功）
            success = total_reward >= 10.0
        
        # 稀疏奖励分配：只在最后一步给予奖励
        rewards = [0.0] * T
        if T > 0:
            rewards[-1] = total_reward  # 最后一步获得全部奖励
        
        done = True  # 假设都是完整轨迹
        
        adv, ret = compute_milestone_gae(
            phis=phis,
            rewards=rewards,
            done=done,
            success=success,
            gamma=gamma,
            lam=lam,
            cost=cost,
        )
        
        # 写入对应位置
        for i, idx in enumerate(indices):
            if i < len(adv):
                step_advantages[idx] = adv[i]
                step_returns[idx] = ret[i]
        
        traj_details.append({
            "traj_id": int(traj_id),
            "length": T,
            "success": success,
            "total_reward": total_reward,
            "final_phi": phis[-1] if phis else 0.0,
        })
    
    # 全局归一化
    valid_mask = step_advantages != 0  # 简单过滤
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
        "raw_mean": mean_adv.item(),
        "raw_std": std_adv.item(),
        "num_trajectories": len(unique_trajs),
        "total_steps": batch_size,
        "traj_details": traj_details,
    }
    
    return advantages, returns, details
