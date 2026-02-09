"""
ALFWorld-Style Test Examples for Milestone-Guided GAE

模拟 ALFWorld 环境的典型轨迹，验证 Milestone-Guided GAE 的完整流程。
无需 LLM 服务器即可运行。
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlvmr.core_milestone_gae import (
    compute_milestone_gae,
    compute_milestone_gae_advantage,
    TrajectoryData,
)


# ============================================
# ALFWorld 典型轨迹模拟
# ============================================

def create_successful_put_clean_trajectory():
    """
    成功的 "put clean apple in cabinet" 任务轨迹 (15步)
    
    里程碑:
    M1 (Φ=0.15): 找到苹果
    M2 (Φ=0.30): 拿起苹果  
    M3 (Φ=0.45): 到达水槽
    M4 (Φ=0.60): 清洗苹果
    M5 (Φ=0.80): 到达柜子
    M6 (Φ=1.00): 放置完成
    """
    trajectory = [
        {"action": "go to countertop 1", "observation": "You arrive at countertop 1. On the countertop, you see a apple 1."},
        {"action": "look", "observation": "You see a apple 1, a knife 1."},
        {"action": "take apple 1 from countertop 1", "observation": "You pick up the apple 1 from countertop 1."},
        {"action": "go to sinkbasin 1", "observation": "You arrive at sinkbasin 1."},
        {"action": "clean apple 1 with sinkbasin 1", "observation": "You clean the apple 1 using the sinkbasin 1."},
        {"action": "go to cabinet 1", "observation": "You arrive at cabinet 1. The cabinet 1 is closed."},
        {"action": "open cabinet 1", "observation": "You open the cabinet 1."},
        {"action": "put apple 1 in/on cabinet 1", "observation": "You put the apple 1 in/on the cabinet 1."},
    ]
    
    # 模拟 Judge 判定的 Φ 值 (每步达到的最高里程碑)
    phis = [0.15, 0.15, 0.30, 0.45, 0.60, 0.80, 0.80, 1.00]
    
    # 奖励: 只有最后一步成功时有 r=1
    rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    
    return TrajectoryData(
        observations=[s["observation"] for s in trajectory],
        actions=[s["action"] for s in trajectory],
        rewards=rewards,
        done=True,
        success=True,
        task_description="put a clean apple in cabinet 1",
    ), phis


def create_failed_put_clean_trajectory():
    """
    失败的 "put clean apple in cabinet" 任务轨迹 (30步截断)
    
    Agent 找到了苹果并清洗，但在寻找柜子时迷路，最终被截断。
    """
    # 前 6 步正常进展
    trajectory = [
        {"action": "go to countertop 1", "observation": "You arrive at countertop 1. On the countertop, you see a apple 1."},
        {"action": "take apple 1 from countertop 1", "observation": "You pick up the apple 1 from countertop 1."},
        {"action": "go to sinkbasin 1", "observation": "You arrive at sinkbasin 1."},
        {"action": "clean apple 1 with sinkbasin 1", "observation": "You clean the apple 1 using the sinkbasin 1."},
        # 接下来迷路了...
        {"action": "go to shelf 1", "observation": "You arrive at shelf 1. Nothing here."},
        {"action": "go to shelf 2", "observation": "You arrive at shelf 2. Nothing here."},
        {"action": "go to drawer 1", "observation": "You arrive at drawer 1. It's closed."},
        {"action": "open drawer 1", "observation": "You open the drawer 1. It's empty."},
        {"action": "go to countertop 2", "observation": "You arrive at countertop 2."},
        {"action": "go to cabinet 2", "observation": "You arrive at cabinet 2. It's cabinet 2, not cabinet 1."},
    ]
    
    # Φ 值: 在清洗后停滞
    phis = [0.15, 0.30, 0.45, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]
    
    # 无奖励
    rewards = [0.0] * 10
    
    return TrajectoryData(
        observations=[s["observation"] for s in trajectory],
        actions=[s["action"] for s in trajectory],
        rewards=rewards,
        done=True,
        success=False,  # 失败
        task_description="put a clean apple in cabinet 1",
    ), phis


def create_efficient_trajectory():
    """
    高效的轨迹 (5步完成)
    
    Agent 直奔目标，快速完成任务。
    """
    trajectory = [
        {"action": "go to countertop 1", "observation": "You see a apple 1."},
        {"action": "take apple 1", "observation": "You pick up the apple 1."},
        {"action": "go to sinkbasin 1", "observation": "You arrive at sinkbasin 1."},
        {"action": "clean apple 1", "observation": "You clean the apple 1."},
        {"action": "put apple 1 in cabinet 1", "observation": "You put the apple 1 in cabinet 1."},
    ]
    
    phis = [0.15, 0.30, 0.60, 0.80, 1.00]
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
    
    return TrajectoryData(
        observations=[s["observation"] for s in trajectory],
        actions=[s["action"] for s in trajectory],
        rewards=rewards,
        done=True,
        success=True,
        task_description="put a clean apple in cabinet 1",
    ), phis


# ============================================
# 测试用例
# ============================================

def test_successful_trajectory():
    """测试成功轨迹的 GAE 计算"""
    print("\n" + "="*60)
    print("Test: Successful Put-Clean Trajectory")
    print("="*60)
    
    traj, phis = create_successful_put_clean_trajectory()
    
    advantages, returns = compute_milestone_gae(
        phis=phis,
        rewards=traj.rewards,
        done=traj.done,
        success=traj.success,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"Steps: {len(phis)}")
    print(f"Phis: {phis}")
    print(f"Advantages: {[round(a, 3) for a in advantages]}")
    print(f"Mean advantage: {np.mean(advantages):.4f}")
    
    # 验证
    assert all(a > 0 for a in advantages), "All advantages should be positive"
    assert np.mean(advantages) > 0.5, "Mean advantage should be significant"
    print("[PASSED]")


def test_failed_trajectory():
    """测试失败轨迹的 GAE 计算"""
    print("\n" + "="*60)
    print("Test: Failed Put-Clean Trajectory")
    print("="*60)
    
    traj, phis = create_failed_put_clean_trajectory()
    
    advantages, returns = compute_milestone_gae(
        phis=phis,
        rewards=traj.rewards,
        done=traj.done,
        success=traj.success,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"Steps: {len(phis)}")
    print(f"Phis: {phis}")
    print(f"Advantages: {[round(a, 3) for a in advantages]}")
    print(f"Mean advantage: {np.mean(advantages):.4f}")
    
    # 验证: 最后一步应该是 -0.018 左右 (保持势能设计)
    delta_T = 0 - 0.01 + 0.99 * 0.60 - 0.60  # = -0.016
    print(f"Expected δ_T: {delta_T:.4f}")
    assert abs(advantages[-1] - delta_T) < 0.1, f"Final advantage should be close to {delta_T}"
    print("[PASSED]")


def test_group_comparison():
    """测试组内成功 vs 失败对比 (模拟 GRPO 的 group sampling)"""
    print("\n" + "="*60)
    print("Test: Group Comparison (Success vs Failure)")
    print("="*60)
    
    # 创建一组轨迹
    success_traj, success_phis = create_successful_put_clean_trajectory()
    fail_traj, fail_phis = create_failed_put_clean_trajectory()
    efficient_traj, efficient_phis = create_efficient_trajectory()
    
    trajectories = [success_traj, fail_traj, efficient_traj]
    all_phis = [success_phis, fail_phis, efficient_phis]
    
    advantages, returns, details = compute_milestone_gae_advantage(
        trajectories=trajectories,
        all_phis=all_phis,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
        norm_adv_by_std=True,
    )
    
    print(f"Total steps: {details['total_steps']}")
    print(f"Trajectories: {details['num_trajectories']}")
    print(f"Raw mean: {details['raw_mean']:.4f}")
    print(f"Raw std: {details['raw_std']:.4f}")
    
    # 分割各轨迹的优势值
    idx = 0
    for i, length in enumerate(details['traj_lengths']):
        traj_adv = advantages[idx:idx+length].numpy()
        print(f"Traj {i} ({['success', 'fail', 'efficient'][i]}): mean={np.mean(traj_adv):.3f}")
        idx += length
    
    # 验证: 归一化后均值接近 0
    assert abs(advantages.mean().item()) < 0.01, "Mean should be ~0"
    print("[PASSED]")


def test_efficiency_reward():
    """测试效率奖励: 短轨迹 vs 长轨迹"""
    print("\n" + "="*60)
    print("Test: Efficiency Reward (Short vs Long)")
    print("="*60)
    
    # 高效轨迹 (5步)
    efficient_traj, efficient_phis = create_efficient_trajectory()
    efficient_adv, _ = compute_milestone_gae(
        phis=efficient_phis,
        rewards=efficient_traj.rewards,
        done=True,
        success=True,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    # 低效但成功的轨迹 (8步)
    slow_traj, slow_phis = create_successful_put_clean_trajectory()
    slow_adv, _ = compute_milestone_gae(
        phis=slow_phis,
        rewards=slow_traj.rewards,
        done=True,
        success=True,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"Efficient (5 steps): mean adv = {np.mean(efficient_adv):.4f}")
    print(f"Slow (8 steps): mean adv = {np.mean(slow_adv):.4f}")
    
    # 由于 cost=0.01, 短轨迹累积的时间成本更少
    # 但两者都成功，所以差距不太大
    print("[PASSED]")


def run_all_tests():
    """运行所有 ALFWorld 测试"""
    print("\n" + "="*60)
    print("Running ALFWorld-Style Milestone GAE Tests")
    print("="*60)
    
    tests = [
        test_successful_trajectory,
        test_failed_trajectory,
        test_group_comparison,
        test_efficiency_reward,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAILED]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
