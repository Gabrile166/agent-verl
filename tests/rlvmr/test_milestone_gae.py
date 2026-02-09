"""
Unit Tests for Milestone-Guided GAE

测试 core_milestone_gae.py 中的优势计算逻辑，确保：
1. GAE 递归计算正确
2. 边界条件处理正确 (成功/失败/截断)
3. 全局归一化逻辑正确
"""

import numpy as np
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlvmr.core_milestone_gae import (
    compute_milestone_gae,
    compute_milestone_gae_advantage,
    TrajectoryData,
)


def test_basic_gae_computation():
    """测试基本的 GAE 计算"""
    print("\n" + "="*60)
    print("Test: Basic GAE Computation")
    print("="*60)
    
    # 简单的 5 步轨迹
    phis = [0.2, 0.2, 0.4, 0.4, 1.0]  # 达到 2 个里程碑
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0]  # 只有最后一步有奖励
    
    advantages, returns = compute_milestone_gae(
        phis=phis,
        rewards=rewards,
        done=True,
        success=True,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"phis: {phis}")
    print(f"rewards: {rewards}")
    print(f"advantages: {[round(a, 4) for a in advantages]}")
    
    # 验证：所有 advantage 应该 > 0（因为最终成功）
    assert all(a > 0 for a in advantages), f"All advantages should be positive for success"
    # 验证：advantage 经过 GAE 倒推，前面的 advantage 可能略高于最后一步
    # 因为 δ 值在里程碑跳跃时为正，加上倒推的成功信号
    print("[PASSED]")


def test_failed_trajectory_boundary():
    """测试失败轨迹的边界条件 (势能保持)"""
    print("\n" + "="*60)
    print("Test: Failed Trajectory Boundary (Preserve Potential)")
    print("="*60)
    
    # 失败轨迹：达到 Φ=0.8 后截断
    phis = [0.15, 0.30, 0.45, 0.60, 0.80]
    rewards = [0.0, 0.0, 0.0, 0.0, 0.0]  # 没有奖励
    
    advantages, returns = compute_milestone_gae(
        phis=phis,
        rewards=rewards,
        done=True,
        success=False,  # 失败
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"phis: {phis}")
    print(f"rewards: {rewards}")
    print(f"advantages: {[round(a, 4) for a in advantages]}")
    
    # 验证：最后一步的 δ 应该接近 -0.018 (不是 -0.81)
    # δ_T = r_T - c + γ × Φ_T - Φ_T = 0 - 0.01 + 0.99×0.8 - 0.8 = -0.018
    expected_delta_T = 0 - 0.01 + 0.99 * 0.8 - 0.8
    print(f"Expected δ_T: {expected_delta_T:.4f}")
    
    # 验证：advantage 应该接近 0（不是巨大负值）
    assert abs(advantages[-1]) < 0.5, f"Final advantage should be small, got {advantages[-1]}"
    print("[PASSED]")


def test_success_vs_failure_comparison():
    """测试成功 vs 失败轨迹的对比"""
    print("\n" + "="*60)
    print("Test: Success vs Failure Comparison")
    print("="*60)
    
    # 成功轨迹
    success_phis = [0.2, 0.4, 0.6, 0.8, 1.0]
    success_rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
    success_adv, _ = compute_milestone_gae(
        phis=success_phis,
        rewards=success_rewards,
        done=True,
        success=True,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    # 失败轨迹 (同样进度，但没有最终奖励)
    fail_phis = [0.2, 0.4, 0.6, 0.8, 0.8]  # 最后停在 0.8
    fail_rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
    fail_adv, _ = compute_milestone_gae(
        phis=fail_phis,
        rewards=fail_rewards,
        done=True,
        success=False,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"Success advantages: {[round(a, 4) for a in success_adv]}")
    print(f"Failure advantages: {[round(a, 4) for a in fail_adv]}")
    print(f"Success mean: {np.mean(success_adv):.4f}")
    print(f"Failure mean: {np.mean(fail_adv):.4f}")
    
    # 验证：成功轨迹的平均 advantage 显著高于失败轨迹
    assert np.mean(success_adv) > np.mean(fail_adv), "Success should have higher mean advantage"
    # 验证：差距主要来自最终奖励 (约 1.0)
    gap = np.mean(success_adv) - np.mean(fail_adv)
    print(f"Gap: {gap:.4f}")
    assert gap > 0.5, f"Gap should be significant (>0.5), got {gap}"
    print("[PASSED]")


def test_global_normalization():
    """测试全局归一化"""
    print("\n" + "="*60)
    print("Test: Global Normalization")
    print("="*60)
    
    # 创建两条轨迹
    traj1 = TrajectoryData(
        observations=["obs1", "obs2", "obs3"],
        actions=["act1", "act2", "act3"],
        rewards=[0.0, 0.0, 1.0],
        done=True,
        success=True,
        task_description="task1",
    )
    traj2 = TrajectoryData(
        observations=["obs1", "obs2"],
        actions=["act1", "act2"],
        rewards=[0.0, 0.0],
        done=True,
        success=False,
        task_description="task2",
    )
    
    phis1 = [0.3, 0.6, 1.0]
    phis2 = [0.3, 0.5]
    
    advantages, returns, details = compute_milestone_gae_advantage(
        trajectories=[traj1, traj2],
        all_phis=[phis1, phis2],
        gamma=0.99,
        lam=0.95,
        cost=0.01,
        norm_adv_by_std=True,
    )
    
    print(f"Raw mean: {details['raw_mean']:.4f}")
    print(f"Raw std: {details['raw_std']:.4f}")
    print(f"Normalized advantages mean: {advantages.mean().item():.6f}")
    print(f"Normalized advantages std: {advantages.std().item():.4f}")
    
    # 验证：归一化后均值接近 0
    assert abs(advantages.mean().item()) < 0.01, "Mean should be ~0 after normalization"
    # 验证：归一化后标准差接近 1
    assert abs(advantages.std().item() - 1.0) < 0.1, "Std should be ~1 after normalization"
    print("[PASSED]")


def test_empty_trajectory():
    """测试空轨迹处理"""
    print("\n" + "="*60)
    print("Test: Empty Trajectory")
    print("="*60)
    
    advantages, returns = compute_milestone_gae(
        phis=[],
        rewards=[],
        done=True,
        success=False,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    assert advantages == [], "Empty trajectory should return empty list"
    assert returns == [], "Empty trajectory should return empty list"
    print("[PASSED]")


def test_single_step_trajectory():
    """测试单步轨迹"""
    print("\n" + "="*60)
    print("Test: Single Step Trajectory")
    print("="*60)
    
    # 成功的单步轨迹
    advantages, returns = compute_milestone_gae(
        phis=[1.0],
        rewards=[1.0],
        done=True,
        success=True,
        gamma=0.99,
        lam=0.95,
        cost=0.01,
    )
    
    print(f"Single step advantage: {advantages[0]:.4f}")
    
    # δ = 1.0 - 0.01 + 0.99×1.0 - 1.0 = 0.98
    expected = 1.0 - 0.01 + 0.99 * 1.0 - 1.0
    assert abs(advantages[0] - expected) < 0.01, f"Expected {expected}, got {advantages[0]}"
    print("[PASSED]")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Running Milestone-Guided GAE Tests")
    print("="*60)
    
    tests = [
        test_basic_gae_computation,
        test_failed_trajectory_boundary,
        test_success_vs_failure_comparison,
        test_global_normalization,
        test_empty_trajectory,
        test_single_step_trajectory,
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
