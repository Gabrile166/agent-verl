# Copyright 2025 Hybrid Reward Integration
# RLVMR module for agent-verl
# Provides discriminator-based reward calculation and hybrid advantage computation

from rlvmr.discriminator_reward import DiscriminatorRewardCalculator, DiscriminatorConfig
from rlvmr.core_hybrid import compute_hybrid_outcome_advantage, compute_hybrid_step_advantage

__all__ = [
    "DiscriminatorRewardCalculator",
    "DiscriminatorConfig",
    "compute_hybrid_outcome_advantage",
    "compute_hybrid_step_advantage",
]
