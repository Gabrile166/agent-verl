# Copyright 2025 Hybrid Reward Integration
# RLVMR module for agent-verl
# Provides discriminator-based reward calculation and hybrid advantage computation

from rlvmr.discriminator_reward import (
    DiscriminatorRewardCalculator,
    DiscriminatorConfig,
    format_policy_trajectory,
    format_expert_trajectory,
)
from rlvmr.core_hybrid import compute_hybrid_outcome_advantage
from rlvmr.expert_trajectory import (
    ExpertTrajectoryGeneratorBase,
    AlfWorldExpertGenerator,
    create_expert_generator,
    create_expert_generator_from_config,
    register_expert_generator,
)
from rlvmr.trajectory_saver import TrajectorySaver

__all__ = [
    "DiscriminatorRewardCalculator",
    "DiscriminatorConfig",
    "format_policy_trajectory",
    "format_expert_trajectory",
    "compute_hybrid_outcome_advantage",
    "ExpertTrajectoryGeneratorBase",
    "AlfWorldExpertGenerator",
    "create_expert_generator",
    "create_expert_generator_from_config",
    "register_expert_generator",
    "TrajectorySaver",
]
