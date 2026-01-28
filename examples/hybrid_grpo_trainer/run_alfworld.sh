#!/bin/bash
# Hybrid GRPO Training Script for ALFWorld
# This script demonstrates how to use the HybridGRPO advantage estimator
# with optional Discriminator rewards for ALFWorld environment.

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ===================== Configuration =====================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
BASE_PATH=$(dirname $(dirname $(realpath $0)))

train_data_size=16
val_data_size=128
group_size=8

# Hybrid Reward Configuration
REWARD_MODE="grpo"              # Options: "grpo" | "discriminator" | "hybrid"
EPISODE_REWARD_WEIGHT=1.0       # Weight for episode-level rewards
STEP_REWARD_WEIGHT=1.0          # Weight for step-level rewards

# Discriminator Configuration (only used when REWARD_MODE="discriminator" or "hybrid")
DISCRIMINATOR_ENABLE=False      # Set to True to enable Discriminator
DISCRIMINATOR_URL="http://127.0.0.1:8080/v1"

# ===================== Data Preparation =====================
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ===================== Training =====================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=hybrid_grpo \
    algorithm.hybrid_reward.enable=True \
    algorithm.hybrid_reward.reward_mode=$REWARD_MODE \
    algorithm.hybrid_reward.episode_reward_weight=$EPISODE_REWARD_WEIGHT \
    algorithm.hybrid_reward.step_reward_weight=$STEP_REWARD_WEIGHT \
    algorithm.discriminator.enable=$DISCRIMINATOR_ENABLE \
    algorithm.discriminator.base_urls="[$DISCRIMINATOR_URL]" \
    data.train_files=$BASE_PATH/data/text/train.parquet \
    data.val_files=$BASE_PATH/data/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='hybrid_grpo_qwen2.5_3b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True $@
