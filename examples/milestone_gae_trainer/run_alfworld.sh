#!/bin/bash
# Milestone-Guided GAE Training Script for ALFWorld
# This script demonstrates how to use the MilestoneGAE advantage estimator
# with LLM Judge for milestone evaluation on ALFWorld environment.

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ===================== Configuration =====================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
BASE_PATH=$(dirname $(dirname $(realpath $0)))

train_data_size=16
val_data_size=128
group_size=8

# Milestone GAE Configuration
MILESTONE_GAMMA=0.99
MILESTONE_LAMBDA=0.95
MILESTONE_COST=0.01

# Judge LLM Configuration
JUDGE_LLM_URL=${JUDGE_LLM_URL:-"http://127.0.0.1:8000/v1"}
JUDGE_LLM_MODEL=${JUDGE_LLM_MODEL:-"qwen2.5-7b-instruct"}
MILESTONE_TEMPLATE="alfworld"

# ===================== Data Preparation =====================
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ===================== Training =====================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=milestone_gae \
    algorithm.milestone_gae.gamma=$MILESTONE_GAMMA \
    algorithm.milestone_gae.lambda=$MILESTONE_LAMBDA \
    algorithm.milestone_gae.cost=$MILESTONE_COST \
    algorithm.milestone_gae.judge_llm.base_url=$JUDGE_LLM_URL \
    algorithm.milestone_gae.judge_llm.model=$JUDGE_LLM_MODEL \
    algorithm.milestone_gae.milestone_template=$MILESTONE_TEMPLATE \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='milestone_gae_alfworld' \
    trainer.experiment_name='milestone_gae_run' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 \
    ++env.name=alfworld \
    ++env.num_envs=16 \
    ++env.timeout=60 \
    ++env.max_actions=30 \
    ++trainer.multi_turn=True \
    ++data.group_size=$group_size \
    ++trainer.invalid_action_penalty_coef=0.5 \
    "$@"
