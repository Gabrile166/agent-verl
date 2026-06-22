#!/usr/bin/env bash
# Reproducible Milestone-GAE training script for ALFWorld.
#
# Usage:
#   bash examples/milestone_gae_trainer/run_alfworld.sh [vllm] [hydra overrides...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../release_common.sh
source "${SCRIPT_DIR}/../release_common.sh"

release_parse_engine "$@"
set -- "${REMAINING_ARGS[@]}"
[[ "${DEBUG:-0}" == "1" ]] && set -x

EXP_NAME="${EXP_NAME:-milestone_gae_qwen2_5_7b_alfworld_ood}"
PROJECT_NAME="${PROJECT_NAME:-milestone_gae_alfworld}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
BASE_PATH="${BASE_PATH:-${REPO_ROOT}}"
DATA_DIR="${DATA_DIR:-${BASE_PATH}/data/verl-agent/text}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_PATH}/output/${EXP_NAME}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-48}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

MILESTONE_GAMMA="${MILESTONE_GAMMA:-0.99}"
MILESTONE_LAMBDA="${MILESTONE_LAMBDA:-0.95}"
MILESTONE_COST="${MILESTONE_COST:-0.05}"
GENERATOR_ENABLE="${GENERATOR_ENABLE:-true}"
GENERATOR_NUM_MILESTONES="${GENERATOR_NUM_MILESTONES:-5}"
MILESTONE_TEMPLATE="${MILESTONE_TEMPLATE:-alfworld}"
JUDGE_LLM_MODEL="${JUDGE_LLM_MODEL:-Qwen3-VL-32B-Instruct-FP8}"
JUDGE_LLM_URL_1="${JUDGE_LLM_URL_1:-http://127.0.0.1:8080/v1}"
JUDGE_LLM_URL_2="${JUDGE_LLM_URL_2:-http://127.0.0.1:8081/v1}"
JUDGE_LLM_BASE_URLS="${JUDGE_LLM_BASE_URLS:-[\"${JUDGE_LLM_URL_1}\",\"${JUDGE_LLM_URL_2}\"]}"

export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

release_activate_conda
release_prepare_runtime "${EXP_NAME}"
release_chat_template_args

python3 -m verl.trainer.main_ppo \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    algorithm.adv_estimator=milestone_gae \
    "algorithm.milestone_gae.gamma=${MILESTONE_GAMMA}" \
    "algorithm.milestone_gae.lam=${MILESTONE_LAMBDA}" \
    "algorithm.milestone_gae.cost=${MILESTONE_COST}" \
    "algorithm.milestone_gae.judge_llm.base_urls=${JUDGE_LLM_BASE_URLS}" \
    "algorithm.milestone_gae.judge_llm.model=${JUDGE_LLM_MODEL}" \
    algorithm.milestone_gae.judge_llm.temperature=0.6 \
    "algorithm.milestone_gae.generator.enable=${GENERATOR_ENABLE}" \
    "algorithm.milestone_gae.generator.num_milestones=${GENERATOR_NUM_MILESTONES}" \
    "algorithm.milestone_gae.generator.llm.base_urls=${JUDGE_LLM_BASE_URLS}" \
    "algorithm.milestone_gae.generator.llm.model=${JUDGE_LLM_MODEL}" \
    algorithm.milestone_gae.generator.llm.temperature=0.3 \
    "algorithm.milestone_gae.fallback_template=${MILESTONE_TEMPLATE}" \
    algorithm.trajectory_save.enable=true \
    "algorithm.trajectory_save.output_dir=${OUTPUT_DIR}" \
    "data.train_files=${DATA_DIR}/train.parquet" \
    "data.val_files=${DATA_DIR}/test.parquet" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.val_batch_size=${VAL_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    "actor_rollout_ref.rollout.name=${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    env.seed=42 \
    env.max_steps=30 \
    env.history_length=10 \
    "env.rollout.n=${GROUP_SIZE}" \
    "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER}" \
    "ray_init.num_cpus=${RAY_NUM_CPUS}" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.critic_warmup=0 \
    "trainer.logger=${TRAINER_LOGGER}" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXP_NAME}" \
    "trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE}" \
    "trainer.nnodes=${NNODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True \
    "$@"
