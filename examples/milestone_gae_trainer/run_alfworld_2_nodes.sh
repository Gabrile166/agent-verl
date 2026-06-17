#!/bin/bash
# Milestone-Guided GAE Training Script for ALFWorld
# Two-node training version: 2 nodes x 4 GPUs per node
# Debug/diagnostic edition

set -euo pipefail
set -x

ENGINE=${1:-vllm}

echo "[BOOT] Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ALFWORLD_DATA=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/.cache/alfworld
export RAY_DEDUP_LOGS=0
export PYTHONUTF8=1
source /mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/.bashrc

num_cpus_per_env_worker=0.15
BASE_PATH=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin

if [[ $MODEL_PATH == *"3B"* ]]; then
    MODEL_SIZE="3b"
elif [[ $MODEL_PATH == *"7B"* ]]; then
    MODEL_SIZE="7b"
elif [[ $MODEL_PATH == *"14B"* ]]; then
    MODEL_SIZE="14b"
else
    echo "[FATAL] Unknown model size in MODEL_PATH"
    exit 1
fi

EXP_NAME="alfworld_milestone_gae_qwen2.5_${MODEL_SIZE}_seed_${SEED}_ood"

train_data_size=16
val_data_size=128
group_size=8

MILESTONE_GAMMA=0.99
MILESTONE_LAMBDA=0.95
MILESTONE_COST=0.05

JUDGE_LLM_URL_1=${JUDGE_LLM_URL_1:-"http://10.102.208.26:8081/v1"}
JUDGE_LLM_URL_2=${JUDGE_LLM_URL_2:-"http://10.102.241.20:8082/v1"}
JUDGE_LLM_MODEL=${JUDGE_LLM_MODEL:-"MiMo-VL-7B-RL-2508"}

GENERATOR_ENABLE=${GENERATOR_ENABLE:-true}
GENERATOR_NUM_MILESTONES=${GENERATOR_NUM_MILESTONES:-5}
MILESTONE_TEMPLATE="alfworld"

export WANDB_API_KEY=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
export WANDB_MODE=offline
export WANDB_DIR=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/wandb/${EXP_NAME}
mkdir -p "${WANDB_DIR}"

SUCCESS_REWARD=${ENV_REWARD:-10}

TARGET_FILE="/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/addr_${EXP_NAME}.txt"
ABORT_FILE="/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/agent-verl/.alfworld_abort_${EXP_NAME}"

RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}
MASTER_ADDR=${MASTER_ADDR:-}

# ===== Diagnostic env =====
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

echo "[ENV] hostname=$(hostname)"
echo "[ENV] date=$(date)"
echo "[ENV] RANK=${RANK}"
echo "[ENV] MASTER_ADDR=${MASTER_ADDR}"
echo "[ENV] MASTER_PORT=${MASTER_PORT}"
echo "[ENV] EXP_NAME=${EXP_NAME}"
echo "[ENV] MODEL_PATH=${MODEL_PATH}"
echo "[ENV] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "[ENV] NCCL_DEBUG=${NCCL_DEBUG}"
echo "[ENV] NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "[ENV] TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE}"
echo "[ENV] NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
echo "[ENV] TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"

echo "[DIAG] ===== Network / IB info on $(hostname) ====="
(ip -brief addr || true)
echo "[DIAG] ----- ibdev2netdev -----"
(ibdev2netdev || true)
echo "[DIAG] ----- ibstat -----"
(ibstat || true)
echo "[DIAG] ----- ibv_devinfo -----"
(ibv_devinfo | egrep 'hca_id|phys_port_cnt|port:|state:|link_layer:|gid' || true)
echo "[DIAG] ----- nvidia-smi topo -m -----"
(nvidia-smi topo -m || true)
echo "[DIAG] ----- env grep NCCL -----"
(env | grep -E '^(NCCL|TORCH_NCCL|TORCH_DISTRIBUTED)' | sort || true)

if [ "$RANK" -eq 0 ]; then
    echo "[HEAD] Starting head node on port ${MASTER_PORT}..."

    if [ -z "$MASTER_ADDR" ]; then
        echo "[HEAD][FATAL] MASTER_ADDR is empty."
        echo "head_failed_empty_master_addr at $(date)" > "$ABORT_FILE"
        exit 1
    fi

    rm -f "$TARGET_FILE" "$ABORT_FILE"
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    ray start \
        --head \
        --node-ip-address="$MASTER_ADDR" \
        --port="$MASTER_PORT" \
        --num-gpus 4 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --disable-usage-stats \
        --block &
    RAY_HEAD_PID=$!

    echo "[HEAD] Waiting briefly for Ray head..."
    sleep 30
    echo "[HEAD] ray status after head startup:"
    ray status || true

    echo "[HEAD] Launching trainer..."

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=milestone_gae \
        algorithm.milestone_gae.gamma=$MILESTONE_GAMMA \
        algorithm.milestone_gae.lam=$MILESTONE_LAMBDA \
        algorithm.milestone_gae.cost=$MILESTONE_COST \
        'algorithm.milestone_gae.judge_llm.base_urls=["'$JUDGE_LLM_URL_1'","'$JUDGE_LLM_URL_2'"]' \
        algorithm.milestone_gae.judge_llm.model=$JUDGE_LLM_MODEL \
        algorithm.milestone_gae.judge_llm.temperature=0.1 \
        algorithm.milestone_gae.generator.enable=$GENERATOR_ENABLE \
        algorithm.milestone_gae.generator.num_milestones=$GENERATOR_NUM_MILESTONES \
        'algorithm.milestone_gae.generator.llm.base_urls=["'$JUDGE_LLM_URL_1'","'$JUDGE_LLM_URL_2'"]' \
        algorithm.milestone_gae.generator.llm.model=$JUDGE_LLM_MODEL \
        algorithm.milestone_gae.generator.llm.temperature=0.3 \
        algorithm.milestone_gae.fallback_template=$MILESTONE_TEMPLATE \
        algorithm.expert.enable=true \
        algorithm.trajectory_save.enable=true \
        algorithm.trajectory_save.output_dir=$BASE_PATH/agent-verl/output/$EXP_NAME \
        data.train_files=$BASE_PATH/data/verl-agent/text/train.parquet \
        data.val_files=$BASE_PATH/data/verl-agent/text/test.parquet \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=4096 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.95 \
        env.env_name=alfworld/AlfredTWEnv \
        env.alfworld.eval_dataset='eval_out_of_distribution' \
        env.seed=$SEED \
        env.max_steps=30 \
        env.history_length=2 \
        env.rollout.n=$group_size \
        env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
        env.success_reward=$SUCCESS_REWARD \
        trainer.ray_wait_register_center_timeout=600 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='milestone_gae_alfworld' \
        trainer.experiment_name=$EXP_NAME \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=2 \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.total_epochs=100 \
        trainer.val_before_train=True $@

else
    echo "[WORKER] Waiting for head node address..."
    sleep 5

    while true; do
        if [ -f "$ABORT_FILE" ]; then
            echo "[WORKER] Head aborted launch. Worker exits."
            exit 0
        fi

        if [ -r "$TARGET_FILE" ] && [ -n "$(cat "$TARGET_FILE")" ]; then
            break
        fi

        echo "[WORKER] Waiting for valid master address file..."
        sleep 2
    done

    MASTER_ADDR=$(cat "$TARGET_FILE")
    echo "[WORKER] Detected master address: $MASTER_ADDR"

    ray start \
        --address "${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus 4 \
        --block &
    RAY_WORKER_PID=$!

    echo "[WORKER] Waiting briefly for Ray worker registration..."
    sleep 15
    echo "[WORKER] ray status after join:"
    ray status || true

    while true; do
        if [ -f "$ABORT_FILE" ]; then
            echo "[WORKER] Head aborted after startup. Worker exits."
            exit 0
        fi

        status=$(ray status 2>&1 || true)
        echo "[WORKER] ray status heartbeat at $(date)"
        echo "$status" | head -n 30

        if echo "$status" | grep -q "Active:"; then
            echo "[WORKER] Cluster still alive. Sleeping 600s..."
            sleep 600
        else
            echo "[WORKER] No active nodes found. Exiting..."
            exit 0
        fi
    done
fi