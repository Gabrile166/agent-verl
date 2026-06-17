set -x
ENGINE=${1:-vllm}
# Clean up any existing Ray processes first
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export ALFWORLD_DATA=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/.cache/alfworld
export PYTHONUTF8=1
source /mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/.bashrc
num_cpus_per_env_worker=0.15 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

BASE_PATH=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin

if [[ $MODEL_PATH == *"3B"* ]]; then
    MODEL_SIZE="3b"
    MODEL_FAMILY="qwen2.5"
elif [[ $MODEL_PATH == *"4B"* ]]; then
    MODEL_SIZE="4b"
    MODEL_FAMILY="qwen3"
elif [[ $MODEL_PATH == *"7B"* ]]; then
    MODEL_SIZE="7b"
    MODEL_FAMILY="qwen2.5"
else
    echo "Unknown model size in MODEL_PATH"
    exit 1
fi
EXP_NAME="alfworld_grpo_${MODEL_FAMILY}_${MODEL_SIZE}_seed_${SEED}_ood"


train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

export WANDB_API_KEY=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
export WANDB_MODE=offline
export WANDB_DIR=/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/wandb/${EXP_NAME}
mkdir -p ${WANDB_DIR}


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$BASE_PATH/data/verl-agent/text/train.parquet \
    data.val_files=$BASE_PATH/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
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
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
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
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.alfworld.eval_dataset='eval_out_of_distribution' \
    env.seed=$SEED \
    env.max_steps=50 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='qwen3' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True $@
