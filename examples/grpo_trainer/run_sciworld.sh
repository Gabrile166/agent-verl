# set -x
ENGINE=${1:-vllm}


# Clean up any existing Ray processes first
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2


export RAY_worker_register_timeout_seconds=600
export http_proxy="http://10.70.11.190:8412"
export https_proxy="http://10.70.11.190:8412"
export no_proxy="localhost,127.0.0.1,0.0.0.0"

export MY_TEMP_DIR="/workdir/temp_cache/${USER}/${EXP_NAME}"
mkdir -p $MY_TEMP_DIR

export RAY_TMPDIR="${MY_TEMP_DIR}/ray"
mkdir -p $RAY_TMPDIR

export TORCH_COMPILE_CACHE_DIR="${MY_TEMP_DIR}/torch_compile_cache"
export VLLM_CACHE_DIR="${MY_TEMP_DIR}/vllm_cache"
export TRITON_CACHE_DIR="${MY_TEMP_DIR}/triton_cache"
mkdir -p $TORCH_COMPILE_CACHE_DIR $VLLM_CACHE_DIR $TRITON_CACHE_DIR

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ALFWORLD_DATA=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/.cache/alfworld

# Ray configuration to prevent worker explosion
export RAY_DEDUP_LOGS=0


source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/setupconda.sh
conda activate verl-agent
export PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/envs/verl-agent/bin:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/condabin:$PATH"

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/common/HF_MODELS/Qwen2.5-7B-Instruct
BASE_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin


train_data_size=16
val_data_size=128
group_size=8

# # We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    env.env_name=sciworld/ScienceWorldEnv \
    env.sciworld.generalization_level=1 \
    env.sciworld.env_step_limit=50 \
    env.seed=0 \
    env.max_steps=30 \
    env.history_length=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ray_init.num_cpus=128 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='new_max_50steps_ood' \
    trainer.experiment_name='mt_sci_grpo_qwen2.5b_7b_ood' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True $@