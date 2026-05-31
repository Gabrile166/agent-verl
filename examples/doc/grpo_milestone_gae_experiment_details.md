# GRPO 与 Milestone-GAE 实验设置说明

本文根据当前仓库中的训练脚本、Hydra 默认配置、数据预处理逻辑、环境封装、奖励函数与验证流程整理，作为论文实验部分的详细设置记录。核心脚本位于：

- `examples/grpo_trainer/run_alfworld.sh`
- `examples/grpo_trainer/run_sciworld.sh`
- `examples/milestone_gae_trainer/run_alfworld.sh`
- `examples/milestone_gae_trainer/run_sciworld.sh`

## 1. 实验任务与模型

本文比较 baseline GRPO 与 Milestone-GAE 两类训练算法，在两个文本交互环境上训练 Qwen2.5 系列 instruction model：

| 环境 | 环境名 | 任务类型 | 主要评估集 |
|---|---|---|---|
| ALFWorld | `alfworld/AlfredTWEnv` | 基于 ALFRED 的家居文本交互任务，包括取放、加热、清洗、冷却、开关、检查等任务类型 | `eval_out_of_distribution`，即 ALFWorld `valid_unseen` |
| SciWorld | `sciworld/ScienceWorldEnv` | 小学科学场景中的文本交互任务，包含探索、移动、拾取、放置、读写、连接设备、实验操作等 | generalization level 1 的 `test` variations |

策略模型使用 Qwen2.5-3B-Instruct 与 Qwen2.5-7B-Instruct 两种规模。当前脚本中的 `MODEL_PATH` 默认指向 `Qwen2.5-7B-Instruct`；3B 实验使用相同训练配置，仅将 `actor_rollout_ref.model.path` 或脚本中的 `MODEL_PATH` 替换为 `Qwen2.5-3B-Instruct`。

Milestone-GAE 额外使用一个 OpenAI-compatible API 形式部署的 judge/generator 模型：

| 组件 | 默认模型 | 默认地址 | 作用 |
|---|---|---|---|
| Milestone Judge | `Qwen3-VL-32B-Instruct-FP8` | `http://127.0.0.1:8080/v1`, `http://127.0.0.1:8081/v1` | 对完整轨迹逐步判定 milestone potential `phi(s_t)` |
| Milestone Generator | `Qwen3-VL-32B-Instruct-FP8` | 同上 | 根据任务描述和 expert trajectory 生成 task-specific milestones |

## 2. 数据来源与数据内容

训练脚本中的 `data.train_files` 与 `data.val_files` 均指向：

```text
$BASE_PATH/data/verl-agent/text/train.parquet
$BASE_PATH/data/verl-agent/text/test.parquet
```

这两个 parquet 文件由 `examples/data_preprocess/prepare.py` 生成。代码中加载了 `hiyouga/geometry3k`，但注释明确说明并不使用 Geometry3K 的真实题目内容，而只是用它来确定数据模态和样本数量。对于 `--mode text`，每条 parquet 样本主要包含：

- `data_source = "text"`
- `prompt = [{"role": "user", "content": ""}]`
- `ability = "agent"`
- `extra_info = {"split": ..., "index": ...}`

因此，本实验的真实任务并不来自 parquet 中的题目文本，而来自环境 reset 后采样到的 ALFWorld / SciWorld 任务实例。parquet 数据在这里的作用是提供固定数量的 rollout/query 槽位，并通过 `train_batch_size`、`val_batch_size` 控制每轮训练和验证启动多少个并行环境。

### 2.1 ALFWorld 数据

ALFWorld 数据路径由环境变量 `ALFWORLD_DATA` 指定。`agent_system/environments/env_package/alfworld/configs/config_tw.yaml` 中的数据划分为：

| 划分 | 配置路径 | 用途 |
|---|---|---|
| train | `$ALFWORLD_DATA/json_2.1.1/train` | 训练环境采样 |
| valid_seen | `$ALFWORLD_DATA/json_2.1.1/valid_seen` | in-distribution 验证，可选 |
| valid_unseen | `$ALFWORLD_DATA/json_2.1.1/valid_unseen` | out-of-distribution 验证 |

本实验脚本设置 `env.alfworld.eval_dataset='eval_out_of_distribution'`，因此验证分数来自 ALFWorld `valid_unseen`。`num_train_games=-1` 与 `num_eval_games=-1` 表示不额外截断环境内的数据集，使用对应 split 的完整可用任务。

### 2.2 SciWorld 数据

SciWorld 的任务实例来自 `agent_system/environments/env_package/sciworld/variations_idx/L1_idx.json`。脚本设置：

```text
env.sciworld.generalization_level=1
```

因此训练使用 `L1_idx.json` 中的 `train` variations，验证使用其中的 `test` variations。当前文件中数量为：

| SciWorld split | 数量 |
|---|---:|
| L1 train variations | 3322 |
| L1 test variations | 1684 |

环境 worker 在 reset 时从对应 variations 中随机抽取一个 `(task_id, variation_id)`。训练 worker 使用 `env.seed`，验证 worker 使用 `env.seed + 1000`。

## 3. 每轮训练与验证的数据规模

代码中真正的组采样由 `env.rollout.n` 完成，而不是由 `actor_rollout_ref.rollout.n` 完成。脚本统一设置：

```text
train_data_size = 16
group_size = 8
env.rollout.n = 8
actor_rollout_ref.rollout.n = 1  # 默认值
```

因此，每个训练 step 中有 16 个 query/group，每个 group 并行采样 8 条 policy trajectory，总计：

```text
16 groups * 8 rollouts = 128 policy trajectories / training step
```

由于 `train.parquet` 也只有 16 条样本且 `data.train_batch_size=16`，每个 epoch 通常对应 1 个 PPO training step；脚本设置 `trainer.total_epochs=100`，所以单个 seed 通常进行 100 个训练 step，约 12,800 条 policy trajectory rollout。

验证阶段不使用 group sampling，`is_train=False` 时不会按 `env.rollout.n` 复制。验证轨迹数由 `data.val_batch_size` 决定：

| 方法/环境 | `val_data_size` | 每次验证 policy trajectory 数 |
|---|---:|---:|
| GRPO ALFWorld | 128 | 128 |
| GRPO SciWorld | 128 | 128 |
| Milestone-GAE ALFWorld | 64 | 64 |
| Milestone-GAE SciWorld | 128 | 128 |

Milestone-GAE SciWorld 脚本额外设置 `algorithm.expert.enable=true`，采用 N+1 worker 架构：每个 group 除 8 个 policy worker 外，再启动 1 个 expert worker，用于采集 expert trajectory。此时每个训练 step 的物理 worker 数约为 `16 * (8 + 1) = 144`，但参与策略更新的仍是 128 条 policy trajectory。当前 ALFWorld Milestone-GAE 脚本启用了 generator 和 `fallback_template=alfworld`，但没有显式设置 `algorithm.expert.enable=true`；因此如果运行时没有额外覆盖该配置，generator 在缺少 expert trajectory 时会使用默认/fallback milestones。

## 4. 验证指标口径

环境奖励是二值成功奖励。`agent_system/environments/reward_utils.py` 中定义：

```text
reward = 10.0 * float(info["won"])
```

其中 `info["won"]` 由环境判断任务是否完成。验证中的成功率来自 `EnvironmentManagerBase.success_evaluator()`，它取每条 episode 最后一个有效 step 的 `info["won"]`，并统计均值：

```text
val/success_rate = mean(float(info["won"]))
```

论文表格中的 `val` 使用百分制成功率，即 `100 * val/success_rate`。代码中还会通过 `EpisodeRewardManager` 把 episode reward 写入 response 最后一个有效 token；由于 rollout 结束后 batch 会按环境 step 展平，`val/text/test_score` 属于 reward tensor 统计，可能受到 episode 长度的 step-level 展平权重影响。因此论文表格不采用 `val/text/test_score`，而采用环境 episode 粒度的百分制成功率。

`step_length` 表示每条 episode 的平均交互步数，对应 rollout 中累计的 `episode_lengths`。训练日志中对应 `episode/length/mean`；论文表格中的 `step_length` 与该定义一致，表示完成或截断前平均执行了多少个环境 action。

## 5. 共同训练超参数

除 advantage estimator 和少量环境长度设置外，GRPO 与 Milestone-GAE 共享以下主要 RL 训练配置：

| 参数 | 设置 |
|---|---|
| 训练入口 | `python3 -m verl.trainer.main_ppo` |
| train batch size | `data.train_batch_size=16` |
| rollout group size | `env.rollout.n=8` |
| rollout engine | `vllm` |
| 节点/GPU | `trainer.nnodes=1`, `trainer.n_gpus_per_node=8` |
| data filtering | `filter_overlong_prompts=True` |
| raw chat | `return_raw_chat=True` |
| actor optimizer lr | `1e-6` |
| PPO mini batch | `actor_rollout_ref.actor.ppo_mini_batch_size=256` |
| PPO micro batch / GPU | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32` |
| PPO epoch | 默认 `1` |
| clip ratio | 默认 `0.2` |
| entropy coeff | 默认 `0.001` |
| grad clip | 默认 `1.0` |
| actor KL loss | `use_kl_loss=True` |
| KL loss coef/type | `0.01`, `low_var_kl` |
| KL in reward | `algorithm.use_kl_in_reward=False` |
| invalid action penalty | `use_invalid_action_penalty=True`, coef `0.1` |
| gradient checkpointing | `True` |
| remove padding | `True` |
| actor FSDP offload | `param_offload=False`, `optimizer_offload=False` |
| reference policy offload | `ref.fsdp_config.param_offload=True` |
| tensor model parallel | `1` |
| save/test frequency | `save_freq=10`, `test_freq=5` |
| validation before training | `True` |
| total epochs | `100` |

采样与序列长度：

| 环境/方法 | max prompt | max response | train truncation | validation sampling |
|---|---:|---:|---|---|
| GRPO ALFWorld | 2048 | 1024 | `error` | `temperature=0.4`, `do_sample=True` |
| GRPO SciWorld | 4096 | 2048 | `error` | `temperature=0.4`, `do_sample=True` |
| Milestone-GAE ALFWorld | 4096 | 2048 | `error` | `temperature=0.4`, `do_sample=True` |
| Milestone-GAE SciWorld | 4096 | 2048 | `error` | `temperature=0.4`, `do_sample=True` |

训练 rollout 的默认采样温度来自 `ppo_trainer.yaml`，为 `temperature=1.0`, `top_p=1`, `top_k=-1`, `do_sample=True`。

## 6. GRPO 设置

GRPO baseline 通过以下配置启用：

```text
algorithm.adv_estimator=grpo
```

实现入口是 `verl/trainer/ppo/core_algos.py` 中的 `compute_grpo_outcome_advantage()`。对于同一 prompt group `uid` 下的多条 trajectory，GRPO 使用 group-relative outcome reward 计算 advantage。当前实现中，轨迹完成后每个环境 step 都携带同一条 trajectory 的 episode reward，因此同一条 trajectory 内所有 step 的 outcome score 相同。

GRPO 的关键设置：

| 环境 | seed | max steps | history length | 环境验证集 | 额外环境设置 |
|---|---:|---:|---:|---|---|
| ALFWorld | 0 | 30 | 10 | `eval_out_of_distribution` | `ray_init.num_cpus=96` |
| SciWorld | 0 | 30 | 10 | L1 `test` variations | `env_step_limit=50`, `ray_init.num_cpus=128` |

## 7. Milestone-GAE 设置

Milestone-GAE 通过以下配置启用：

```text
algorithm.adv_estimator=milestone_gae
algorithm.milestone_gae.gamma=0.99
algorithm.milestone_gae.lam=0.95
algorithm.milestone_gae.cost=0.05
```

Milestone-GAE 脚本还设置了 `algorithm.gamma=0.95`，但在 `adv_estimator=milestone_gae` 路径中，真正用于 Milestone-GAE TD/GAE 递推的是 `algorithm.milestone_gae.gamma=0.99` 与 `algorithm.milestone_gae.lam=0.95`。

实现入口是 `rlvmr/core_milestone_gae.py`。其核心思想是由 LLM Judge 对每条 policy trajectory 的每个环境 step 给出 milestone potential：

```text
phi(s_t) in [0, 1]
```

然后用该 potential 替代传统 critic value，构造：

```text
delta_t = r_t - cost + gamma * phi(s_{t+1}) - phi(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

其中 episode reward 被压缩到最后一个环境 step，失败或截断轨迹的终止 potential 保持为最后一步的 `phi(s_T)`，成功轨迹的终止 potential 设为 `1.0`。计算出的 step-level advantage 会扩展到该 step response 的所有有效 token，并按 batch 中有效 step 做标准化。

Milestone-GAE 的 judge/generator 设置：

| 环境 | Judge temp | Generator temp | generator | num milestones | fallback |
|---|---:|---:|---|---:|---|
| ALFWorld | 0.6 | 0.3 | enabled | 5 | `alfworld` |
| SciWorld | 0.6 | 0.6 | enabled | 5 | `none` |

Milestone-GAE 的环境设置：

| 环境 | seed | max steps | history length | 环境验证集 | 额外环境设置 |
|---|---:|---:|---:|---|---|
| ALFWorld | 42 | 30 | 10 | `eval_out_of_distribution` | `ray_init.num_cpus=48` |
| SciWorld | 64 | 30 | 2 | L1 `test` variations | `env_step_limit=100`, `algorithm.expert.enable=true`, `ray_init.num_cpus=96` |

## 8. 主实验结果

ALFWorld 结果为 3 个 seed 训练后的均值。`val` 为百分制成功率，`step_length` 为平均 episode 步长。

### 8.1 ALFWorld

| 方法 | Qwen2.5 规模 | val | step_length |
|---|---:|---:|---:|
| GRPO baseline | 3B | 53.9 | 17.36 |
| GRPO baseline | 7B | 68.7 | 14.30 |
| Milestone-GAE | 3B | 60.5 | 14.06 |
| Milestone-GAE | 7B | 72.1 | 10.71 |

相对 GRPO，Milestone-GAE 在 ALFWorld 上：

- 3B：成功率提升 `+6.6`，平均步长减少 `3.30`。
- 7B：成功率提升 `+3.4`，平均步长减少 `3.59`。

### 8.2 SciWorld

| 方法 | Qwen2.5 规模 | val | step_length |
|---|---:|---:|---:|
| GRPO baseline | 3B | 51.56 | 17.72 |
| GRPO baseline | 7B | 66.40 | 19.52 |
| Milestone-GAE | 3B | 55.39 | 12.79 |
| Milestone-GAE | 7B | 74.73 | 10.66 |

相对 GRPO，Milestone-GAE 在 SciWorld 上：

- 3B：成功率提升 `+3.83`，平均步长减少 `4.93`。
- 7B：成功率提升 `+8.33`，平均步长减少 `8.86`。

## 9. 可复现实验命令模板

7B 默认可直接运行对应脚本：

```bash
bash examples/grpo_trainer/run_alfworld.sh
bash examples/grpo_trainer/run_sciworld.sh
bash examples/milestone_gae_trainer/run_alfworld.sh
bash examples/milestone_gae_trainer/run_sciworld.sh
```

3B 实验使用同一脚本和同一组超参数，只替换策略模型路径，例如：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  actor_rollout_ref.model.path=/path/to/Qwen2.5-3B-Instruct \
  trainer.experiment_name=grpo_qwen2.5_3b_alfworld

bash examples/milestone_gae_trainer/run_sciworld.sh \
  actor_rollout_ref.model.path=/path/to/Qwen2.5-3B-Instruct \
  trainer.experiment_name=milestone_gae_qwen2.5_3b_sciworld
```

对于 Milestone-GAE，需要先启动 judge/generator 模型服务，并保证脚本中的 `JUDGE_LLM_URL_1`、`JUDGE_LLM_URL_2` 可访问。

## 10. 论文写法建议

实验部分可以概括为：本实验使用 parquet 文件仅作为 agent rollout 的固定 query 槽位，真实任务实例由环境按训练/验证 split 动态采样。GRPO 与 Milestone-GAE 共用相同 PPO actor 更新、KL 正则、invalid-action penalty、batch size 和 rollout group size；区别在于 GRPO 使用同组 trajectory 的 outcome reward 标准化构造 advantage，而 Milestone-GAE 使用 LLM judge 给出的 step-level milestone potential 构造 GAE advantage，从而提供更细粒度的过程信用分配。验证分数统一来自环境 `info["won"]` 的成功率，ALFWorld 在 `valid_unseen` 上评估，SciWorld 在 L1 generalization 的 test variations 上评估。
