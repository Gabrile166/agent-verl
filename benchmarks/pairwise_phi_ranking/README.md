# Pairwise Potential Ranking Benchmark 使用说明

本文档说明 `agent-verl/benchmarks/pairwise_phi_ranking/` 下两套轨迹对生成框架的使用方法：

- `SciWorld` 轨迹对生成：`build/build_expert_fork.py`
- `ALFWorld` 轨迹对生成：`build/build_expert_fork_alfworld.py`

这两个脚本的共同目标，是构造用于 pairwise ranking / preference learning 的 A/B 轨迹对样本。每个样本都包含：

- 同一个任务下的两条候选轨迹 `trajectory_a` 和 `trajectory_b`
- 一个监督标签 `label`，表示哪条轨迹更优
- 两条轨迹各自的进度分数 `progress_scalar_a` / `progress_scalar_b`
- 样本难度 `difficulty`
- 轨迹对类型 `pair_type`

生成后的数据默认保存为 JSONL，每行一个 `BenchmarkSample`。

---

## 1. 目录结构

当前目录中与生成/评测最相关的部分如下：

- `build/`
  - `build_expert_fork.py`：SciWorld 轨迹对生成脚本
  - `build_expert_fork_alfworld.py`：ALFWorld 轨迹对生成脚本
  - `samples_demo.jsonl`：示例输出
- `core/`
  - `schema.py`：样本数据结构定义
  - `label_rules.py`：标签规则
  - `filters.py`：difficulty、A/B 随机交换、subset 检测等逻辑
- `eval/`
  - 评测 prompt 构造、结果解析、指标统计

---

## 2. 输出样本格式

两个生成脚本输出的都是统一的 `BenchmarkSample` 结构，核心字段包括：

- `sample_id`：样本唯一 ID
- `task_description`：任务描述
- `trajectory_a` / `trajectory_b`：两条候选轨迹
- `label`：`A` 或 `B`
- `progress_scalar_a` / `progress_scalar_b`：两条轨迹的进度分数
- `progress_gap`：两者差值绝对值
- `difficulty`：`easy` / `medium` / `hard`
- `task_type`：任务类型
- `track`：环境标识，例如 `sciworld_exact`、`alfworld_tw`
- `pair_type`：轨迹对类型
- `is_subset_pair`：是否存在 prefix/subset 关系
- `uses_expert_branch`：是否使用 expert 分支

其中每条 trajectory 的结构是：

- `steps`：按时间顺序排列的状态转移序列
- `task_description`：任务描述
- `truncated_at_step`：轨迹截断长度
- `is_completed`：是否完成任务

每个 `step` 包含：

- `obs_before`
- `action`
- `obs_after`

---

## 3. SciWorld 生成框架

对应脚本：`build/build_expert_fork.py`

### 3.1 核心思路

SciWorld 版本采用的是 fork-based 构造方式：

1. 先读取环境的 gold path
2. 用前 `k` 步 gold action 回放到 fork 点
3. 从 fork 点向后分叉出两条轨迹
4. 比较两条轨迹的最终 progress，生成监督标签

这个脚本当前主要支持两类分支来源：

- `gold/expert` continuation：继续沿 gold path 往后走
- `model/random/heuristic` continuation：从 fork 点后自行生成动作

### 3.2 当前支持的 pair 类型

SciWorld builder 实际会生成以下两类 fork pair：

- `expert_fork_model`
  - A 分支通常来自模型 / random / heuristic
  - B 分支来自 gold continuation
- `expert_fork_expert`
  - 两个分支都来自 expert/gold
  - 其中一条只走更短的 gold 后缀，用于构造 expert-vs-expert 的进度差

说明：

- 当 `--mix-expert-ratio > 0` 时，会以一定概率生成 `expert_fork_expert`
- 否则主要生成 `expert_fork_model`

### 3.3 进度评分方式

SciWorld 使用环境自带分数：

- `raw_score`
- `progress_scalar = raw_score / 100.0`

标签规则：

- 比较 `progress_scalar_a` 和 `progress_scalar_b`
- 若两者相等，或差值小于过滤阈值，则样本会被过滤

### 3.4 命令行参数

`build_expert_fork.py` 主要参数如下：

- `--action-source`
  - 可选：`openai`、`random`、`heuristic`
  - 含义：fork 后的非 expert 分支如何生成动作
- `--base-url`
  - OpenAI-compatible 接口地址
- `--model`
  - 模型名
- `--api-key`
  - API key
- `--temperature`
  - 采样温度
- `--top-p`
  - nucleus sampling 参数
- `--top-k`
  - top-k 参数
- `--num-tasks`
  - 选取多少个 SciWorld task
- `--fork-depth-ratio`
  - fork 点深度占 gold path 长度的比例
- `--num-branch-steps`
  - fork 后每条分支最多走多少步
- `--variations-per-task`
  - 每个 task 采样多少个 variation
- `--output`
  - 输出 JSONL 文件路径
- `--mix-expert-ratio`
  - Branch A 退化为 expert shorter branch 的概率
- `--simplification`
  - SciWorld simplification preset，默认 `easy`
- `--seed`
  - 随机种子

### 3.5 推荐用法

#### 方式一：随机动作快速验证

适合做流程联调，确认脚本能跑通。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source random \
  --num-tasks 3 \
  --variations-per-task 1 \
  --output /tmp/sciworld_fork_random.jsonl
```

#### 方式二：启发式动作生成

适合做比 random 更“像样”的错误分支测试。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source heuristic \
  --num-tasks 3 \
  --variations-per-task 2 \
  --output /tmp/sciworld_fork_heuristic.jsonl
```

#### 方式三：接入 OpenAI-compatible 模型

适合正式生成 ranking 数据。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen2.5-7B-Instruct \
  --temperature 1.0 \
  --top-p 0.95 \
  --num-tasks 5 \
  --variations-per-task 2 \
  --fork-depth-ratio 0.4 \
  --num-branch-steps 8 \
  --output /tmp/sciworld_fork_openai.jsonl
```

#### 方式四：混入 expert-vs-expert 样本

如果你希望数据里既有 `expert_fork_model`，也有一部分 `expert_fork_expert`：

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen2.5-7B-Instruct \
  --mix-expert-ratio 0.3 \
  --num-tasks 5 \
  --variations-per-task 2 \
  --output /tmp/sciworld_fork_mixed.jsonl
```

这表示约 30% 的样本会尝试构造成 expert-vs-expert。

### 3.6 参数建议

SciWorld 上常用经验：

- `--fork-depth-ratio`
  - 推荐从 `0.3 ~ 0.5` 开始
  - 太小：前缀太短，轨迹差异大但噪声高
  - 太大：接近任务结尾，容易 tie
- `--num-branch-steps`
  - 推荐 `6 ~ 10`
  - 太短：难以拉开 progress gap
  - 太长：模型更容易彻底跑偏，但运行时间更长
- `--temperature`
  - 冷启动建议 `0.8 ~ 1.2`

### 3.7 SciWorld 常见问题

#### 1）为什么很多样本被过滤？

常见原因：

- A/B 两条轨迹 progress 太接近
- fork 点选得太晚
- branch steps 太短
- 模型和 gold 在后半段表现过于相似

可尝试：

- 降低 `--fork-depth-ratio`
- 增大 `--num-branch-steps`
- 提高 `--temperature`

#### 2）`heuristic` 有什么用？

`heuristic` 会尽量选“看起来合理但不完全跟 gold 一致”的动作，适合：

- 不接模型时做中间验证
- 观察数据分布
- 构造比 random 更稳定的错误轨迹

---

## 4. ALFWorld 生成框架

对应脚本：`build/build_expert_fork_alfworld.py`

这是当前重点使用的 ALFWorld 版本，已经改造成在线 rollout 模式。

### 4.1 核心思路

ALFWorld 版本与 SciWorld 不同，重点在于：

- 分支动作不是一次性预生成，而是逐步在线 rollout
- 每一步都会读取当前 observation、历史和 admissible commands
- 规则式 progress 不再使用旧的 `expert_prefix_depth`
- 现在直接基于 handcoded expert 的子目标完成度来判断进度

### 4.2 当前支持的 pair 模式

当前只支持你指定的两种：

#### 1）`expert_vs_model`

- Branch A：expert continuation
- Branch B：model online rollout

作用：

- 最容易构造
- 冷启动最稳定
- 适合作为第一批 ranking 数据来源

#### 2）`model_vs_model_same_model`

- Branch A/B：都来自同一个模型
- 通过不同 `seed`、`temperature`、`base-url` 来制造采样差异

作用：

- 更贴近真实 ranking 数据分布
- 不依赖两个不同模型
- 适合在已有模型服务后进一步扩大数据多样性

### 4.3 规则式 progress 评分

ALFWorld 当前使用 `AlfworldRuleProgressSolver`：

1. 读取对应游戏的 `task_type` 和 `pddl_params`
2. 初始化 handcoded expert policy
3. 每走一步，用当前 `obs`、`facts`、`admissible_commands` 更新 policy 状态
4. 调用 `check_subgoal_completion(...)` 判断当前已经完成到哪个 subgoal
5. 将进度映射为：

```text
progress = subgoal_idx / num_subgoals
```

如果任务完成：

```text
progress = 1.0
```

这个评分方式的好处是：

- 与任务结构强相关
- 比单纯用步数或 expert-prefix 更合理
- 对 pick/place、toggle、heat/cool/clean 等任务都更一致

### 4.4 在线 rollout 过程

ALFWorld builder 的每个分支都遵循以下流程：

1. 先用 expert replay 到 fork 点
2. 从 fork 点开始，每一步在线执行：
   - 读取当前 `obs`
   - 读取 `admissible_commands`
   - 把最近历史拼成 prompt
   - 调用模型得到下一步动作
   - 执行 `env.step([action])`
3. 每一步都更新 progress solver
4. branch 结束后记录最终 `progress_score`
5. A/B 比较生成标签

### 4.5 命令行参数

`build_expert_fork_alfworld.py` 主要参数如下：

- `--pair-mode`
  - 可选：`expert_vs_model`、`model_vs_model_same_model`
- `--action-source`
  - 可选：`openai`、`random`
- `--base-url`
  - 默认模型服务地址
- `--base-url-a` / `--base-url-b`
  - A/B 分支各自的服务地址
  - 若不指定，则继承 `--base-url`
- `--model`
  - 模型名
- `--api-key`
  - API key
- `--temperature`
  - 默认采样温度
- `--temperature-a` / `--temperature-b`
  - A/B 分支各自的采样温度
- `--seed`
  - 默认随机种子
- `--seed-a` / `--seed-b`
  - A/B 分支各自的 seed
- `--top-p`
  - top-p 采样参数
- `--top-k`
  - top-k 采样参数
- `--max-tokens`
  - 每一步动作生成的最大 token 数
- `--num-games`
  - 处理多少个 ALFWorld game
- `--fork-depth-ratio`
  - fork 点深度比例
- `--num-branch-steps`
  - fork 后每个分支最多 rollout 多少步
- `--config`
  - ALFWorld config YAML 路径；默认自动发现
- `--output`
  - 输出 JSONL 路径

### 4.6 推荐用法

#### 方式一：随机动作联调 `expert_vs_model`

适合验证环境、规则评分和在线 rollout 路径。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode expert_vs_model \
  --action-source random \
  --num-games 3 \
  --output /tmp/alfworld_expert_vs_model_random.jsonl
```

#### 方式二：正式生成 `expert_vs_model`

这是 ALFWorld 冷启动最推荐的方式。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode expert_vs_model \
  --action-source openai \
  --base-url-b http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --temperature-b 1.0 \
  --seed-b 43 \
  --num-games 20 \
  --fork-depth-ratio 0.4 \
  --num-branch-steps 6 \
  --output /tmp/alfworld_expert_vs_model.jsonl
```

说明：

- 这个模式下 Branch A 由 expert 自动接管
- Branch B 使用模型在线 rollout
- 如果只配了 `--base-url-b`，A 分支不会访问模型

#### 方式三：正式生成 `model_vs_model_same_model`

用于同一个模型生成两条不同采样轨迹。

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode model_vs_model_same_model \
  --action-source openai \
  --base-url-a http://127.0.0.1:8000/v1 \
  --base-url-b http://127.0.0.1:8001/v1 \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --temperature-a 0.8 \
  --temperature-b 1.1 \
  --seed-a 101 \
  --seed-b 202 \
  --num-games 20 \
  --fork-depth-ratio 0.4 \
  --num-branch-steps 6 \
  --output /tmp/alfworld_model_vs_model_same_model.jsonl
```

如果你只有一个模型，也可以这样配置：

- `--base-url-a` 和 `--base-url-b` 指向同一个服务
- 通过不同 `seed` / `temperature` 制造差异

例如：

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode model_vs_model_same_model \
  --action-source openai \
  --base-url-a http://127.0.0.1:8000/v1 \
  --base-url-b http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --temperature-a 0.8 \
  --temperature-b 1.1 \
  --seed-a 101 \
  --seed-b 202 \
  --num-games 20 \
  --output /tmp/alfworld_model_vs_model_same_model_single_server.jsonl
```

### 4.7 参数建议

ALFWorld 上建议优先这样调：

- `--pair-mode`
  - 冷启动优先 `expert_vs_model`
  - 数据扩充再用 `model_vs_model_same_model`
- `--fork-depth-ratio`
  - 推荐 `0.35 ~ 0.5`
- `--num-branch-steps`
  - 推荐 `5 ~ 8`
- `--temperature-a` / `--temperature-b`
  - `model_vs_model_same_model` 下建议拉开一点
  - 例如 `0.7 / 1.0` 或 `0.8 / 1.2`
- `--seed-a` / `--seed-b`
  - 尽量不同

### 4.8 ALFWorld 常见问题

#### 1）为什么 `model_vs_model_same_model` 样本更容易被过滤？

因为两条轨迹来自同一个模型，且任务较短时很容易：

- 两边都失败
- 两边都卡在同一个 subgoal
- 最终 progress gap 不足

这属于正常现象。

可尝试：

- 增大 `--num-games`
- 拉开 `temperature-a` / `temperature-b`
- 使用不同 `seed-a` / `seed-b`
- 适当提前 fork：减小 `--fork-depth-ratio`

#### 2）为什么某些 game 没产出样本？

可能原因：

- expert path 太短
- fork 后剩余步数不够
- 两边 progress 太接近，被视为 tie
- 模型生成动作虽然合法，但没有形成可区分的进度差

#### 3）ALFWorld progress 为什么不是连续 reward？

TextWorld 版 ALFWorld 没有像 SciWorld 那样稳定直接给出可用的连续 progress 分数，因此这里采用了更任务结构化的规则评分：

- handcoded expert subgoal completion
- 完成任务则记为 `1.0`

这通常比“按步数”更合理。

---

## 5. 两个框架的主要差异

### 5.1 评分差异

SciWorld：

- 直接使用环境 score
- `progress_scalar = score / 100`

ALFWorld：

- 使用规则式 progress solver
- 基于 handcoded expert 的 subgoal 完成度

### 5.2 分支生成差异

SciWorld：

- 当前仍以“先得到分支动作，再 replay”这种模式为主
- 更适合快速实验 gold-fork 结构

ALFWorld：

- 现在是在线 rollout
- 每一步都重新读 observation + admissible commands
- 更贴近真实 agent rollout

### 5.3 适用场景差异

SciWorld 更适合：

- 快速构建有明确连续分数的 pairwise 数据
- 分析 gold / model 在科学任务上的进度差

ALFWorld 更适合：

- 构造更加真实的 online interaction ranking 数据
- 研究模型在具身文本环境中的局部决策错误

---

## 6. 生成流程建议

如果你要从零开始构造一批训练数据，建议顺序如下：

### 第一阶段：联调

- SciWorld：先跑 `random` 或 `heuristic`
- ALFWorld：先跑 `expert_vs_model + random`

目的：

- 验证环境、依赖和 JSONL 输出都正常

### 第二阶段：冷启动数据

- SciWorld：`expert_fork_model`
- ALFWorld：`expert_vs_model`

目的：

- 快速拿到一批质量相对稳定、标签清晰的样本

### 第三阶段：提升数据真实性

- ALFWorld：`model_vs_model_same_model`
- 通过不同 seed / 温度 / URL 扩展多样性

目的：

- 构造更接近真实模型比较场景的 ranking 数据

---

## 7. 输出结果检查建议

生成后建议至少检查以下内容：

- 样本总数是否足够
- `pair_type` 分布是否符合预期
- `difficulty` 分布是否过于偏 easy 或 hard
- `progress_gap` 是否普遍太小
- `trajectory_a` / `trajectory_b` 是否存在过多 subset pair
- `is_completed` 比例是否合理

如果发现：

- 全是 tie / 样本太少
  - 降低 `fork_depth_ratio`
  - 增大 `num_branch_steps`
  - 提高采样温度
- 全是非常简单样本
  - 提高 fork 深度
  - 减少 expert branch 占比
- 全是明显错误轨迹
  - 降低温度
  - 缩短 branch rollout 长度

---

## 8. 最小命令速查

### SciWorld：随机测试

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source random \
  --num-tasks 3 \
  --output /tmp/sciworld_test.jsonl
```

### SciWorld：模型生成

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork \
  --action-source openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen2.5-7B-Instruct \
  --num-tasks 5 \
  --output /tmp/sciworld_openai.jsonl
```

### ALFWorld：`expert_vs_model`

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode expert_vs_model \
  --action-source openai \
  --base-url-b http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --num-games 20 \
  --output /tmp/alfworld_expert_vs_model.jsonl
```

### ALFWorld：`model_vs_model_same_model`

```bash
cd agent-verl
python -m benchmarks.pairwise_phi_ranking.build.build_expert_fork_alfworld \
  --pair-mode model_vs_model_same_model \
  --action-source openai \
  --base-url-a http://127.0.0.1:8000/v1 \
  --base-url-b http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --temperature-a 0.8 \
  --temperature-b 1.1 \
  --seed-a 101 \
  --seed-b 202 \
  --num-games 20 \
  --output /tmp/alfworld_model_vs_model_same_model.jsonl
```

---

## 9. 总结

如果只记住最重要的几点，可以记下面这几条：

- SciWorld 用环境 score 做 progress，ALFWorld 用规则式 subgoal progress
- SciWorld 适合快速构造 fork 样本，ALFWorld 现在更强调在线 rollout
- ALFWorld 当前只建议用两种模式：`expert_vs_model`、`model_vs_model_same_model`
- 冷启动优先 `expert_vs_model`
- 想让数据更接近真实 ranking，再加入 `model_vs_model_same_model`
- 如果样本太少，优先从 `fork_depth_ratio`、`num_branch_steps`、`temperature` 三个参数调起
