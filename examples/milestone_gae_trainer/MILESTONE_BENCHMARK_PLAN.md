# Pairwise Potential Ranking Benchmark —— 详细框架生成计划

> 综合两轮分析与审查意见后的最终修订版。本文档的目标不是“讨论方向”，而是沉淀一个可执行、可审计、可发布的 benchmark 方案。

---

## 一、核心定位与关键决策

### 1.1 Benchmark 的本质

> **给定同一个任务实例下的两条截断轨迹，势能函数能否正确判断哪一条离成功更近。**

这比单纯的 success rate 更贴近 Milestone-GAE 的核心，因为训练中真正起作用的是 `phi` 的相对高低与增量，而不是只看最终是否成功。

### 1.2 五个已采纳的关键修正

| 修正点 | 结论 | 最终采纳 |
|--------|------|---------|
| SciWorld score 尺度 | Python wrapper 暴露的是 `0-100`，benchmark 内部统一归一化到 `[0,1]` | ✅ 保留 `raw_score`，新增 `progress_scalar = raw_score / 100.0` |
| V1 定义与样本混合 | `exact_state` 不能与 `exact_by_construction` 混在一个总榜里 | ✅ 拆成独立 track，不给混合总分 |
| ALFWorld TextWorld 缺少 `obs_after` | 现有缓存数据不够，必须独立重放采集 | ✅ Phase 2 前置 replay 采集链路 |
| Schema 过于 SciWorld-specific | 不同环境的 progress 证据结构不同 | ✅ 采用共享主 schema + `label_evidence` 分离方案 |
| Milestone 所有权未定 | `full_pipeline` 与 `fixed_milestones` 是两种不同 benchmark | ✅ 核心样本移除 `milestones`，改为 companion file |

### 1.3 评测线与 track 的最终定义

本 benchmark 不再尝试用一个混合总分覆盖所有环境，而是拆成多个语义清晰的 track。

- `sciworld_exact`
  基于环境重放与 `goal_progress` / `score` 的精确规则标签。
- `alfworld_tw_constructed`
  基于同一 expert trajectory 的 prefix 深度构造标签。
- `alfworld_thor_exact`
  预留给后续 THOR 版 ALFWorld 的 exact-state 标签。

每个 track 都支持两条评测线：

- `full_pipeline`
  被测系统自己生成 milestones，再独立对 A/B 打 `phi`。
- `fixed_milestones`
  使用官方 `reference_milestones.jsonl`，只测 judge 打分能力。

主 leaderboard 不再给一个混合总分，而是按 `track × eval_line` 分列展示。

---

## 二、标签正确性方案

### 2.1 SciWorld：Exact-State Label

**已核实事实**

- SciWorld 底层 `getScore()` 语义是 `[0,1]` 的任务进展分数。
- Python wrapper 在 [scienceworld.py](D:/Workspace/Agentic/agent-verl/agent_system/environments/env_package/sciworld/ScienceWorld/scienceworld/scienceworld.py#L454) 把它乘以 `100` 并四舍五入成整数 `score`。
- benchmark 内部统一恢复为：
  `progress_scalar = raw_score / 100.0`
- SciWorld Python API 提供 `get_goal_progress()`，可返回精确子目标完成状态。

**标签规则**

对同一 `task_variation` 的两条截断轨迹 A、B：

1. 在新环境实例中重放动作序列到各自截断点。
2. 获取 `raw_score_A`、`raw_score_B`。
3. 归一化得到 `progress_scalar_A = raw_score_A / 100.0`，`progress_scalar_B = raw_score_B / 100.0`。
4. 同时解析 `goal_progress`，得到：
   - `ordered_done`
   - `unordered_done`
5. 构造：
   `progress_key = (ordered_done, unordered_done, progress_scalar)`

判定规则：

- `progress_key_A > progress_key_B` → `label = "A"`
- `progress_key_A < progress_key_B` → `label = "B"`
- `progress_key_A == progress_key_B` → 丢弃

过滤规则：

- `abs(progress_scalar_A - progress_scalar_B) < 0.10` → 丢弃

**为什么保留三层证据**

- `ordered_done` 优先级最高，代表强约束主线进展。
- `unordered_done` 次之，代表可选或并行子目标进展。
- `progress_scalar` 作为细粒度 tiebreak，同时便于难度分桶。
- `raw_score` 仅用于审计与 debug，不作为 benchmark 内部阈值单位。

### 2.2 ALFWorld TextWorld：Exact-by-Construction Label

当前代码所用环境是 TextWorld 版 ALFWorld。它没有像 SciWorld 那样现成暴露统一、稳定的 exact partial-progress API，因此 v1 不把它伪装成 `exact_state`，而是明确定位为 `exact_by_construction`。

**标签规则**

对同一个任务实例、同一条 expert trajectory：

- `A = prefix[:d_A]`
- `B = prefix[:d_B]`
- `d_B > d_A`
- `label = "B"`

这不是主观标签，而是构造性确定标签：在同一条已知正确路径上，更深的 prefix 客观上更接近成功。

过滤规则：

- `d_B - d_A < 3` → 丢弃
- 两条 prefix 的 `terminal_observation` 高度相似 → 丢弃

补充说明：

- `terminal_observation` 不能直接依赖现有缓存。
- benchmark 构造时必须在 ALFWorld 环境中独立 replay expert 动作序列，主动采集每步 `obs_after`。

### 2.3 ALFWorld THOR：预留 Exact-State Label

后续若切换到 THOR 版 ALFWorld，则使用：

- `goal_conditions_met() -> (conditions_met, conditions_total)`

标签规则与 SciWorld 同风格：

- 用 `conditions_met / conditions_total` 作为 `progress_scalar`
- `label_evidence.scheme = "alfworld_thor_goal_conditions"`
- tie 丢弃

### 2.4 标签质量分级

| Track | 方法 | 标签质量 | 用途 |
|------|------|---------|------|
| `sciworld_exact` | replay + goal_progress + normalized score | ⭐⭐⭐⭐⭐ | v1 主榜 |
| `alfworld_tw_constructed` | expert-prefix construction | ⭐⭐⭐⭐ | v1 辅榜 |
| `alfworld_thor_exact` | goal_conditions_met | ⭐⭐⭐⭐⭐ | 后续扩展 |
| off-policy rollout pairs | replay-based ordering | ⭐⭐⭐ | v2 扩展 |

---

## 三、Benchmark 分版规划

### 3.1 V1：双 Track 发布

V1 不再定义成“一个混合 benchmark”，而是两个独立发布的 track：

- `sciworld_exact`
- `alfworld_tw_constructed`

二者并列展示，不做加权平均。

### 3.2 V1-a：`sciworld_exact`

**目标**

- 快速、干净、完全规则化
- 标签 100% 来自环境重放与规则函数

**规格**

| 维度 | 规格 |
|------|------|
| 数据来源 | expert prefix + expert fork + perturbed replay |
| 标签 | replay + `goal_progress` + 归一化 `progress_scalar` |
| 样本数 | 200 |
| 轨迹类型 | 均为截断，允许 `done=False` |
| 控制 | 同 `task_variation`、过滤 tie、过滤 `progress_gap < 0.10`、过滤非前缀类子集 |

**样本构成**

- Expert prefix pairs: 60
- Expert fork pairs: 80
- Perturbed replay pairs: 60

### 3.3 V1-b：`alfworld_tw_constructed`

**目标**

- 在不伪造 exact-state 的前提下，构造高质量、无歧义的 pairwise benchmark

**规格**

| 维度 | 规格 |
|------|------|
| 数据来源 | 同任务实例下的 expert trajectory prefix pairs |
| 标签 | exact-by-construction |
| 样本数 | 80 |
| 轨迹类型 | 均为截断 prefix |
| 控制 | `d_B - d_A >= 3`、过滤高相似末态 |

**样本构成**

- 6 大任务类型均衡采样
- 每类约 13 对

### 3.4 V2：Off-Policy Progress Benchmark

V2 再引入更真实的 detour、噪声与错误动作轨迹。

| 维度 | 规格 |
|------|------|
| 数据来源 | 真实 policy rollout |
| 标签 | SciWorld 用 replay+goal_progress；ALFWorld THOR 用 `goal_conditions_met` |
| 样本数 | 500 |
| 控制 | 长度均衡、难度分桶、位置随机化 |

---

## 四、评测协议

### 4.1 主指标：Independent Phi Ranking Accuracy

对每个样本 `(task, traj_A, traj_B, label)`：

1. 获取 milestones。
   - `full_pipeline`：被测系统自己生成
   - `fixed_milestones`：从 `reference_milestones.jsonl` 读取
2. 对 `traj_A` 独立打分，得到 `phi_A = phi_sequence_A[-1]`
3. 对 `traj_B` 独立打分，得到 `phi_B = phi_sequence_B[-1]`
4. 预测：
   - `phi_A > phi_B` → `"A"`
   - 其他情况 → `"B"`
5. 若 `phi_A == phi_B`，记为错误

主指标：

- `phi_rank_acc`

### 4.2 辅指标：Direct Pairwise Choice Accuracy

给 judge 同时看 `traj_A` 与 `traj_B`，直接问“哪条更接近成功”。

这不是训练时使用的 pointwise `phi` 机制，因此只作为辅指标：

- `direct_choice_acc`

### 4.3 两条评测线

**评测线 A：`full_pipeline`**

- milestone 生成能力
- judge 打分能力
- 两者共同作用

**评测线 B：`fixed_milestones`**

- milestone 固定
- 只测 judge 的 `phi` 排序能力

leaderboard 必须按两条评测线分别报告。

### 4.4 完整指标体系

| 指标 | 含义 |
|------|------|
| `phi_rank_acc` | 主指标，独立打分后的排序准确率 |
| `direct_choice_acc` | 辅指标，直接二选一准确率 |
| `acc_easy` | 简单子集准确率 |
| `acc_hard` | 困难子集准确率 |
| `same_length_subset_acc` | 长度受控子集准确率 |
| `phi_consistency` | 同样本多次打分的一致率 |
| `acc_sciworld_exact` | `sciworld_exact` track 准确率 |
| `acc_alfworld_tw_constructed` | `alfworld_tw_constructed` track 准确率 |
| `acc_by_tasktype` | 按任务类型细报 |

---

## 五、最终数据集 Schema

核心样本不再直接内嵌 `milestones`，而是采用“共享主 schema + `label_evidence` + companion milestone file”。

### 5.1 核心样本 Schema

```jsonc
{
  "sample_id": "sw_task3_v12_pair_001",
  "track": "sciworld_exact",                // "sciworld_exact" | "alfworld_tw_constructed" | "alfworld_thor_exact"
  "env": "sciworld",                        // "sciworld" | "alfworld_tw" | "alfworld_thor"
  "instance_id": {
    "task_name": "measure-melting-point-known",
    "variation": 12
  },
  "task_description": "Your task is to measure the melting point...",
  "split": "test",                          // "dev" | "test"
  "pair_type": "expert_prefix_pair",        // "expert_prefix_pair" | "cross_rollout_pair"
  "label_type": "exact_state",              // "exact_state" | "exact_by_construction"
  "label": "A",                             // "A" | "B"
  "label_source": "replay_goal_progress_v1",
  "label_rule_version": "v1.0",
  "difficulty": "hard",                     // "easy" | "medium" | "hard"
  "progress_gap": 0.18,                     // 统一使用 [0,1] 标量差
  "reference_milestones_id": "sw_task3_v12_ref_001",   // 没有官方参考里程碑时可为 null

  "trajectory_a": {
    "steps": [
      {
        "step_idx": 0,
        "obs_before": "...",
        "action": "...",
        "obs_after": "..."
      }
    ],
    "truncated_at_step": 15,
    "terminal": {
      "observation": "...",
      "done": false,
      "won": false
    }
  },

  "trajectory_b": {
    "steps": [
      {
        "step_idx": 0,
        "obs_before": "...",
        "action": "...",
        "obs_after": "..."
      }
    ],
    "truncated_at_step": 8,
    "terminal": {
      "observation": "...",
      "done": false,
      "won": false
    }
  },

  "label_evidence": {
    "scheme": "sciworld_goal_progress"
  }
}
```

### 5.2 `label_evidence` 的三种子结构

**A. SciWorld**

```jsonc
{
  "scheme": "sciworld_goal_progress",
  "a": {
    "progress_scalar": 0.42,
    "ordered_done": 2,
    "unordered_done": 1,
    "raw_score": 42
  },
  "b": {
    "progress_scalar": 0.15,
    "ordered_done": 1,
    "unordered_done": 0,
    "raw_score": 15
  }
}
```

**B. ALFWorld TextWorld**

```jsonc
{
  "scheme": "expert_prefix_depth",
  "a": {
    "prefix_depth": 5,
    "expert_total_steps": 18
  },
  "b": {
    "prefix_depth": 11,
    "expert_total_steps": 18
  }
}
```

**C. ALFWorld THOR**

```jsonc
{
  "scheme": "alfworld_thor_goal_conditions",
  "a": {
    "conditions_met": 2,
    "conditions_total": 4,
    "progress_scalar": 0.50
  },
  "b": {
    "conditions_met": 1,
    "conditions_total": 4,
    "progress_scalar": 0.25
  }
}
```

### 5.3 Companion File：`reference_milestones.jsonl`

对于支持 `fixed_milestones` 评测线的样本，官方参考里程碑不再写入核心样本，而是单独存在 companion file：

```jsonc
{
  "reference_milestones_id": "sw_task3_v12_ref_001",
  "instance_id": {
    "task_name": "measure-melting-point-known",
    "variation": 12
  },
  "source": "generator",                    // "generator" | "template" | "manual"
  "milestones": [
    {"id": "M1", "name": "...", "phi": 0.2, "criteria": "..."},
    {"id": "M2", "name": "...", "phi": 0.5, "criteria": "..."},
    {"id": "M3", "name": "...", "phi": 1.0, "criteria": "..."}
  ]
}
```

---

## 六、数据生成 Pipeline

### Phase 1：SciWorld 样本构造

1. 加载 SciWorld，枚举 `task_types × variations`
2. 对每个 `(task, variation)`：
   - 获取 expert gold path
   - 构造 expert prefix pairs
   - 构造 cross-rollout pairs
3. 对所有 candidate pairs：
   - 在新环境中 replay 到截断点
   - 采集 `raw_score`
   - 计算 `progress_scalar = raw_score / 100.0`
   - 解析 `goal_progress`
   - 生成 `label_evidence`
4. 过滤：
   - tie
   - `progress_gap < 0.10`
5. 保存 `obs_after`
6. 写入 JSONL

### Phase 2：ALFWorld TextWorld 的 replay 采集

这一步是新增前置步骤，不能依赖已有缓存。

1. 加载 ALFWorld TextWorld 环境
2. 对每个任务实例获取 expert action sequence
3. 独立 replay expert 轨迹
4. 每步主动采集：
   - `obs_before`
   - `action`
   - `obs_after`
5. 形成新的 benchmark 专用 expert trajectory cache

### Phase 3：ALFWorld TextWorld 样本构造

1. 基于 Phase 2 采集到的 expert trajectory
2. 构造 prefix pairs：
   - `A = steps[:d_A]`
   - `B = steps[:d_B]`
   - `d_B - d_A >= 3`
3. 保存：
   - `terminal.observation = steps[d-1]["obs_after"]`
   - `label_evidence.scheme = "expert_prefix_depth"`
4. 过滤：
   - 末态高度相似
   - 长度偏见过强的样本

### Phase 4：Companion Milestone 生成

对每个可支持 `fixed_milestones` 的实例：

1. 生成或整理参考 milestones
2. 写入 `reference_milestones.jsonl`
3. 在核心样本中写 `reference_milestones_id`

### Phase 5：质量审计

1. 随机抽样 20% 样本，人工检查 label 直觉一致性
2. 检查 `progress_gap` 分布，确保 easy/medium/hard 三桶覆盖
3. 检查长度分布与长度偏见
4. 检查各 task type / variation 覆盖
5. 检查 `obs_after` 完整率是否为 100%

---

## 七、关键工程修正

### 修正 1：末步偏差

当前 judge prompt 默认看到的是每步动作前状态，但 benchmark 要比较的是截断轨迹执行后的终态。

因此：

1. 每步必须保存 `obs_after`
2. `terminal.observation = steps[-1]["obs_after"]`
3. judge prompt 在最后一步补充 terminal state

示意：

```text
Step N:
  Environment State: {obs_before_N}
  Agent Action: {action_N}
  [Terminal State after action]: {obs_after_N}
```

### 修正 2：长度偏见控制

更长轨迹天然更可能得到更高 `phi`，因此 benchmark 必须单独测长度控制子集。

措施：

- 构造 `same_length_pairs`
- 约束 `|len_A - len_B| <= 2`
- 额外报告 `same_length_subset_acc`
- A/B 呈现顺序随机化

### 修正 3：标签不依赖缓存

所有标签必须由 benchmark 构造工具独立生成：

- 环境 replay
- 规则函数
- 结构化证据保存

不信任训练时保存的缓存 metadata。

### 修正 4：SciWorld 进展统一归一化

虽然 Python wrapper 返回 `0-100` 的整数 `score`，benchmark 内部统一恢复原始语义：

- `progress_scalar = raw_score / 100.0`

所有阈值、难度分桶、`progress_gap` 都以 `[0,1]` 归一化尺度为准。

---

## 八、文件结构设计

```text
benchmarks/
└── pairwise_phi_ranking/
    ├── sciworld_exact/
    │   ├── README.md
    │   ├── dev.jsonl
    │   ├── test.jsonl
    │   ├── metadata.json
    │   └── reference_milestones.jsonl
    ├── alfworld_tw_constructed/
    │   ├── README.md
    │   ├── dev.jsonl
    │   ├── test.jsonl
    │   ├── metadata.json
    │   └── reference_milestones.jsonl
    ├── alfworld_thor_exact/
    │   └── README.md
    ├── build/
    │   ├── replay_and_label.py
    │   ├── build_sciworld.py
    │   ├── build_alfworld_tw.py
    │   ├── collect_alfworld_obs_after.py
    │   ├── build_reference_milestones.py
    │   └── filter_and_bucket.py
    ├── eval/
    │   ├── eval.py
    │   ├── judge_wrapper.py
    │   └── metrics.py
    └── results/
        └── leaderboard.md
```

---

## 九、实施时间线

| 阶段 | 任务 | 产出 |
|------|------|------|
| Week 1 | 实现 `replay_and_label.py` 与 SciWorld 规则标签链路 | `sciworld_exact` 原始样本 |
| Week 1 | 生成 SciWorld expert prefix pairs | `sciworld_exact/dev.jsonl` 初版 |
| Week 2 | 生成 SciWorld cross-rollout pairs | `sciworld_exact/test.jsonl` 初版 |
| Week 2 | 实现 `collect_alfworld_obs_after.py` | benchmark 专用 ALFWorld expert trajectory cache |
| Week 3 | 构造 `alfworld_tw_constructed` 样本 | `alfworld_tw_constructed/dev.jsonl` |
| Week 3 | 实现 `build_reference_milestones.py` | `reference_milestones.jsonl` |
| Week 4 | 实现 `eval.py`、`judge_wrapper.py`、`metrics.py` | 可运行评测脚本 |
| Week 4 | 跑 baseline 并写 leaderboard | `leaderboard.md` |

---

## 十、Baseline 评测计划

发布时每个 track 都分别报告 `full_pipeline` 与 `fixed_milestones` 两条线的结果。

| Judge 配置 | 预期表现 | 说明 |
|-----------|---------|------|
| Random | ~50% | 理论下界 |
| 长度启发 | 高于随机 | 用于测长度偏见上界 |
| 静态 milestone + 关键词匹配 | 待测 | 非 LLM 基线 |
| Qwen2.5-7B judge | 待测 | 当前可部署模型 |
| Qwen3-32B judge | 待测 | 更强 judge |
| GPT-4o judge | 待测 | 商业上界 |
| Milestone-GAE pipeline | 待测 | 核心评测对象 |

leaderboard 表头建议至少包含：

- `model`
- `track`
- `eval_line`（`no_milestones` / `with_milestones` / `phi_pointwise`）
- `pairwise_choice_acc`（主指标）
- `acc_easy`
- `acc_hard`
- `same_length_subset_acc`
- `phi_rank_acc`（可选，有 Judge API 时填写）


---

## 十一、总体判断与优先级

### 立即执行

1. 实现 `sciworld_exact` 的 replay + 规则标签生成
2. 修正末步偏差，统一保存 `obs_after`
3. 明确 `progress_scalar` 为 `[0,1]` 的 benchmark 内部标准
4. 实现 `eval.py` 的两条评测线

### 紧接着做

1. 实现 ALFWorld TextWorld 的 replay 采集
2. 构造 `alfworld_tw_constructed` track
3. 引入 companion milestone file

### 后续扩展

1. 接入 `alfworld_thor_exact`
2. 扩展 off-policy rollout benchmark
3. 扩展更多 judge / milestone 生成策略

这个 benchmark 的最终价值，不是替代 success rate，而是成为 Milestone-GAE 系列工作的标准配套评测工具：既能验证当前 judge 的势能判别能力，也能稳定比较未来不同 milestone 生成策略与不同 judge 模型的真实进展感知能力。
