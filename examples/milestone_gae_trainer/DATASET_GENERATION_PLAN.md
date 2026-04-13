# Pairwise Benchmark 数据集生成计划

> 本文档聚焦于 **如何在不依赖模型 API 的前提下**，生成高质量的 pairwise 轨迹对数据集。
> 模型 API 只在评测阶段使用（让被测模型判断 A 还是 B），数据集生成阶段完全由环境重放 + 规则标签驱动。

---

## 一、核心问题：子集偏见

当前计划中的"专家前缀对"构造法有一个结构性缺陷：

```
Expert gold path:  a1 → a2 → a3 → a4 → a5 → a6 → a7 → a8
Trajectory A:      a1 → a2 → a3           (prefix depth=3)
Trajectory B:      a1 → a2 → a3 → a4 → a5 (prefix depth=5)
```

**A 永远是 B 的子集。** 这意味着：
- 模型只用检测"B 是否包含 A 的全部内容 + 额外步骤"就能得分
- 这种捷径不反映真正的进度理解能力
- 长度启发基线（LengthHeuristicBaseline）的准确率可能接近 100%

### 解决方向

不是放弃专家前缀对（它标签最干净），而是**混合多种构造方式**：

| 构造方式 | 子集关系 | 标签来源 | 标签置信度 | 用途 |
|---------|---------|---------|-----------|------|
| 专家前缀对 | ✅ 有 | 构造性 / replay | ⭐⭐⭐⭐⭐ | 基线集，标签置信度锚点 |
| 专家分叉对 | ❌ 无 | replay 打分 | ⭐⭐⭐⭐ | 测试真正的进度理解 |
| 扰动重放对 | ❌ 无 | replay 打分 | ⭐⭐⭐⭐ | 测试噪声容忍度 |

最终数据集应分开统计不同构造类型的准确率，避免某一种构造方式主导指标。

---

## 二、构造方法详解

### 2.1 方法 A：专家前缀对 (Expert Prefix Pair)

**适用环境**：SciWorld、ALFWorld TW

**原理**：从同一条专家轨迹截取两个不同深度的前缀。

```
Gold path:      a1 → a2 → a3 → a4 → a5 → a6 → a7 → a8
Traj A (d=3):   a1 → a2 → a3
Traj B (d=6):   a1 → a2 → a3 → a4 → a5 → a6
Label:          "B"（深度更大 = 进度更高）
```

**特点**：
- A 是 B 的子集
- 标签 100% 可靠（构造性确定）
- 如果用 SciWorld replay，还能获得精确的 progress_scalar
- 简单但有子集偏见

**标签生成**：
- SciWorld：replay 后用 `getScore()` + `get_goal_progress()` 标签（双重验证）
- ALFWorld TW：深度差 ≥ 3 即判定，构造性标签

**过滤规则**：
- SciWorld：`progress_gap < 0.10` 过滤
- ALFWorld TW：`depth_diff < 3` 过滤

**样本 ID 命名规范**：`{track}_prefix_{instance}_{dA}_{dB}`

---

### 2.2 方法 B：专家分叉对 (Expert Fork Pair)

**适用环境**：SciWorld（推荐）、ALFWorld TW（有条件）

**原理**：在专家路径的某个步骤故意替换成一个错误/偏离动作，然后继续走后续的专家动作（部分可能已无法执行或效果改变），形成一条分叉轨迹。与同等截断深度的正常专家路径配对。

```
Gold path:        a1 → a2 → a3 → a4 → a5 → a6
Forked path:      a1 → a2 → X  → a4 → a5 → a6   (第3步替换为偏离动作 X)
                                                     (后续步骤可能成功或失败)

Traj A (fork):    a1 → a2 → X  → a4 → a5 → a6    replay score = 0.35
Traj B (gold):    a1 → a2 → a3 → a4 → a5 → a6    replay score = 0.70
Label:            "B"（replay 打分更高）
```

**关键：A 不是 B 的子集。** 因为第 3 步动作不同，后续所有 obs_after 也可能不同。

**分叉动作的选择策略（按优先级）**：

1. **随机有效动作**：调用 `env.getValidActions()` 获取当前步骤的所有合法动作，随机挑一个非 gold 的
2. **look around / wait**：选择类似"look around"、"wait"这种不改变状态的动作（产生最小偏离）
3. **完全随机字符串**：`env.step("do nothing")` 等无效输入（环境通常会返回错误提示）

**SciWorld 实现细节**：

```python
def build_fork_pair(replayer, task_name, variation, gold_actions, fork_step, truncate_step):
    """
    构造一个分叉对。
    
    Args:
        fork_step: 在第几步分叉（1-indexed）
        truncate_step: 截断到第几步（两条轨迹相同长度）
    
    流程:
    1. 正常轨迹：replay gold_actions[:truncate_step]
    2. 分叉轨迹：
       a. 取 gold_actions 的副本
       b. 将 gold_actions[fork_step] 替换为一个偏离动作
       c. 方法：在 fork_step 位置调用 env.getValidActions()，选择非 gold 动作
       d. replay 修改后的序列到 truncate_step
    3. 两个 ReplayResult 的 progress_scalar 做标签
    """
```

**ALFWorld TW 的局限**：
- ALFWorld TW 没有 `getScore()` 或 `get_goal_progress()` API
- 分叉后无法用环境信号精确打标
- **V1 不对 ALFWorld TW 使用分叉法**，留待 THOR 版本

**过滤规则**：
- `progress_gap < 0.10` 过滤
- 如果分叉后 replay 报错（动作序列无法执行），直接丢弃该对
- 如果分叉轨迹得分反而更高（可能发生，说明 gold path 未必是唯一最优），丢弃

**样本 ID 命名规范**：`{track}_fork_{instance}_f{fork_step}_t{truncate_step}`

---

### 2.3 方法 C：扰动重放对 (Perturbed Replay Pair)

**适用环境**：SciWorld

**原理**：对专家路径进行随机扰动（插入无效动作、跳过某步、重复某步），产生一条"不干净"的轨迹。与原始 gold path 等步数截断后配对。

```
Gold path (8 steps):      a1 → a2 → a3 → a4 → a5 → a6 → a7 → a8
Perturbed path (8 steps):  a1 → noop → a2 → a3 → noop → a4 → a5 → a6
                           (插入 2 个 noop，等步数截断后只完成到 a6)

Traj A (perturbed, t=8):  进度 = replay score at step 8 (完成到 a6 的效果)
Traj B (gold, t=8):       进度 = replay score at step 8 (完成到 a8 的效果)
Label:                     "B"（同步数下 gold 进度更高）
```

**扰动类型**：

| 扰动类型 | 操作 | 效果 |
|---------|------|------|
| `noop_insertion` | 在随机位置插入 "look around" / "wait" 等无效动作 | 浪费步数，减慢进度 |
| `action_skip` | 跳过 gold path 中的某一步 | 可能导致后续动作失败 |
| `action_repeat` | 重复执行某一步 | 通常无效果但浪费步数 |
| `random_detour` | 将某段连续 2-3 步替换为随机有效动作 | 产生非专家行为段 |

**实现要点**：
- 在 gold_actions 列表上进行操作（插入/删除/替换）
- 修改后的列表整体 replay
- 最终截断到与对照组相同的步数（以 step count 为准，不是动作数）
- 两条轨迹都做完整 replay，用 `progress_scalar` 打标

**过滤规则**：
- `progress_gap < 0.10` 过滤
- 扰动轨迹 replay 全程报错（不可执行）→ 丢弃
- 两条轨迹的步数必须相同（`|len_A - len_B| == 0`），这是天然的 `same_length_pair`

**样本 ID 命名规范**：`{track}_perturbed_{instance}_{perturbation_type}_{truncate_step}`

---

## 三、各 Track 的构造方式分配

### 3.1 `sciworld_exact` Track（V1：200 样本）

SciWorld 拥有完整的 `getScore()` + `get_goal_progress()` API，所有构造方式的标签都可以通过环境 replay 独立验证。

| 构造方式 | 样本数 | 子集关系 | 用途 |
|---------|--------|---------|------|
| Expert Prefix Pair (方法 A) | 60 | 有 | 标签锚点，基线集 |
| Expert Fork Pair (方法 B) | 80 | 无 | 核心评测集 |
| Perturbed Replay Pair (方法 C) | 60 | 无 | 同步数对照，控制长度偏见 |
| **合计** | **200** | — | — |

**子集标注**：每个样本的 metadata 中额外记录 `pair_type` 字段：
- `"expert_prefix"` / `"expert_fork"` / `"perturbed_replay"`

**指标分报**：在 leaderboard 中额外报告：
- `acc_prefix_subset`：仅 expert prefix pair 的准确率
- `acc_non_subset`：仅 fork + perturbed 的准确率
- `acc_same_length`：仅 perturbed replay pair（步数相同）的准确率

### 3.2 `alfworld_tw_constructed` Track（V1：80 样本）

ALFWorld TW 没有 exact progress API，只能使用构造性标签。

| 构造方式 | 样本数 | 子集关系 | 用途 |
|---------|--------|---------|------|
| Expert Prefix Pair (方法 A) | 80 | 有 | 全部样本 |

**V1 限制说明**：
- ALFWorld TW 在 V1 中只使用专家前缀对
- 原因：缺乏 exact progress API，非前缀对无法可靠打标
- 后续 V2 可用 THOR 版本的 `goal_conditions_met()` API 支持方法 B/C

---

## 四、SciWorld 分叉对与扰动对的生成 Pipeline

### 4.1 需要新增/修改的模块

在 `benchmarks/pairwise_phi_ranking/build/` 下：

**1. 新增 `build/trajectory_perturbation.py`**

```python
"""
轨迹扰动工具。不依赖模型 API，纯规则操作。
用于生成非子集的 pairwise 轨迹对。
"""

@dataclass
class ForkConfig:
    """分叉配置"""
    fork_step: int                # 在第几步分叉（0-indexed）
    fork_action: str              # 替换动作（由 getValidActions 选择）
    truncate_step: int            # 截断步数

@dataclass
class PerturbationConfig:
    """扰动配置"""
    perturbation_type: str        # "noop_insertion" | "action_skip" | "action_repeat" | "random_detour"
    positions: List[int]          # 扰动发生的位置
    truncate_step: int            # 截断步数

class TrajectoryPerturber:
    """
    轨迹扰动器。在 gold path 基础上生成偏离轨迹。
    纯列表操作，不调用环境和 API。
    """

    def create_forked_actions(
        self,
        gold_actions: List[str],
        fork_step: int,
        fork_action: str,
    ) -> List[str]:
        """
        在 gold_actions 的 fork_step 位置替换为 fork_action。
        
        返回修改后的动作序列（fork_step 之前不变，fork_step 处替换，之后保留原 gold）。
        """
        ...

    def create_perturbed_actions(
        self,
        gold_actions: List[str],
        perturbation_type: str,
        num_perturbations: int = 2,
        noop_action: str = "look around",
    ) -> List[str]:
        """
        对 gold_actions 进行扰动。
        
        noop_insertion: 在 num_perturbations 个随机位置插入 noop_action
        action_skip:    删除 num_perturbations 个随机位置的动作
        action_repeat:  在 num_perturbations 个随机位置复制该步动作
        random_detour:  将某段连续 2-3 步替换为 noop_action（模拟迷路）
        
        返回修改后的动作序列。
        """
        ...

class ForkActionSelector:
    """
    分叉动作选择器。
    需要环境实例来获取当前步骤的合法动作。
    """

    def select_fork_action(
        self,
        env,                          # ScienceWorldEnv 实例
        gold_action: str,             # 该步的 gold 动作（需要排除）
        strategy: str = "random_valid",  # "random_valid" | "noop" | "random_string"
    ) -> str:
        """
        选择一个不同于 gold_action 的分叉动作。
        
        random_valid: env.getValidActions() 中随机选一个非 gold 的
        noop:         固定返回 "look around"
        random_string: 固定返回 "wait"（SciWorld 会返回 "I'm not sure what you mean"）
        """
        ...
```

**2. 修改 `build/build_sciworld.py`**

在 `SciWorldBenchmarkBuilder` 中新增两个方法：

```python
class SciWorldBenchmarkBuilder:
    # ... 现有 build_expert_prefix_pairs 保持不变 ...

    def build_expert_fork_pairs(
        self,
        task_name: str,
        variation: int,
        num_pairs: int = 5,
        fork_step_range: Tuple[float, float] = (0.2, 0.7),  # 分叉点在 gold path 的 20%-70% 位置
    ) -> List[BenchmarkSample]:
        """
        构造 Expert Fork Pairs（方法 B）。

        流程:
        1. 获取 gold_actions
        2. 生成多个 fork_step 位置（在 gold path 的 20%-70%）
        3. 对每个 fork_step:
           a. 重放 gold path 到 fork_step - 1
           b. 在 fork_step 调用 env.getValidActions()，选择一个非 gold 动作
           c. 构造 forked_actions = gold[:fork_step] + [fork_action] + gold[fork_step+1:]
           d. 选择 truncate_step > fork_step（通常 = min(fork_step + 5, len(gold))）
           e. replay gold[:truncate_step] → ReplayResult_gold
           f. replay forked[:truncate_step] → ReplayResult_fork
           g. label_sciworld_pair(score_gold, score_fork)
           h. 构造 BenchmarkSample（pair_type="expert_fork"）
        4. 过滤 tie 和 progress_gap < 0.10 的样本
        """
        ...

    def build_perturbed_replay_pairs(
        self,
        task_name: str,
        variation: int,
        num_pairs: int = 3,
        perturbation_types: List[str] = ["noop_insertion", "action_repeat"],
    ) -> List[BenchmarkSample]:
        """
        构造 Perturbed Replay Pairs（方法 C）。

        流程:
        1. 获取 gold_actions
        2. 对每种 perturbation_type:
           a. perturbed_actions = perturber.create_perturbed_actions(gold_actions, type)
           b. 选择 truncate_step = len(gold_actions)（两条截断到相同步数）
              注意：perturbed 路径因为插入了无效动作，
              在相同步数下实际执行的有效动作更少
           c. replay gold[:truncate_step] → ReplayResult_gold
           d. replay perturbed[:truncate_step] → ReplayResult_perturbed
           e. label_sciworld_pair(score_gold, score_perturbed)
           f. 构造 BenchmarkSample（pair_type="perturbed_replay"）
        3. 过滤
        
        关键优势:
        - 两条轨迹步数完全相同
        - 天然属于 same_length_pair（控制长度偏见的最佳数据源）
        """
        ...

    def build_all(
        self,
        target_prefix_pairs: int = 60,
        target_fork_pairs: int = 80,
        target_perturbed_pairs: int = 60,
        task_variations: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[BenchmarkSample], Dict]:
        """
        构造全部 sciworld_exact 样本的主入口。

        流程:
        1. 加载 variations_idx
        2. 枚举 (task_id, variation)
        3. 均衡分配三种构造类型的配额
        4. 依次调用:
           - build_expert_prefix_pairs()
           - build_expert_fork_pairs()
           - build_perturbed_replay_pairs()
        5. filter_and_bucket()
        6. randomize_ab_position()（对所有样本都做 A/B 随机化）
        7. split_dev_test()
        8. 写入 JSONL

        返回:
        - samples: 全部样本
        - stats: {
            "total": N,
            "by_pair_type": {"expert_prefix": 60, "expert_fork": 80, "perturbed_replay": 60},
            "by_difficulty": {"easy": ..., "medium": ..., "hard": ...},
            "by_task_type": {...},
            "subset_ratio": 0.30,  # 有子集关系的样本占比
            "same_length_ratio": 0.30,  # 步数相同的样本占比
          }
        """
        ...
```

---

## 五、Schema 扩展

`BenchmarkSample` 需要新增一个字段来标识构造方式：

```python
# core/schema.py 中 BenchmarkSample 新增字段：

@dataclass
class BenchmarkSample:
    # ... 现有字段不变 ...
    
    pair_type: str = ""             # "expert_prefix" | "expert_fork" | "perturbed_replay"
    is_subset_pair: bool = False    # 标记 A 是否是 B 的子集（或反向）
```

对应的 `to_dict()` / `from_dict()` 也需要更新。

`filters.py` 中新增一个子集关系检测函数：

```python
# core/filters.py 新增：

def detect_subset_relationship(sample: BenchmarkSample) -> bool:
    """
    检测两条轨迹是否存在前缀/子集关系。
    
    判定规则:
    短轨迹的所有 action 序列是长轨迹 action 序列的前缀。
    
    返回 True 表示存在子集关系。
    """
    actions_a = [step.action for step in sample.trajectory_a.steps]
    actions_b = [step.action for step in sample.trajectory_b.steps]
    shorter, longer = sorted([actions_a, actions_b], key=len)
    return longer[:len(shorter)] == shorter


def filter_prefix_relationship(sample: BenchmarkSample) -> bool:
    """
    过滤具有前缀关系的样本（用于非前缀类型的构造方法）。
    
    如果 pair_type 不是 "expert_prefix" 但检测到子集关系，
    返回 True（应过滤）。
    """
    if sample.pair_type == "expert_prefix":
        return False  # 前缀对允许子集关系
    return detect_subset_relationship(sample)
```

---

## 六、Metrics 扩展

`eval/metrics.py` 中新增按构造类型分报的指标：

```python
# eval/metrics.py 新增：

def compute_accuracy_by_pair_type(
    results: List[dict],
    samples: List[BenchmarkSample],
) -> Dict[str, float]:
    """
    按 pair_type 分组统计准确率。
    
    返回:
    {
        "acc_expert_prefix": 0.85,
        "acc_expert_fork": 0.72,
        "acc_perturbed_replay": 0.68,
        "acc_subset_pairs": 0.85,      # 所有有子集关系的样本
        "acc_non_subset_pairs": 0.70,  # 所有无子集关系的样本
    }
    """
    ...
```

---

## 七、前缀关系过滤的完整流程

对于方法 B（分叉对）和方法 C（扰动对），必须在构造后验证确实不存在子集关系：

```
构造样本
  ↓
replay 两条轨迹拿到 ReplayResult
  ↓
label_rules 打标
  ↓
filter_ties (progress_gap < 0.10)
  ↓
filter_prefix_relationship  ← 新增：如果非前缀类型却出现前缀关系，丢弃
  ↓
assign_difficulty
  ↓
detect_subset_relationship → 写入 is_subset_pair 字段
  ↓
randomize_ab_position
  ↓
写入 JSONL
```

---

## 八、SciWorld 分叉动作获取的详细流程

这是最关键也是最需要与环境交互的部分：

```python
def get_fork_action_for_step(env, task_name, variation, gold_actions, fork_step):
    """
    获取指定步骤的分叉动作。
    
    流程:
    1. env.load(task_name, variation, ...)
    2. env.reset()
    3. for i in range(fork_step):
          env.step(gold_actions[i])   # 重放到分叉点
    4. valid_actions = env.getValidActions()  # 获取当前合法动作
    5. candidates = [a for a in valid_actions if a != gold_actions[fork_step]]
    6. if not candidates:
          return None  # 无可用分叉动作，跳过该 fork_step
    7. return random.choice(candidates)
    
    注意:
    - 这需要额外的 env 交互，但不需要模型 API
    - getValidActions() 是 SciWorld 的内置 API
    - 每个分叉点需要完整 replay 到该步骤
    - 可以批量处理：一次 replay 中在多个步骤记录 valid_actions
    """
```

---

## 九、实施时间线与推荐编码顺序

| 阶段 | 任务 | 产出 | 前置依赖 |
|------|------|------|---------|
| **S0** | schema.py 新增 `pair_type` / `is_subset_pair` 字段 | 更新后的 schema | 无 |
| **S0** | filters.py 新增 `detect_subset_relationship` / `filter_prefix_relationship` | 更新后的 filters | schema |
| **S1** | 实现 `build/trajectory_perturbation.py` | TrajectoryPerturber + ForkActionSelector | schema |
| **S2** | 实现 `build/replay_sciworld.py` | SciWorldReplayer | schema, 需要 SciWorld JVM |
| **S3** | 修改 `build/build_sciworld.py`，新增 `build_expert_fork_pairs` + `build_perturbed_replay_pairs` | 完整的三类构造 | replay_sciworld, trajectory_perturbation |
| **S4** | 实现 `build/collect_alfworld_obs_after.py` | ALFWorldObsCollector | 需要 ALFWorld 环境 |
| **S5** | 实现 `build/build_alfworld_tw.py`（仅前缀对） | ALFWorld TW 样本 | collect |
| **S6** | metrics.py 新增 `compute_accuracy_by_pair_type` | 按构造类型分报指标 | 无 |
| **S7** | `build/run_build_all.py`：打印完整的数据集统计报告 | 一键构造 + 质量报告 | 全部 build 模块 |

---

## 十一、总结

| 关键决策 | 结论 |
|---------|------|
| 数据集生成是否需要模型 API？ | **不需要**。全程使用环境 replay + 规则标签 |
| 如何避免子集关系？ | 混合三种构造法：前缀对（有子集）+ 分叉对（无子集）+ 扰动对（无子集） |
| ALFWorld TW 怎么办？ | V1 只用前缀对（承认局限）；V2 等 THOR 版有 exact progress API 后再扩展 |
| 标签从哪来？ | SciWorld：`getScore()` + `get_goal_progress()` 环境 replay；ALFWorld TW：构造性深度差 |
| 如何控制长度偏见？ | 扰动对天然等步数；指标分开报告 `acc_same_length` |
| 如何知道哪个构造类型难？ | 每个样本标注 `pair_type`，指标按类型分报 |
