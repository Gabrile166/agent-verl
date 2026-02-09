# Milestone-Guided GAE: 基于 LLM 势能引导的强化学习方案

> **核心思想**：利用强模型（Teacher LLM）的先验知识构建"势能地势图"，通过结构化里程碑判定驱动弱模型（Student）高效完成任务。

---

## 目录

1. [方案概述](#1-方案概述)
2. [理论基础](#2-理论基础)
3. [三层架构设计](#3-三层架构设计)
4. [核心算法](#4-核心算法)
5. [与 GRPO 的融合](#5-与-grpo-的融合)
6. [工程实现](#6-工程实现)
7. [实验设计](#7-实验设计)

---

## 1. 方案概述

### 1.1 问题背景

在 Agent 强化学习中，传统方法面临以下挑战：

| 问题 | 传统 GRPO | 传统 GAE |
|------|-----------|----------|
| 信用分配 | Episode-level（粗粒度） | Step-level（需要 Critic） |
| Value 估计 | 无 | 需训练 Critic 网络 |
| 稀疏奖励 | 受影响大 | 受影响大 |

### 1.2 核心创新

本方案提出 **"LLM-as-Critic"** 范式：

```
传统 GAE:  Critic Network → V(s) → TD Error → Advantage
本方案:    LLM Judge → Φ(s) → TD Error → Advantage
```

**关键设计**：
- 用 LLM 的任务理解能力替代 Critic 网络
- 将开放式"打分问题"转化为结构化"里程碑判定问题"
- 每条轨迹仅调用一次 Judge（而非每步一次）

### 1.3 方案优势

| 维度 | 效果 |
|------|------|
| **无需 Critic 训练** | 工程复杂度大幅降低 |
| **Step-level 信用分配** | 样本效率提升 |
| **结构化判定** | Judge 输出稳定可靠 |
| **里程碑复用** | 同类任务共享模板 |

---

## 2. 理论基础

### 2.1 Potential-Based Reward Shaping (PBRS)

标准 PBRS 定理保证：添加势能差形式的奖励塑形不会改变最优策略。

$$r'(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

**本方案的关键区分**：
- **Φ(s) 仅作为 Value Function 估计**，不参与奖励定义
- **r_t 保持为原始环境奖励**（稀疏的成功/失败信号）

### 2.2 GAE (Generalized Advantage Estimation)

标准 GAE 公式：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

递归形式：

$$A_t = \delta_t + (\gamma \lambda) A_{t+1}$$

**本方案的替换**：用 LLM Judge 输出的 $\Phi(s)$ 替代 $V(s)$。

### 2.3 信用倒推机制

当 $\lambda$ 设置较高（如 0.95）时，GAE 具有强大的信用倒推能力：

**示例**：5 步轨迹，只有最后一步成功

| 步骤 | δ_t | A_t (λ=0.95) | 解释 |
|------|-----|--------------|------|
| s1 | -0.01 | **1.36** | 成功信号倒推 |
| s2 | -0.01 | **1.46** | 成功信号倒推 |
| s3 | -0.01 | **1.56** | 成功信号倒推 |
| s4 | +0.19 | **1.67** | 里程碑达成 |
| s5 | +1.58 | **1.58** | 最终成功 |

**结论**：即使中间步骤的 δ_t < 0，高 λ 的 GAE 也能将成功信号倒推回去。

---

## 3. 三层架构设计

### 3.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Milestone-Guided GAE 架构                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 第一层：离线里程碑提取 (每种任务类型一次)                    │   │
│  │   强模型 → 里程碑列表 + 判定标准 + Φ 分配                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 第二层：在线结构化判定 (每条轨迹一次)                        │   │
│  │   Judge LLM + 完整轨迹 → 每步的 highest_milestone          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 第三层：优势计算 (纯数学)                                   │   │
│  │   Φ(s) → TD Error → GAE → 全局归一化 → PPO 更新            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 第一层：离线里程碑提取

#### 输入
- 任务类型描述
- （可选）专家示例轨迹

#### 输出格式

```json
{
  "task_type": "把{物品}放到{位置}",
  "milestones": [
    {
      "id": "M1",
      "name": "找到目标物品",
      "phi": 0.15,
      "criteria": "观察文本中出现了 {物品}",
      "keywords": ["see", "find", "{物品}"]
    },
    {
      "id": "M2",
      "name": "拿起目标物品",
      "phi": 0.30,
      "criteria": "执行了 take 动作且确认 pick up 成功",
      "keywords": ["pick up", "take"]
    },
    {
      "id": "M3",
      "name": "到达清洗工具",
      "phi": 0.45,
      "criteria": "当前位置包含 sinkbasin",
      "keywords": ["sinkbasin", "sink"]
    },
    {
      "id": "M4",
      "name": "完成清洗",
      "phi": 0.60,
      "criteria": "执行了 clean 动作且确认清洗成功",
      "keywords": ["clean", "wash"]
    },
    {
      "id": "M5",
      "name": "到达目的地",
      "phi": 0.80,
      "criteria": "当前位置是目标放置地点 {位置}",
      "keywords": ["{位置}"]
    },
    {
      "id": "M6",
      "name": "放置完成",
      "phi": 1.00,
      "criteria": "执行了 put 动作且确认放置成功",
      "keywords": ["put", "place"]
    }
  ]
}
```

#### 设计原则

1. **里程碑数量**：4-7 个（太少失去引导作用，太多增加 Judge 负担）
2. **判定标准客观化**：避免 "做得不错" 等主观描述
3. **Φ 分配非均匀**：根据难度分配，难步骤前后跳跃大

### 3.3 第二层：在线结构化判定

#### 核心设计：每条轨迹调用一次

**不是**：对每个 s_t 单独调用 Judge
**而是**：一次性输入完整轨迹，输出每步的判定结果

#### Prompt 模板

```
你是一个任务进度评估器。

## 任务描述
{task_description}

## 里程碑清单
M1 (Φ=0.15): {milestone_1_name} — 判定标准：{criteria_1}
M2 (Φ=0.30): {milestone_2_name} — 判定标准：{criteria_2}
M3 (Φ=0.45): {milestone_3_name} — 判定标准：{criteria_3}
M4 (Φ=0.60): {milestone_4_name} — 判定标准：{criteria_4}
M5 (Φ=0.80): {milestone_5_name} — 判定标准：{criteria_5}
M6 (Φ=1.00): {milestone_6_name} — 判定标准：{criteria_6}

## Agent 执行轨迹

Step 1:
  Action: {action_1}
  Observation: {obs_1}

Step 2:
  Action: {action_2}
  Observation: {obs_2}

... (共 {T} 步)

## 任务

请对每个步骤判断已达成的最高里程碑。

输出格式 (JSON):
{
  "judgments": [
    {"step": 1, "highest_milestone": "M0", "phi": 0.0},
    {"step": 2, "highest_milestone": "M1", "phi": 0.15},
    ...
  ],
  "final_success": true/false,
  "reasoning": "简要说明判断依据"
}

注意：
1. M0 表示尚未达成任何里程碑，phi=0.0
2. 里程碑通常是单调递增的（偶尔可能因错误动作回退）
3. 只有最终成功才能达到 M6 (phi=1.0)
```

#### 输出解析

```python
def parse_judge_output(output: str) -> List[float]:
    """解析 Judge 输出，返回每步的 Φ 值"""
    data = json.loads(output)
    return [step["phi"] for step in data["judgments"]]
```

### 3.4 第三层：优势计算

详见 [第 4 节](#4-核心算法)。

---

## 4. 核心算法

### 4.1 符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $r_t$ | 环境原始奖励 | 成功=1, 其他=0 |
| $c$ | 时间成本 | 0.01 |
| $\Phi(s_t)$ | LLM 估计的势能 | [0, 1] |
| $\gamma$ | 折扣因子 | 0.99 |
| $\lambda$ | GAE 系数 | 0.95 |
| $G$ | 组大小 | 8 |

### 4.2 完整算法流程

```
输入：
  - G 条轨迹 {τ_1, τ_2, ..., τ_G}
  - 每条轨迹 τ_i = [(s_1, a_1, r_1), ..., (s_{T_i}, a_{T_i}, r_{T_i})]
  - 里程碑模板 M

输出：
  - 归一化后的优势值 A_norm

算法：

1. 里程碑判定（每条轨迹调用一次 Judge）
   For i = 1 to G:
     Φ_i = Judge(τ_i, M)  // 返回 [Φ_1, Φ_2, ..., Φ_{T_i}]

2. TD Error 计算
   For i = 1 to G:
     For t = 1 to T_i:
       r̃_t = r_t - c  // 加入时间成本
       
       // 处理边界条件（关键设计）
       If t == T_i:
         If done AND success:
           Φ_{next} = 1.0           // 成功：未来势能为理想值
         Else:
           Φ_{next} = Φ_i[T_i]      // 失败/截断：保持当前势能
       Else:
         Φ_{next} = Φ_i[t+1]
       
       δ_i[t] = r̃_t + γ · Φ_{next} - Φ_i[t]

   // 核心区分因素：成功轨迹有 r_T=1，失败轨迹 r_T=0
   // 势能不跌落，惩罚仅来自"没拿到最终奖励"

3. GAE 递归
   For i = 1 to G:
     A_i[T_i] = δ_i[T_i]
     For t = T_i-1 down to 1:
       A_i[t] = δ_i[t] + γλ · A_i[t+1]

4. 全局归一化
   all_A = concat([A_1, A_2, ..., A_G])  // 共 Σ T_i 个值
   mean_A = mean(all_A)
   std_A = std(all_A)
   A_norm = (all_A - mean_A) / (std_A + ε)

5. PPO 更新
   L = -E[min(r(θ) · A_norm, clip(r(θ), 1-ε, 1+ε) · A_norm)]
```

### 4.3 伪代码实现

```python
def compute_milestone_gae(
    trajectories: List[Trajectory],
    milestones: MilestoneTemplate,
    judge_llm: JudgeLLM,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost: float = 0.01,
) -> np.ndarray:
    """
    计算 Milestone-Guided GAE 优势值
    
    Args:
        trajectories: G 条轨迹
        milestones: 里程碑模板
        judge_llm: Judge LLM 接口
        gamma: 折扣因子
        lam: GAE 系数
        cost: 时间成本
    
    Returns:
        归一化后的优势值数组
    """
    all_advantages = []
    
    for traj in trajectories:
        # Step 1: 调用 Judge（每条轨迹一次）
        phis = judge_llm.judge_trajectory(traj, milestones)
        
        T = len(traj)
        deltas = []
        
        # Step 2: 计算 TD Error
        for t in range(T):
            r_t = traj.rewards[t] - cost
            
            # 边界处理（关键设计）
            # 失败/截断时保持当前势能，不跌落到0
            # 成功vs失败的区分仅来自最终奖励 r_T
            if t == T - 1:
                if traj.done and traj.success:
                    phi_next = 1.0        # 成功：未来势能为理想值
                else:
                    phi_next = phis[t]    # 失败/截断：保持当前势能
            else:
                phi_next = phis[t + 1]
            
            delta_t = r_t + gamma * phi_next - phis[t]
            deltas.append(delta_t)
        
        # Step 3: GAE 递归
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * lam * gae
            advantages[t] = gae
        
        all_advantages.extend(advantages)
    
    # Step 4: 全局归一化
    all_advantages = np.array(all_advantages)
    mean_adv = np.mean(all_advantages)
    std_adv = np.std(all_advantages) + 1e-8
    normalized = (all_advantages - mean_adv) / std_adv
    
    return normalized
```

---

## 5. 与 GRPO 的融合

### 5.1 组采样机制

本方案完全兼容 GRPO 的组采样：

```
同一个 Prompt → 生成 G 条轨迹 → 每条轨迹独立计算 Φ 和 A
```

### 5.2 归一化策略对比

| 方法 | 归一化维度 | 说明 |
|------|-----------|------|
| GRPO 原始 | G 个 episode 奖励 | Episode-level |
| 本方案 | Σ T_i 个 step 优势 | Step-level |

**示例**：
```
G = 8, 每条轨迹约 20 步
GRPO: 对 8 个值归一化
本方案: 对 ~160 个值归一化
```

### 5.3 为什么全局归一化更好

1. **更多样本参与**：160 vs 8，方差估计更稳定
2. **step-level 区分**：同一轨迹内的好步和坏步可以区分
3. **长轨迹公平性**：不会因长度不同导致不公平

---

## 6. 工程实现

### 6.1 模块结构

```
rlvmr/
├── milestone/
│   ├── __init__.py
│   ├── extractor.py           # 里程碑提取（调用强模型）
│   ├── templates/             # 里程碑模板库
│   │   ├── alfworld.json
│   │   ├── webshop.json
│   │   └── math.json
│   └── judge.py               # Judge LLM 接口
├── core_milestone_gae.py      # 核心算法实现
└── __init__.py
```

### 6.2 与 agent-verl 框架对接

#### 新增 AdvantageEstimator 类型

在 `verl/trainer/ppo/ray_trainer.py` 中：

```python
class AdvantageEstimator(str, Enum):
    # ... 现有类型
    MilestoneGAE = 'milestone_gae'  # 新增
```

#### compute_advantage 函数扩展

```python
elif adv_estimator == AdvantageEstimator.MilestoneGAE:
    from rlvmr.core_milestone_gae import compute_milestone_gae_advantage
    
    advantages, returns = compute_milestone_gae_advantage(
        batch=data,
        milestones=kwargs.get("milestones"),
        judge_llm=kwargs.get("judge_llm"),
        gamma=gamma,
        lam=lam,
        cost=kwargs.get("cost", 0.01),
    )
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
```

### 6.3 配置文件示例

```yaml
algorithm:
  adv_estimator: milestone_gae
  gamma: 0.99
  
  milestone_gae:
    lambda: 0.95
    cost: 0.01
    judge_llm:
      base_url: "http://localhost:8000/v1"
      model: "qwen2.5-7b-instruct"
    milestone_template: "alfworld"
```

---

## 7. 实验设计

### 7.1 消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| A1 | λ ∈ {0.9, 0.95, 0.99} | 验证信用倒推效果 |
| A2 | c ∈ {0, 0.01, 0.05} | 验证时间成本作用 |
| A3 | 全局归一化 vs 轨迹内归一化 | 验证归一化策略 |
| A4 | 每步调用 vs 每轨迹调用 Judge | 验证效率与质量权衡 |

### 7.2 基线对比

| 基线 | 对比点 |
|------|--------|
| GRPO | Episode-level 信用分配 |
| GiGPO | 环境提供的 step reward |
| GAE + Critic | 训练 Critic 网络 |
| HybridGRPO | Discriminator 奖励 |

### 7.3 评估指标

| 指标 | 说明 |
|------|------|
| **Success Rate** | 任务完成率 |
| **Episode Length** | 完成任务的平均步数 |
| **Sample Efficiency** | 达到目标成功率需要的训练样本数 |
| **Judge Calls** | Judge LLM 调用次数 |
| **Training Time** | 总训练时间 |

### 7.4 预期结果

| 指标 | vs GRPO | vs GiGPO |
|------|---------|----------|
| Success Rate | +5~10% | +3~5% |
| Episode Length | -10~20% | 持平 |
| Sample Efficiency | +30~50% | +10~20% |

---

## 附录

### A. 数学推导：GAE 信用倒推

设：
- 轨迹长度 T = 5
- 只有 t=5 时 r_5 = 1（成功），其他 r_t = 0
- Φ 值：[0.2, 0.2, 0.2, 0.2, 0.4]（M1 在 s1 达成，M2 在 s5 达成）
- γ = 0.99, λ = 0.95, c = 0.01

TD Error 计算：
```
δ_1 = -0.01 + 0.99×0.2 - 0.2 = -0.012
δ_2 = -0.01 + 0.99×0.2 - 0.2 = -0.012
δ_3 = -0.01 + 0.99×0.2 - 0.2 = -0.012
δ_4 = -0.01 + 0.99×0.4 - 0.2 = 0.186
δ_5 = 0.99 + 0.99×1.0 - 0.4 = 1.58
```

GAE 递归：
```
A_5 = 1.58
A_4 = 0.186 + 0.9405×1.58 = 1.67
A_3 = -0.012 + 0.9405×1.67 = 1.56
A_2 = -0.012 + 0.9405×1.56 = 1.46
A_1 = -0.012 + 0.9405×1.46 = 1.36
```

**结论**：所有步骤的 A_t > 0，证明成功信号有效倒推。

### A.2 成功 vs 失败轨迹详细对比

**场景设定**：
- 轨迹 A：15 步成功完成任务
- 轨迹 B：30 步失败（达到 Φ=0.8 后被截断）
- 参数：γ=0.99, λ=0.95, c=0.01

#### 轨迹 A（15步成功）

```
Φ 序列: [0.15, 0.15, 0.15, 0.30, 0.30, 0.45, 0.45, 0.45, 
         0.60, 0.60, 0.80, 0.80, 0.80, 1.00, 1.00]
r 序列: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

最后一步（成功）:
```
δ_15 = r_15 - c + γ × Φ_next - Φ_15
     = 1 - 0.01 + 0.99 × 1.0 - 1.0
     = 0.98  ← 巨大正值！
```

GAE 倒推后：所有 A_t ∈ [0.83, 1.03]，**全部为正**

#### 轨迹 B（30步失败，保持势能设计）

```
Φ 序列: [0.15×5, 0.30×5, 0.45×8, 0.60×6, 0.80×6]
r 序列: [0, 0, ..., 0]  （全部为 0）
```

最后一步（失败/截断，保持势能）:
```
δ_30 = r_30 - c + γ × Φ_30 - Φ_30    ← Φ_next = Φ_30 = 0.80
     = 0 - 0.01 + 0.99 × 0.80 - 0.80
     = -0.01 + 0.792 - 0.80
     = -0.018  ← 普通停滞惩罚，非极端负值！
```

GAE 倒推后：大部分 A_t ∈ [-0.08, +0.15]，**接近零或微正**

#### 为什么这样设计更合理

| 设计 | 轨迹 B 最后步 δ_T | 效果 |
|------|------------------|------|
| 旧方案（Φ跌到0） | -0.81 | 整条轨迹被严重惩罚 |
| 新方案（保持势能） | -0.018 | 只有累积时间成本惩罚 |

**核心洞察**：
- 成功 vs 失败的**唯一区分因素**是最终奖励 r=1
- 势能 Φ 代表"已完成进度"，截断不应归零
- 失败轨迹的前期好步骤（如正确找到物品）仍应获得微正激励

#### 全局归一化后的效果

```
轨迹 A 均值: ≈ 0.94
轨迹 B 均值: ≈ 0.05

归一化后：
轨迹 A: (0.94 - 0.35) / 0.40 ≈ +1.48 (强正激励)
轨迹 B: (0.05 - 0.35) / 0.40 ≈ -0.75 (负激励)
```

**结论**：新设计既保留了成功/失败的明确区分，又避免了对失败轨迹的过度惩罚。


### B. Judge LLM 推荐

| 模型 | 参数量 | 推荐场景 |
|------|--------|----------|
| GPT-4o | - | 里程碑提取（离线） |
| Claude 3.5 Sonnet | - | 里程碑提取（离线） |
| Qwen2.5-7B-Instruct | 7B | 在线判定（本地部署） |
| Llama-3-8B-Instruct | 8B | 在线判定（本地部署） |

### C. 常见问题

**Q: 如果 Judge 判断错误怎么办？**

A: GAE 的 λ 系数提供了平滑作用。单步判断错误会被前后步骤稀释。同时，可以使用多次采样取众数来降低噪声。

**Q: 为什么不直接让 Judge 输出连续分数？**

A: 结构化的里程碑判定将"回归问题"转化为"分类问题"，LLM 在分类任务上表现更稳定、更一致。

**Q: 这个方法适用于所有任务吗？**

A: 最适合有明确子目标结构的任务（如 ALFWorld、WebShop、Math）。对于完全开放式的创意任务，里程碑定义可能较困难。
