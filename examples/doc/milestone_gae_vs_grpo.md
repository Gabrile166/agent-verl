# Milestone GAE 与 GRPO 的统一对照

本文把当前仓库中的 `milestone_gae` 与 `grpo` 放到同一个数学框架里对照，重点不是做经验层面的优劣讨论，而是给后续理论证明准备统一记号、统一数据流和统一的差异点。

## 1. 共同训练骨架

两者共享如下主流程：

1. 从数据集中取一批 prompt。
2. 每个 prompt 在环境侧采样 \(G\) 条并行 trajectories。
3. 每条 trajectory 在环境中进行多轮交互，累计 episode reward。
4. 轨迹被按环境 step 展平成训练样本。
5. 每个 step-sample 的 response 最后一个 token 被赋予 trajectory 的 episode reward。
6. 根据不同 advantage estimator 计算 token-level advantages。
7. 用相同的 PPO actor loss 更新策略。

因此两者的主要区别不在 rollout，不在 PPO，而在第 6 步。

## 2. 统一记号

对任意 prompt 组 \(u\)：

- \(k \in \{1,\dots,G\}\)：组内第 \(k\) 条 trajectory
- \(t \in \{1,\dots,T_{u,k}\}\)：trajectory 内第 \(t\) 个环境 step
- \(j\)：当前 step response 的 token 下标
- \(R^{\mathrm{ep}}_{u,k}\)：trajectory 总回报
- \(m_{u,k,t,j}\)：token mask

若定义 step sample score 为

\[
S_{u,k,t}
=
\sum_j \text{score}_{u,k,t,j},
\]

则当前实现中恒有

\[
S_{u,k,t} = R^{\mathrm{ep}}_{u,k}.
\]

因为 reward manager 把 trajectory 总回报写在当前 response 的最后一个 token 上。

## 3. GRPO 的估计器

### 3.1 定义

GRPO 在 prompt group \(u\) 内计算：

\[
\mu_u = \mathrm{mean}(\{S_{u,k,t}\}_{k,t}),\qquad
\sigma_u = \mathrm{std}(\{S_{u,k,t}\}_{k,t}).
\]

然后定义：

\[
A^{\mathrm{GRPO}}_{u,k,t}
=
\frac{S_{u,k,t}-\mu_u}{\sigma_u+\varepsilon}.
\]

扩展到 token 级：

\[
\hat A^{\mathrm{GRPO}}_{u,k,t,j}
=
A^{\mathrm{GRPO}}_{u,k,t}\cdot m_{u,k,t,j}.
\]

### 3.2 关键性质

因为

\[
S_{u,k,t} = R^{\mathrm{ep}}_{u,k},
\]

所以：

\[
A^{\mathrm{GRPO}}_{u,k,1}
=
\cdots
=
A^{\mathrm{GRPO}}_{u,k,T_{u,k}}.
\]

即：**同一条 trajectory 内所有 step 共享同一个常数优势。**

## 4. Milestone GAE 的估计器

### 4.1 势能函数

对每个状态 \(s_{u,k,t}\)，judge 输出：

\[
\Phi_{u,k,t} := \Phi(s_{u,k,t}) \in [0,1].
\]

### 4.2 奖励压缩

当前实现把 trajectory 总回报只放在最后一步：

\[
r_{u,k,t}
=
\mathbf 1[t=T_{u,k}] \cdot R^{\mathrm{ep}}_{u,k}.
\]

再减去每步时间成本 \(c\)：

\[
\tilde r_{u,k,t} = r_{u,k,t} - c.
\]

### 4.3 终止边界

\[
\Phi_{u,k,T+1}
=
\begin{cases}
1.0, & \text{成功终止} \\
\Phi_{u,k,T}, & \text{失败或截断}
\end{cases}
\]

### 4.4 TD 与 GAE

\[
\delta_{u,k,t}
=
\tilde r_{u,k,t}
+
\gamma \Phi_{u,k,t+1}
-
\Phi_{u,k,t},
\]

\[
A^{\mathrm{MGAE}}_{u,k,t}
=
\delta_{u,k,t}
+
\gamma\lambda A^{\mathrm{MGAE}}_{u,k,t+1}.
\]

然后在整个 batch 的所有有效 step 上做全局标准化：

\[
\hat A^{\mathrm{MGAE}}_{u,k,t}
=
\frac{A^{\mathrm{MGAE}}_{u,k,t}-\mu_A}{\sigma_A+\varepsilon}.
\]

再扩展到 token 级：

\[
\hat A^{\mathrm{MGAE}}_{u,k,t,j}
=
\hat A^{\mathrm{MGAE}}_{u,k,t}\cdot m_{u,k,t,j}.
\]

### 4.5 关键性质

与 GRPO 不同，Milestone GAE 一般满足：

\[
A^{\mathrm{MGAE}}_{u,k,t_1}
\neq
A^{\mathrm{MGAE}}_{u,k,t_2}
\quad \text{当 } t_1 \neq t_2.
\]

即：**同一条 trajectory 内，不同步骤可以拿到不同优势。**

## 5. 两者的统一写法

如果写成同一个抽象形式：

\[
\hat A_{u,k,t,j}
=
f_{u,k,t}(\text{trajectory data}) \cdot m_{u,k,t,j},
\]

那么：

### 5.1 GRPO

\[
f^{\mathrm{GRPO}}_{u,k,t}
=
g_u(R^{\mathrm{ep}}_{u,k}),
\]

其中 \(g_u\) 是组内中心化和标准化算子。

也就是说，GRPO 的 \(f\) 不显式依赖 \(t\)。

### 5.2 Milestone GAE

\[
f^{\mathrm{MGAE}}_{u,k,t}
=
h(\Phi_{u,k,1:T_{u,k}}, R^{\mathrm{ep}}_{u,k}, S_{u,k}, t),
\]

其中 \(h\) 是由 TD + GAE 递推定义的时序算子。

也就是说，Milestone GAE 的 \(f\) 显式依赖 \(t\)。

## 6. 最核心的理论差异

### 6.1 信用分配粒度

- GRPO：trajectory-level / outcome-level
- Milestone GAE：step-level

可形式化为：

\[
\frac{\partial A^{\mathrm{GRPO}}_{u,k,t}}{\partial t} = 0
\quad \text{(在当前实现意义下)}
\]

而

\[
\frac{\partial A^{\mathrm{MGAE}}_{u,k,t}}{\partial t}
\neq 0
\]

通常成立。

### 6.2 baseline 来源

- GRPO 的 baseline 是 prompt group 内的相对均值 \(\mu_u\)
- Milestone GAE 的 baseline 是状态势能 \(\Phi(s_t)\)

也就是：

\[
\text{GRPO: relative-to-group}
\]

对比

\[
\text{Milestone GAE: relative-to-potential-progress}.
\]

### 6.3 归一化维度

GRPO 使用 group 内归一化：

\[
\mu_u,\sigma_u \text{ 只在组 } u \text{ 内定义}.
\]

Milestone GAE 使用全 batch 的 step-level 归一化：

\[
\mu_A,\sigma_A \text{ 在当前 batch 的所有有效 step 上定义}.
\]

### 6.4 对中间步骤的区分能力

GRPO 无法区分：

- “前 10 步都很好，最后一步失败”
- “前 10 步一直乱走，最后一步也失败”

只要两条 trajectory 的 \(R^{\mathrm{ep}}\) 接近，它们在同组里得到的优势就接近。

Milestone GAE 则可以通过 \(\Phi_{u,k,t}\) 的演化区分：

\[
\Phi_{u,k,t+1} - \Phi_{u,k,t}
\]

是否表示任务进度推进。

### 6.5 对失败轨迹的建模

GRPO 对失败轨迹的处理主要来自组内相对比较。

Milestone GAE 则多了一层终止边界机制：

\[
\Phi_{T+1} = \Phi_T
\]

使得失败轨迹不会因为未成功而把已有进度完全抹掉。

## 7. 证明方向建议

如果后续要做理论证明，我建议把两者都写成“同一 PPO 框架下的不同 advantage estimator”，然后分三层证明。

### 7.1 共同层

证明两者都可以写成：

\[
\nabla_\theta \mathcal L(\theta)
\approx
-\mathbb E\left[
\hat A_{u,k,t,j} \nabla_\theta \log \pi_\theta(a_{u,k,t,j}|\cdot)
\right]
\]

再加 PPO clipping 的偏差控制。

### 7.2 GRPO 特有层

证明重点放在：

1. group-relative baseline 的无偏性或偏差性
2. 当前实现中“step-sample 等权”导致的长度权重效应
3. outcome-only estimator 的方差性质

### 7.3 Milestone GAE 特有层

证明重点放在：

1. 若 \(\Phi\) 逼近某种理想 progress potential，则 GAE 递推如何改善信用分配
2. judge 误差如何影响 TD 误差：

\[
e_{u,k,t} = \Phi_{u,k,t} - \Phi^*_{u,k,t}
\]

3. 全局 step-level 标准化对方差与尺度的影响

## 8. 一句话总结

如果把两者压成一句话：

- **GRPO**：先看整条 trajectory 最终结果，再把这个结果广播回整条 trajectory 的所有 steps。
- **Milestone GAE**：先为每个 step 估计任务进度，再把终局 reward 和进度差分通过 GAE 递推传播回前面的 steps。

因此，从理论证明的角度看，二者最本质的差别是：

\[
\text{GRPO 是 outcome-relative estimator，}
\]

\[
\text{Milestone GAE 是 potential-guided temporal estimator。}
\]
