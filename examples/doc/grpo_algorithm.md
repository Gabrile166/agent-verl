# GRPO 算法流程与公式说明

本文面向 `examples/grpo_trainer` 的实际实现，目标是把当前仓库中的 GRPO 写成适合后续理论证明的形式。这里讨论的不是论文里的抽象 GRPO，而是“在 verl-agent 的多轮环境交互实现里，GRPO 实际是如何算 advantage 并更新策略的”。

## 1. 对应代码入口

- 训练脚本入口：
  - `examples/grpo_trainer/run_alfworld.sh`
  - `examples/grpo_trainer/run_sciworld.sh`
- 统一训练入口：
  - `verl/trainer/main_ppo.py`
- 训练主循环：
  - `verl/trainer/ppo/ray_trainer.py`
- 多轮 rollout 与批数据整理：
  - `agent_system/multi_turn_rollout/rollout_loop.py`
- Episode reward 写入 token：
  - `agent_system/reward_manager/episode.py`
- GRPO 核心：
  - `verl/trainer/ppo/core_algos.py`

## 2. 算法思想

GRPO 的核心思想是：对同一 prompt 采样一组 trajectories，然后用组内相对回报构造优势，而不是训练一个 critic。

如果 prompt 组 \(u\) 下有 \(G\) 条轨迹，GRPO 不关心“这条轨迹的第几步更好”，而关心：

\[
\text{这条轨迹的最终 outcome，相对于同组其他轨迹更好还是更差。}
\]

因此它本质上是 outcome-level / group-relative 的策略优化方法。

## 3. 数据层级与记号

与 Milestone GAE 一样，一次训练中采用如下记号：

- \(u\)：同一 prompt 组，对应 `uid`
- \(k \in \{1,\dots,G\}\)：组内第 \(k\) 条 trajectory，对应 `traj_uid`
- \(t \in \{1,\dots,T_{u,k}\}\)：trajectory 内第 \(t\) 个环境 step
- \(j\)：第 \(t\) 步 response 中的 token 下标
- \(R^{\mathrm{ep}}_{u,k}\)：轨迹总 episode reward
- \(m_{u,k,t,j}\)：response token mask

需要特别强调当前仓库中的一个实现事实：

- 轨迹会先完整 rollout
- 然后被“按环境 step 展平”为多个训练样本

所以在代码层，GRPO 实际看到的是一批 step-sample，而不是一批完整 episode object。

## 4. 一次训练迭代的执行流程

### 4.1 组采样

当前实现不是靠 `actor_rollout_ref.rollout.n > 1` 做分组采样，而是要求：

\[
\texttt{actor\_rollout\_ref.rollout.n} = 1,
\qquad
\texttt{env.rollout.n} = G.
\]

因此同一 prompt 会在环境侧生成 \(G\) 条并行轨迹。

### 4.2 多轮环境交互

对每个环境 step：

1. actor 根据当前 observation 生成 response；
2. response 解码为 action；
3. action 与环境交互，得到 step reward；
4. step reward 被累加进 trajectory 的总回报。

因此 trajectory 总回报定义为

\[
R^{\mathrm{ep}}_{u,k}
=
\sum_{t=1}^{T_{u,k}} r^{\mathrm{env}}_{u,k,t}.
\]

### 4.3 展平为 step-sample

rollout 完成后，每个有效环境 step 变成一个 batch sample，并都携带同一个 trajectory-level episode reward：

\[
\forall t,\quad
\text{sample}(u,k,t)\ \text{附带}\ R^{\mathrm{ep}}_{u,k}.
\]

### 4.4 token-level score 的构造

`EpisodeRewardManager` 把 `episode_rewards` 写到当前 step response 的最后一个 token 上，因此：

\[
\text{score}_{u,k,t,j} =
\begin{cases}
R^{\mathrm{ep}}_{u,k}, & j = L_{u,k,t} \\
0, & \text{otherwise}
\end{cases}
\]

于是该 step sample 的 token 求和分数是

\[
S_{u,k,t}
:=
\sum_j \text{score}_{u,k,t,j}
=
R^{\mathrm{ep}}_{u,k}.
\]

这说明一个关键事实：

\[
S_{u,k,1} = S_{u,k,2} = \cdots = S_{u,k,T_{u,k}}
=
R^{\mathrm{ep}}_{u,k}.
\]

也就是说，在当前实现里，同一条 trajectory 内每个 step sample 拿到的是同一个 outcome score。

## 5. GRPO 优势函数的实现公式

当前实现入口是 `compute_grpo_outcome_advantage(...)`。

### 5.1 组内分数集合

对固定 prompt 组 \(u\)，收集该组所有 step-sample 的分数：

\[
\mathcal S_u
=
\{S_{u,k,t}\}_{k,t}.
\]

注意这里的默认实现 `compute_mean_std_cross_steps=True`，意味着统计量是按“组内所有 step-sample”算的，而不是仅按 trajectory 算。

因此定义：

\[
\mu_u = \mathrm{mean}(\mathcal S_u), \qquad
\sigma_u = \mathrm{std}(\mathcal S_u).
\]

### 5.2 group-relative scalar advantage

对任意 step sample \((u,k,t)\)，其 GRPO scalar advantage 为

\[
A^{\mathrm{GRPO}}_{u,k,t}
=
\frac{S_{u,k,t} - \mu_u}{\sigma_u + \varepsilon}.
\]

如果关闭按标准差缩放，则变为

\[
A^{\mathrm{GRPO}}_{u,k,t}
=
S_{u,k,t} - \mu_u.
\]

由于

\[
S_{u,k,t} = R^{\mathrm{ep}}_{u,k},
\]

可改写为

\[
A^{\mathrm{GRPO}}_{u,k,t}
=
\frac{R^{\mathrm{ep}}_{u,k} - \mu_u}{\sigma_u + \varepsilon}.
\]

因此同一条 trajectory 内部：

\[
A^{\mathrm{GRPO}}_{u,k,1}
=
A^{\mathrm{GRPO}}_{u,k,2}
=
\cdots
=
A^{\mathrm{GRPO}}_{u,k,T_{u,k}}.
\]

### 5.3 扩展到 token 级别

最终 token-level advantage 为

\[
\hat A^{\mathrm{GRPO}}_{u,k,t,j}
=
A^{\mathrm{GRPO}}_{u,k,t} \cdot m_{u,k,t,j}.
\]

所以对于同一个环境 step，所有有效 response token 共享相同的 scalar advantage。

## 6. PPO 更新目标

虽然 advantage 的来源是组内相对回报，但 actor 更新仍然是 PPO clipped objective。

设 token 级 importance ratio 为

\[
\rho_{u,k,t,j}(\theta)
=
\frac{\pi_\theta(a_{u,k,t,j}\mid \cdot)}{\pi_{\theta_{\mathrm{old}}}(a_{u,k,t,j}\mid \cdot)}.
\]

则策略损失为

\[
\mathcal L_{\mathrm{PPO}}
=
-\mathbb E\left[
\min\Big(
\rho \hat A,\,
\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat A
\Big)
\right].
\]

若开启 actor-side KL loss，还会额外加上与 reference policy 的 KL 正则项。

## 7. 当前实现的理论特征

### 7.1 它没有 critic

当前实现中，GRPO 被明确标记为 `use_critic = False`。因此理论上它更接近：

- 一个只依赖 outcome reward 的组相对 policy gradient 方法
- 而不是 actor-critic 方法

### 7.2 它的信用分配是“trajectory 常数”

对于固定 trajectory \((u,k)\)，GRPO 在当前实现里满足：

\[
\forall t,\quad
A^{\mathrm{GRPO}}_{u,k,t}
=
\text{const w.r.t. } t.
\]

这意味着：

- 它能分辨“哪条轨迹更好”
- 但不能分辨“同一条轨迹里哪一步更好”

### 7.3 长轨迹在组统计中会重复出现更多次

因为默认是对组内所有 step-sample 求均值与方差，所以一条长度为 \(T_{u,k}\) 的 trajectory 会在 \(\mathcal S_u\) 中出现 \(T_{u,k}\) 次同一个分数。

因此组统计实际上是：

\[
\mu_u
=
\frac{\sum_k T_{u,k} R^{\mathrm{ep}}_{u,k}}{\sum_k T_{u,k}}.
\]

这不是“每条 trajectory 等权”，而是“每个 step-sample 等权”。

这个实现细节在后续理论分析里很重要，因为它带来了显式的长度权重。

### 7.4 标准化是 group 内，而不是全 batch

GRPO 的中心化和方差缩放都在 prompt group \(u\) 内完成：

\[
(\mu_u,\sigma_u)
\quad \text{只依赖组 } u \text{ 本身}.
\]

因此不同 prompt 组之间不存在直接的 advantage 共享统计量。

## 8. 可用于理论证明的几个切入点

### 8.1 trajectory-level 常数优势

可以把当前实现形式化为：

\[
\hat A^{\mathrm{GRPO}}_{u,k,t,j}
=
g_u(R^{\mathrm{ep}}_{u,k}) \cdot m_{u,k,t,j},
\]

其中

\[
g_u(x) = \frac{x-\mu_u}{\sigma_u+\varepsilon}.
\]

这使得证明中可以把 GRPO 看成“对每条 trajectory 分配一个常数权重，再广播到该轨迹所有 step token”。

### 8.2 outcome-only 监督

当前实现的外部奖励在 GRPO 分支中只体现为

\[
R^{\mathrm{ep}}_{u,k},
\]

而不保留原始 step reward 的时间结构。因此它是一个典型的 outcome-only estimator。

### 8.3 长度加权效应

由于 group mean/std 默认按 step-sample 统计，理论上可以分析：

1. 长轨迹是否在组基线中占更大权重；
2. 长度与 reward 相关时，是否会诱导优化偏置；
3. 关闭 `norm_adv_by_std_in_grpo` 后，这种偏置如何变化。

## 9. 可直接引用的算法摘要

给定一批 prompt group \(\{u\}\)，当前实现中的 GRPO 可以概括为：

1. 对每个 prompt \(u\) 采样 \(G\) 条 trajectories。
2. 计算每条 trajectory 总回报：

\[
R^{\mathrm{ep}}_{u,k}
=
\sum_t r^{\mathrm{env}}_{u,k,t}.
\]

3. 将每个环境 step 展平成一个样本，并给该样本附上相同的 episode reward。
4. 对每个 step sample，定义

\[
S_{u,k,t} = R^{\mathrm{ep}}_{u,k}.
\]

5. 在 prompt group 内计算

\[
\mu_u = \mathrm{mean}(\{S_{u,k,t}\}_{k,t}),\qquad
\sigma_u = \mathrm{std}(\{S_{u,k,t}\}_{k,t}).
\]

6. 定义 step scalar advantage

\[
A^{\mathrm{GRPO}}_{u,k,t}
=
\frac{S_{u,k,t}-\mu_u}{\sigma_u+\varepsilon}.
\]

7. 将其广播到 token 级别：

\[
\hat A^{\mathrm{GRPO}}_{u,k,t,j}
=
A^{\mathrm{GRPO}}_{u,k,t} \cdot m_{u,k,t,j}.
\]

8. 用 \(\hat A^{\mathrm{GRPO}}_{u,k,t,j}\) 进入 PPO actor 更新。
