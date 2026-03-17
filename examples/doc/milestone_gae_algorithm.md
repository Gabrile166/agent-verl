# Milestone GAE 算法流程与公式说明

本文面向 `examples/milestone_gae_trainer` 的实际实现，目标是把“脚本执行时到底发生了什么”写成适合后续理论证明的形式。文中默认以多轮 agent-environment 交互场景为背景，尤其对应 ALFWorld / SciWorld 这类任务。

## 1. 对应代码入口

- 训练脚本入口：
  - `examples/milestone_gae_trainer/run_alfworld.sh`
  - `examples/milestone_gae_trainer/run_sciworld.sh`
- 统一训练入口：
  - `verl/trainer/main_ppo.py`
- 训练主循环：
  - `verl/trainer/ppo/ray_trainer.py`
- 多轮 rollout 与批数据整理：
  - `agent_system/multi_turn_rollout/rollout_loop.py`
- Episode reward 写入 token：
  - `agent_system/reward_manager/episode.py`
- Milestone GAE 核心：
  - `rlvmr/core_milestone_gae.py`
- Milestone 生成与判定：
  - `rlvmr/milestone/generator.py`
  - `rlvmr/milestone/judge.py`
  - `rlvmr/pipeline_data.py`

## 2. 整体思想

Milestone GAE 的核心不是训练一个显式 critic，而是让 LLM judge 给出每个环境 step 的“任务进度势能”

\[
\Phi(s_t) \in [0, 1],
\]

然后把它当作 value-like 信号，构造 TD 误差和 GAE 优势。

与标准 actor-critic GAE 相比，它的替换关系是：

\[
V_\psi(s_t) \Longrightarrow \Phi(s_t).
\]

因此它仍然保留了 GAE 的“信用向前回传”机制，但不需要单独训练 critic 网络。

## 3. 数据层级与记号

一次训练迭代中，给定一个 prompt 组 \(u\)，环境会采样 \(G\) 条 rollout trajectory。记：

- \(u\)：同一组 prompt 的组标识，对应代码里的 `uid`
- \(k \in \{1,\dots,G\}\)：组内第 \(k\) 条轨迹，对应 `traj_uid`
- \(t \in \{1,\dots,T_{u,k}\}\)：轨迹内第 \(t\) 个环境 step
- \(j\)：当前 step 响应中的 token 下标
- \(s_{u,k,t}\)：第 \(t\) 步环境状态
- \(a_{u,k,t}\)：第 \(t\) 步模型产生的 action
- \(R^{\mathrm{ep}}_{u,k}\)：轨迹总 episode reward
- \(S_{u,k}\)：轨迹是否成功
- \(\Phi_{u,k,t} := \Phi(s_{u,k,t})\)：judge 输出的 step 势能
- \(m_{u,k,t,j} \in \{0,1\}\)：response token mask

实现上，`rollout_loop.py` 会先收集完整轨迹，再把“每个有效环境 step”展平成一个 batch sample。所以后面所有 actor 更新，实质上都是在 step-sample 粒度上进行。

## 4. 一次训练迭代的执行流程

### 4.1 组采样

`main_ppo.py` 明确要求：

\[
\texttt{actor\_rollout\_ref.rollout.n} = 1,
\]

而真正的组采样来自

\[
\texttt{env.rollout.n} = G.
\]

也就是说，同一个 prompt 会在环境侧复制成 \(G\) 条并行轨迹。

### 4.2 多轮 agent-environment 交互

每个环境 step 发生如下过程：

1. 当前 observation 与历史被编码成 prompt。
2. actor 生成一个 response，解码成 action。
3. action 送入环境，环境返回：
   - 新 observation
   - step reward
   - done
   - info
4. 累积 episode reward 与 episode length。

定义原始环境 step reward 为 \(r^{\mathrm{env}}_{u,k,t}\)，则 rollout 结束后：

\[
R^{\mathrm{ep}}_{u,k}
=
\sum_{t=1}^{T_{u,k}} r^{\mathrm{env}}_{u,k,t}.
\]

### 4.3 展平为 step-sample batch

轨迹收集结束后，每个有效 step 会被展平成一个训练样本，并附上：

- `episode_rewards = R^{\mathrm{ep}}_{u,k}`
- `success = S_{u,k}`
- `uid = u`
- `traj_uid = (u,k)`

这一步非常关键，因为后面的 advantage 计算虽然用到了完整 trajectory 信息，但 actor 更新本身是在“单个环境 step 对应的一次 action response”上做的。

### 4.4 Episode reward 写到最后一个 token

`EpisodeRewardManager` 会把整条 trajectory 的总 reward 只写到当前 response 的最后一个 token 上。

如果当前 step response 长度为 \(L_{u,k,t}\)，则 token-level score 形如：

\[
\text{score}_{u,k,t,j} =
\begin{cases}
R^{\mathrm{ep}}_{u,k}, & j = L_{u,k,t} \\
0, & \text{otherwise}
\end{cases}
\]

因此：

\[
\sum_j \text{score}_{u,k,t,j} = R^{\mathrm{ep}}_{u,k}.
\]

## 5. Milestone 生成与轨迹判定

Milestone GAE 比 GRPO 多出一个“两层结构化中间层”。

### 5.1 Query 层

对每个唯一 prompt 组 \(u\)，构造一个 `QueryRecord`，包含：

- task description
- expert trajectory
- milestones

如果启用 milestone generator，则根据 task 与 expert trajectory 生成 milestones：

\[
M_u = \{M_{u,1}, \dots, M_{u,m_u}\},
\]

其中每个 milestone 具有单调递增的势能值

\[
0 < \phi_{u,1} < \phi_{u,2} < \cdots < \phi_{u,m_u} = 1.
\]

### 5.2 Trajectory 层

对每条具体轨迹 \((u,k)\)，构造 `TrajectoryRecord`，包含：

- policy trajectory
- episode reward \(R^{\mathrm{ep}}_{u,k}\)
- success \(S_{u,k}\)
- judge 输出的 \(\Phi_{u,k,1:T_{u,k}}\)

### 5.3 Judge 的输出

judge 是“按整条 trajectory 调一次 LLM”，不是每个 step 调一次。它返回：

\[
\Phi_{u,k,1}, \Phi_{u,k,2}, \dots, \Phi_{u,k,T_{u,k}}.
\]

可以把它理解成：

\[
\Phi_{u,k,t} = \text{Judge}(s_{u,k,1:t}, a_{u,k,1:t}, M_u).
\]

注意：在当前实现里，judge 的 `final_success` 只是附加信息；真正用于优势计算的成功标记仍然来自环境返回的 `success`。

## 6. Milestone GAE 的数学定义

### 6.1 奖励压缩

当前实现并不直接使用环境逐步 reward 序列，而是把整个 episode 总 reward 压到最后一步：

\[
r_{u,k,t} =
\begin{cases}
0, & t < T_{u,k} \\
R^{\mathrm{ep}}_{u,k}, & t = T_{u,k}
\end{cases}
\]

再引入每步时间成本 \(c > 0\)，定义修正后的即时奖励：

\[
\tilde r_{u,k,t} = r_{u,k,t} - c.
\]

### 6.2 终止边界条件

当前实现的边界条件是：

\[
\Phi_{u,k,T+1} =
\begin{cases}
1.0, & \text{如果 done 且 success} \\
\Phi_{u,k,T}, & \text{如果失败或截断}
\end{cases}
\]

这意味着：

- 成功轨迹被视为到达“理想终点”
- 失败轨迹不会把势能突然打回 0，而是保留当前进度

### 6.3 TD 误差

Milestone GAE 的 TD 误差定义为

\[
\delta_{u,k,t}
=
\tilde r_{u,k,t}
+
\gamma \Phi_{u,k,t+1}
-
\Phi_{u,k,t}.
\]

写展开即

\[
\delta_{u,k,t}
=
r_{u,k,t} - c + \gamma \Phi_{u,k,t+1} - \Phi_{u,k,t}.
\]

### 6.4 GAE 递推

定义 step-level advantage：

\[
A_{u,k,t}
=
\delta_{u,k,t}
+
\gamma \lambda A_{u,k,t+1}.
\]

边界递推写成：

\[
A_{u,k,T} = \delta_{u,k,T},
\]

\[
A_{u,k,t}
=
\delta_{u,k,t}
+
\gamma \lambda A_{u,k,t+1},
\quad t=T-1,\dots,1.
\]

### 6.5 批内全局标准化

把当前 batch 所有有效 step 的 advantage 拼接起来：

\[
\mathcal A = \{A_{u,k,t}\}_{\text{all valid steps}}.
\]

记

\[
\mu_A = \mathrm{mean}(\mathcal A), \qquad
\sigma_A = \mathrm{std}(\mathcal A).
\]

则归一化 advantage 为

\[
\hat A_{u,k,t}
=
\frac{A_{u,k,t} - \mu_A}{\sigma_A + \varepsilon}.
\]

这一步是全 batch 的 step-level 标准化，不是 group 内标准化。

### 6.6 扩展到 token 级别

由于 actor 更新作用在当前 response token 上，step-level advantage 会扩展为：

\[
\hat A_{u,k,t,j}^{\mathrm{token}}
=
\hat A_{u,k,t} \cdot m_{u,k,t,j}.
\]

因此同一个环境 step 内部，所有有效 token 共享同一个 step advantage。

## 7. PPO 更新目标

虽然 advantage 的构造变了，但最终 actor 仍然用 PPO clipped objective 更新。

记当前策略与旧策略的 token 级比值为

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

如果开启 actor-side KL loss，则总损失还会加上与 reference policy 的 KL 正则项。

## 8. 理论上最值得关注的性质

### 8.1 它不是标准 critic-based GAE

这里没有训练 \(V_\psi\)，而是用外部 judge 给出 \(\Phi\)。因此理论证明时更适合把它看成：

- 一个“外源 potential estimator”
- 一个“无 critic 的 GAE-like policy gradient estimator”

### 8.2 Step-level 信用分配来自两部分

\[
A_{u,k,t}
\text{ 的来源 }
=
\text{终局 reward 信号}
+
\text{势能差分信号}.
\]

其中：

- 终局 reward 决定成功/失败的最终方向
- 势能差分决定中间步骤的进度性信用分配

### 8.3 失败轨迹不会被极端打压

因为失败时设置

\[
\Phi_{T+1} = \Phi_T,
\]

所以最后一步的 TD 误差变成

\[
\delta_T = R^{\mathrm{ep}}_T - c + (\gamma - 1)\Phi_T.
\]

若失败时 \(R^{\mathrm{ep}}_T = 0\)，则

\[
\delta_T = -c - (1-\gamma)\Phi_T,
\]

通常是温和负值，而不是把整条轨迹的前期有效进展全部清零。

### 8.4 适合证明的几个方向

后续如果要做理论证明，建议优先从下面几个点入手：

1. 证明当 judge 完全准确时，\(\Phi\) 作为 potential estimator 能提供更细粒度的信用分配。
2. 证明在固定 \(\Phi\) 条件下，Milestone GAE 仍是一个 PPO-style 一阶更新目标。
3. 分析全局 step-level 标准化对 estimator 方差的影响。
4. 分析 judge 误差

\[
e_{u,k,t} = \Phi_{u,k,t} - \Phi^*_{u,k,t}
\]

如何通过 GAE 递推被传播与平滑。

## 9. 可直接引用的算法摘要

给定一批 trajectory \(\{(u,k)\}\)，Milestone GAE 的实现可以概括为：

1. 用 generator 为每个 query \(u\) 生成 milestones \(M_u\)。
2. 用 judge 为每条 trajectory 生成 \(\Phi_{u,k,1:T_{u,k}}\)。
3. 构造压缩后的 episode reward：

\[
r_{u,k,t} = \mathbf 1[t=T_{u,k}] \cdot R^{\mathrm{ep}}_{u,k}.
\]

4. 计算 TD 误差：

\[
\delta_{u,k,t}
=
r_{u,k,t} - c + \gamma \Phi_{u,k,t+1} - \Phi_{u,k,t}.
\]

5. 计算 GAE：

\[
A_{u,k,t}
=
\delta_{u,k,t} + \gamma \lambda A_{u,k,t+1}.
\]

6. 对所有有效 step 做全局标准化得到 \(\hat A_{u,k,t}\)。
7. 将 \(\hat A_{u,k,t}\) 扩展到 token 级别，送入 PPO actor 更新。
