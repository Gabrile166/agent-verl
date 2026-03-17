# Milestone GAE 数学可行性证明草稿

本文的目标不是直接给出“已经完全严格完成”的最终证明，而是回答一个更关键的问题：**Milestone GAE 这种 advantage estimator 在数学上能否被放进一个自洽的强化学习框架里，并在什么假设下证明它是可行的。**

我会把“可行性”拆成四层：

1. 它是否退化到经典 GAE。
2. 它是否等价于某个明确的 surrogate objective。
3. 当 judge 有误差时，这个 estimator 的偏差能否被控制。
4. 当前实现里哪些部分需要额外假设，哪些部分不能直接套经典定理。

## 1. 问题设定

考虑一个有限时域的轨迹分布

\[
\tau = (s_1,a_1,r_1,\dots,s_T,a_T,r_T),
\qquad \tau \sim p_\theta(\tau),
\]

其中策略为 \(\pi_\theta\)。

定义原始目标函数为

\[
J(\theta) := \mathbb E_{\tau \sim p_\theta}[G(\tau)],
\qquad
G(\tau) := \sum_{t=1}^{T} \gamma^{t-1} r_t.
\]

Milestone GAE 引入一个外部 judge 给出的 progress potential

\[
\Phi_t := \Phi(h_t),
\]

其中 \(h_t\) 可以是状态 \(s_t\)，也可以是历史前缀 \((s_{1:t},a_{1:t-1})\)。

在当前实现里，还引入每步时间成本 \(c>0\)，并定义

\[
\tilde r_t := r_t - c.
\]

然后构造 Milestone TD 误差

\[
\delta_t^{\Phi}
:=
\tilde r_t + \gamma \Phi_{t+1} - \Phi_t.
\]

并进一步定义 Milestone GAE 优势

\[
A_t^{\Phi}
:=
\delta_t^{\Phi} + \gamma \lambda A_{t+1}^{\Phi}.
\]

## 2. 第一层证明：它在理想情形下退化到经典 GAE

这是最重要的一步，因为它说明 Milestone GAE 至少不是“凭空发明的无根估计器”，而是标准 GAE 的一个广义化。

### 定理 1：理想 judge 情形下，Milestone GAE 退化为经典 GAE

假设存在某个精确 value function \(V^\pi(h_t)\)，并且 judge 满足

\[
\Phi_t = V^\pi(h_t), \qquad \Phi_{t+1} = V^\pi(h_{t+1}).
\]

则 Milestone TD 误差满足

\[
\delta_t^{\Phi}
=
\tilde r_t + \gamma V^\pi(h_{t+1}) - V^\pi(h_t),
\]

这正是标准 GAE 使用的 TD 误差。于是递推得到的 \(A_t^{\Phi}\) 与经典 GAE 完全相同。

### 证明

把 \(\Phi_t = V^\pi(h_t)\) 直接代入即可：

\[
\delta_t^{\Phi}
=
\tilde r_t + \gamma \Phi_{t+1} - \Phi_t
=
\tilde r_t + \gamma V^\pi(h_{t+1}) - V^\pi(h_t).
\]

然后对同一个 \(\delta_t\) 使用同一个 GAE 递推

\[
A_t = \delta_t + \gamma \lambda A_{t+1},
\]

得到的优势序列必然一致。

### 含义

这个定理说明：

- Milestone GAE 在理想 judge 下不是新目标，而是经典 GAE。
- 因此它在数学上至少包含了一个“经典可证明特例”。

这已经足够支持“该方法在原理上是可行的”。

## 3. 第二层证明：它优化的是一个明确的 surrogate objective

即使 \(\Phi\) 不是精确 value，它也不是乱来的。它对应一个清晰的 reward shaping 目标。

### 命题 2：Milestone TD 对应一个显式的 shaping 回报

定义 shaped reward

\[
\hat r_t := r_t - c + \gamma \Phi_{t+1} - \Phi_t.
\]

则对应的 discounted shaped return 为

\[
\hat G(\tau)
:=
\sum_{t=1}^{T} \gamma^{t-1} \hat r_t.
\]

该回报可化简为

\[
\hat G(\tau)
=
\sum_{t=1}^{T} \gamma^{t-1}(r_t-c)
-
\Phi_1
+
\gamma^T \Phi_{T+1}.
\]

### 证明

直接展开：

\[
\hat G(\tau)
=
\sum_{t=1}^{T} \gamma^{t-1}(r_t-c)
+
\sum_{t=1}^{T} \gamma^t \Phi_{t+1}
-
\sum_{t=1}^{T} \gamma^{t-1} \Phi_t.
\]

后两项望远镜消去，得到

\[
\sum_{t=1}^{T} \gamma^t \Phi_{t+1}
-
\sum_{t=1}^{T} \gamma^{t-1} \Phi_t
=
-
\Phi_1 + \gamma^T \Phi_{T+1}.
\]

于是

\[
\hat G(\tau)
=
\sum_{t=1}^{T} \gamma^{t-1}(r_t-c)
-
\Phi_1
+
\gamma^T \Phi_{T+1}.
\]

证毕。

### 含义

这说明 Milestone GAE 至少在优化一个**明确写得出来的 surrogate objective**：

\[
J_{\Phi}(\theta) := \mathbb E_{\tau \sim p_\theta}[\hat G(\tau)].
\]

因此它不是“没有目标函数”的启发式方法，而是一个有清晰目标的 shaped policy optimization 方法。

## 4. 第三层证明：近似 judge 只带来可控偏差

真正重要的问题不是理想情形，而是 judge 不精确时还能否工作。

### 假设 1：judge 逼近某个理想 value-like 函数

假设存在某个理想目标函数 \(V_t^*\)，并有统一误差界

\[
|\Phi_t - V_t^*| \le \varepsilon_{\Phi},
\qquad
|\Phi_{T+1} - V_{T+1}^*| \le \varepsilon_{\mathrm{term}}.
\]

定义理想 TD 误差

\[
\delta_t^*
:=
\tilde r_t + \gamma V_{t+1}^* - V_t^*.
\]

### 定理 3：Milestone TD 误差偏差有上界

对非终止步，若 \(t<T\)，则

\[
|\delta_t^{\Phi} - \delta_t^*|
\le
(1+\gamma)\varepsilon_{\Phi}.
\]

对终止步，若 \(t=T\)，则

\[
|\delta_T^{\Phi} - \delta_T^*|
\le
\varepsilon_{\Phi} + \gamma \varepsilon_{\mathrm{term}}.
\]

### 证明

对非终止步：

\[
\delta_t^{\Phi} - \delta_t^*
=
\gamma(\Phi_{t+1}-V_{t+1}^*) - (\Phi_t - V_t^*).
\]

由三角不等式可得

\[
|\delta_t^{\Phi} - \delta_t^*|
\le
\gamma |\Phi_{t+1}-V_{t+1}^*| + |\Phi_t - V_t^*|
\le
(1+\gamma)\varepsilon_{\Phi}.
\]

终止步同理，只需把下一时刻误差改成终止边界误差 \(\varepsilon_{\mathrm{term}}\)。证毕。

### 定理 4：Milestone GAE 优势偏差也有上界

设

\[
\eta := \max\{(1+\gamma)\varepsilon_{\Phi},\ \varepsilon_{\Phi}+\gamma\varepsilon_{\mathrm{term}}\}.
\]

定义理想优势

\[
A_t^* := \delta_t^* + \gamma\lambda A_{t+1}^*.
\]

则对任意 \(t\)，有

\[
|A_t^{\Phi} - A_t^*|
\le
\frac{\eta}{1-\gamma\lambda}.
\]

### 证明

记

\[
\Delta_t := A_t^{\Phi} - A_t^*.
\]

则

\[
\Delta_t
=
(\delta_t^{\Phi} - \delta_t^*) + \gamma\lambda \Delta_{t+1}.
\]

取绝对值并递推：

\[
|\Delta_t|
\le
\eta + \gamma\lambda |\Delta_{t+1}|.
\]

反复展开得到几何级数：

\[
|\Delta_t|
\le
\eta \sum_{l=0}^{\infty} (\gamma\lambda)^l
=
\frac{\eta}{1-\gamma\lambda}.
\]

证毕。

### 含义

这一定理非常关键，因为它说明：

- judge 不需要完全正确；
- 只要它逼近某个“合理的 value-like 目标”，Milestone GAE 的偏差就是有界的；
- 且误差放大因子正好是经典 GAE 常见的 \((1-\gamma\lambda)^{-1}\)。

所以从数学上看，它是一个**有界偏差的近似优势估计器**。

## 5. 第四层证明：方向保持条件

有了偏差上界以后，可以进一步证明：当理想优势的 margin 足够大时，Milestone GAE 不会把更新方向搞反。

### 推论 5：在足够大 margin 下，优势符号保持不变

若某一步满足

\[
|A_t^*| > \frac{\eta}{1-\gamma\lambda},
\]

则

\[
\mathrm{sign}(A_t^{\Phi}) = \mathrm{sign}(A_t^*).
\]

### 证明

由定理 4，

\[
|A_t^{\Phi} - A_t^*| 
\le 
\frac{\eta}{1-\gamma\lambda}.
\]

如果理想优势的绝对值严格大于这个误差半径，则扰动不可能越过 0，因此符号保持不变。证毕。

### 含义

这给出了一个很强的结论：

- 在“强正优势”或“强负优势”步骤上，Milestone GAE 会保持正确的更新方向；
- 它真正不稳定的，只会是接近 0 的模糊步骤；
- 这和很多近似 RL estimator 的行为是一致的。

## 6. 有界性与优化稳定性

为了证明 PPO 更新本身是良定义的，还可以给出 advantage 的有界性。

### 假设 2：奖励与势能有界

设

\[
|r_t| \le R_{\max},
\qquad
0 \le \Phi_t \le 1.
\]

则 TD 误差满足

\[
|\delta_t^{\Phi}|
\le
R_{\max} + c + \gamma + 1.
\]

从而优势满足

\[
|A_t^{\Phi}|
\le
\frac{R_{\max}+c+\gamma+1}{1-\gamma\lambda}.
\]

### 证明

由定义：

\[
|\delta_t^{\Phi}|
=
|r_t - c + \gamma\Phi_{t+1} - \Phi_t|
\le
|r_t| + c + \gamma|\Phi_{t+1}| + |\Phi_t|.
\]

再由有界性得

\[
|\delta_t^{\Phi}|
\le
R_{\max} + c + \gamma + 1.
\]

接着用 GAE 递推作几何级数求和即可得到优势界。证毕。

### 含义

这说明 Milestone GAE 送入 PPO 前的 raw advantage 至少是**有界的**，因此不会在数学上产生发散型无穷大梯度项。

## 7. 对当前实现必须诚实面对的三个理论难点

上面的结论已经足够证明“可行性”，但要写成严格论文证明，还必须把下面三件事说清楚。

### 7.1 judge 当前是离线全轨迹判定，不是严格在线 value function

当前实现里，judge 一次性看完整条 trajectory，再输出每个 step 的 \(\Phi_t\)。

这意味着严格来说，\(\Phi_t\) 可能依赖未来信息，而不仅依赖前缀 \(h_t\)。

因此：

- 如果你想得到最干净的 RL 定理，最好假设 judge 是 prefix-causal 的，即 \(\Phi_t = \Phi(h_t)\)。
- 如果保持现实现状，则更合适的说法是：Milestone GAE 是一个**基于轨迹后验评分的 surrogate advantage estimator**。

这仍然可以做证明，但不应直接宣称“它就是标准 MDP 下的 value function”。

### 7.2 当前终止边界不是经典 PBRS 的标准形式

经典 potential-based reward shaping 一般使用固定终端 potential，而当前实现是：

- 成功：\(\Phi_{T+1}=1\)
- 失败：\(\Phi_{T+1}=\Phi_T\)

因此它不完全等同于教科书里的 PBRS 定理条件。

不过第 3 节已经说明，这不会导致“没有目标函数”，因为它依然对应一个明确定义的 surrogate return：

\[
\hat G(\tau)
=
\sum_{t=1}^{T} \gamma^{t-1}(r_t-c) - \Phi_1 + \gamma^T \Phi_{T+1}.
\]

所以这里更准确的说法是：

- 它未必继承经典 PBRS 的“最优策略严格不变性”定理；
- 但它明确优化了一个带 terminal bonus 的 surrogate objective。

### 7.3 全局标准化最好单独分析，不要混进主定理

当前实现会对 raw advantage 做 batch 内标准化。这个操作在工程上有用，但理论上会引入 batch-coupling。

因此推荐的证明顺序是：

1. 先对未标准化的 \(A_t^{\Phi}\) 做主定理。
2. 再把标准化看成一个额外的优化启发式，单独分析它对尺度和方差的影响。

否则主定理会被不必要地搞复杂。

## 8. 最推荐的正式证明路线

如果你后面真的要把这套方法写成论文级别证明，我建议按下面三步走。

### 路线 A：理想化定理

先定义一个理想版本：

- judge 是 prefix-causal 的；
- \(\Phi\) 精确等于某个 value-like progress function；
- 不做 batch 标准化。

在这个版本下，定理 1 直接说明它退化为经典 GAE。

这是你的“存在性证明”。

### 路线 B：近似误差定理

再加上 judge 逼近误差，使用定理 3 和定理 4 给出偏差界：

\[
|A_t^{\Phi} - A_t^*| \le \frac{\eta}{1-\gamma\lambda}.
\]

这是你的“鲁棒性证明”。

### 路线 C：当前实现解释

最后再解释当前实现与理想版本的差别：

- 全轨迹 judge
- 非标准终止边界
- batch 标准化

并把它们放进“理论近似 + 工程实现”的框架里，而不是硬套最经典的 RL 教科书表述。

## 9. 可以直接放进文档或论文的核心结论

如果要把上面的内容压成一句正式表述，我建议你用下面这段：

> Milestone GAE can be viewed as a generalized advantage estimator built from an external progress potential \(\Phi\). In the ideal case where \(\Phi\) coincides with the exact value-like progress function, Milestone GAE reduces exactly to classical GAE. In the approximate case, if \(\Phi\) is uniformly close to the target value-like function, then both its TD errors and GAE advantages incur only bounded bias, with error amplification factor \((1-\gamma\lambda)^{-1}\). Therefore, Milestone GAE is mathematically feasible as a bounded-bias surrogate advantage estimator for PPO-style policy optimization.

## 10. 结论

所以，严格地说，你现在已经可以证明下面三件事：

1. **存在经典特例**：Milestone GAE 在理想 judge 下退化为标准 GAE。
2. **存在明确目标**：它对应一个写得出的 shaped surrogate objective，而不是无目标启发式。
3. **存在误差上界**：judge 近似误差只会带来有界的 advantage 偏差。

这三点合在一起，已经足以支持“Milestone GAE 在数学上是可行的”。

如果你愿意，下一步最值得做的是把这份草稿继续升级成“定义 - 引理 - 定理 - 推论”的正式证明版本，并把当前实现中的 `success=1 / failure=phi_T` 终止边界单独抽成一个定理来分析。
