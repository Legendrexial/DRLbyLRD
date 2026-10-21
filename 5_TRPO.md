这两章介绍最为常用的 PPO 算法，由于它使用广泛，非常重要。要彻底理解 PPO，需要首先理解 TRPO 算法，它们是一类算法，优化目标与前面的策略梯度算法有明显不同。

#### TRPO 优化目标的推导

TRPO 算法的全称为 Trusted Region Policy Optimization （信任区域策略优化）。

> 这一节的推导难度较大，细节很多，但推导过程却能把前面所有的内容串起来，值得仔细研究

> A2C 算法是非常简单、直观的 Actor-Critic 算法，但是缺点是每步梯度下降的更新步长主要由学习率决定，在深度神经网络的训练中容易造成策略单次更新幅度过大而导致性能显著下降，造成训练不稳定。TRPO 从一个新的切入点出发导出了一个更加容易约束的更新目标，利用约束条件控制更新幅度，保证更新的稳定性。

TRPO 优化目标的出发点和 Policy Gradient 类似，仍然是最大化所有状态上的 state value 的期望：
$$
J(\theta)=\mathbb E_{s}[v_{\pi_\theta}(s)] \label{obj}
$$

但注意TRPO的优化目标和 Policy Gradient 算法不完全一致，区别在于状态 $s$ 的采样分布，PG 要求 $s\sim\rho_\pi(s)$ ，这样才导出了 $q_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)$ 的梯度形式；但 TRPO 要求对 $s$ 的采样分布和 $\pi$ 无关，这样才能继续构造。

> 这种要求是合法的，因为在 Policy Gradient 章节也提到过，$J(\theta)$ 状态采样分布 $E_{s\sim\rho_\pi(s)}$ 其实是任意的。

> PG 利用梯度上升直接最大化 $J(\theta)=\mathbb E_{s\sim\rho_{\pi}(s)}[v_{\pi}(s)]$ 。如果用微分去理解这件事，$J(\theta')-J(\theta)\approx\nabla_\theta J(\theta)(\theta'-\theta)>0$ ，那么 PG 的梯度上升可以被解释为求解什么样的 $\theta'$ 能够使 $J(\theta')-J(\theta)>0$ 。但与其绕一步，把这个差值用梯度来表示，再去算梯度，倒不如直接推导 $J(\theta')-J(\theta)$ 的具体形式，这就是 TRPO 的出发点。

那从 $\theta$ 出发，考虑什么样的 $\theta'$ 能保证 $J(\theta')-J(\theta)>0$ 。这里需要用到一些数学构造和变换，来凑出一个可用的表达式。

首先我们可以选定式 $\eqref{obj}$ 的采样分布为 $s_0\sim\rho_{0}$ 表示任意的初始分布，那么式 $\eqref{obj}$ 可以被拆开为
$$
\begin{aligned}
J(\theta)&=\mathbb E_{s\sim\rho_0}[v_{\pi_\theta}(s)] =\mathbb E_{s_0\sim\rho_{0},\,s_{t+1}\sim P_{\pi'}(\cdot|s_t)}\left[ \sum_{t=0}^{\infty}\gamma^tv_{\pi_{\theta}}(s_t)-\sum_{t=1}^{\infty}\gamma^tv_{\pi_\theta}(s_t) \right]\\
&=-\mathbb E_{s_0\sim\rho_{0},\,s_{t+1}\sim P_{\pi'}(\cdot|s_t)}\left[ \sum_{t=0}^{\infty}\gamma^t\left[ \gamma v_{\pi_\theta}(s_{t+1})-v_{\pi_\theta}(s_t) \right] \right]
\end{aligned}
$$
这一步变换主要是为了让 $J(\theta)$ 表达式里的采样分布与 $\pi(\theta')$ 对齐。

又可以将 $J(\theta')$ 表达为 
$$
J(\theta')=\mathbb{E}_{s_0\sim\rho_{0},\,a_t\sim\pi_{\theta'}(\cdot|s_t),  \,s_{t+1}\sim P(\cdot|s_t,a_t)}\left[ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t) \right]
$$
> $\mathbb E_{s_0\sim\rho_{0},\,s_{t+1}\sim P_{\pi'}(\cdot|s_t)}$ 表示轨迹的分布，即 $s_0$ 从 $\rho_0$ 中采样，后续状态遵循 $\pi_{\theta'}$ 进行演化所得到的轨迹 $\{s_0,s_1,s_2,...\}$ 的分布。
>
> $\mathbb{E}_{s_0\sim\rho_{0},\,a_t\sim\pi_{\theta'}(\cdot|s_t),  \,s_{t+1}\sim P(\cdot|s_t,a_t)}$ 表示 $s_0$ 从 $\rho_0$ 中采样，后续状态和动作遵循 $\pi_{\theta'}$ 进行演化所得到的轨迹 $\{ s_0,a_0,s_1,a_1,s_2,a_2,... \}$ 的分布。

这两个表达式的采样分布意义一致，所以可以合并入 $J(\theta')-J(\theta)$ 得到
$$
\begin{aligned}
J(\theta')-J(\theta)&=\mathbb{E}_{s_0\sim\rho_{0},\,a_t\sim\pi_{\theta'}(\cdot|s_t),  \,s_{t+1}\sim P(\cdot|s_t,a_t)} \left[ \sum_{t=0}^{\infty}\gamma^t\left[ r(s_t,a_t)+\gamma v_{\pi_\theta}(s_{t+1})-v_{\pi_\theta}(s_t)  \right] \right]\\
&=\mathbb{E}_{s_0\sim\rho_{0},\,a_t\sim\pi_{\theta'}(\cdot|s_t),  \,s_{t+1}\sim P(\cdot|s_t,a_t)} \left[ \sum_{t=0}^{\infty}\gamma^tA_{\pi_{\theta}}(s_t,a_t) \right]\\
&=\sum_{t=0}^{\infty}\gamma^t\,\mathbb{E}_{s_0\sim\rho_{0},\,a_t\sim\pi_{\theta'}(\cdot|s_t),  \,s_{t+1}\sim P(\cdot|s_t,a_t)}\left[ A_{\pi_{\theta}}(s_t,a_t) \right]\\
&=\frac{1}{1-\gamma}\mathbb{E}_{s\sim\nu^{\pi'}_\gamma,\, a\sim\pi_{\theta'}(\cdot|s)}\left[ A_{\pi_{\theta}}(s,a) \right]
\end{aligned}\label{trpo}
$$

> 这里，我们在第一行凑出了 TD error of state value，在第二行用 Advantage
> $$
> A_\pi(s_t,a_t)=\mathbb E_{s_{t+1}\sim P(\cdot|s_t,a_t)}[r(s_t,a_t)+\gamma v_\pi(s_{t+1})-v_\pi(s_t)]
> $$
> 替换了 TD error . 第三行把求和与期望交换，并在第四行用 discounted state visitation distribution 的定义 (见 [番外](#番外：(discounted) state visitation distribution & stationary distribution)) 对期望分布进行了替换
> $$
> \begin{aligned}
> \nu^\pi_\gamma&=(1-\gamma)\sum_{t=0}^\infty\gamma^tP^t_\pi\rho_0\\
> \mathbb{E}_{s'\sim\nu^\pi_\gamma}\,\mathbb{1}\{s'=s\}
> &=(1-\gamma)\sum_{t=0}^\infty\gamma^t\mathbb{E}_{s_{t+1}\sim P_{\pi}(\cdot|s_t)}\boldsymbol{\mathbb 1}\{s_{t+1}=s\}
> 
> \end{aligned}
> $$

回想我们的优化目标：只要能找到一个策略 $\pi(\theta')$ 使得 $J(\theta')>J(\theta)$ ，那么就能保证该策略更优。而式 $\eqref{trpo}$ 最后的结果终于给出了一个可采样估计的表达式：只要估计 $A_{\pi_{\theta}}(s,a)$ 在新策略 $\pi_{\theta'}$ 的采样分布上的均值，使之大于0即可.

定义目标函数
$$
L_\theta(\theta')=\mathbb{E}_{s\sim\nu^{\pi'}_\gamma,\, a\sim\pi_{\theta'}(\cdot|s)}\left[ A_{\pi_{\theta}}(s,a) \right]\\
\label{loss1}
$$
则 $\theta'$ 的优化目标应该最大化式 $\eqref{loss1}$ 。但这件事是无法操作的，因为在该目标函数中， $\theta'$ 出现在采样分布里而没有被显式包含在表达式中，而采样过程是不可微的，因此没有办法使用梯度下降等方法直接优化 $\theta'$ .

> 例如，你可以用 $\pi'$ 采样一步 $s_t,a_t,r_{t+1},s_{t+1}$ 并计算 $\delta_t=r_{t+1}+\gamma v_\pi(s_{t+1})-v_\pi(s_t)$ ，而这时 $\delta_t$ 就是对 $L(\theta')$ 的一步估计，正常情况下接下来应该做梯度下降，但发现 $\delta_t$ 里并不包含参数 $\theta'$ ，无法对 $\theta'$ 求导 —— $\delta_t$ 对 $\theta'$ 的依赖隐含在对 $a_t$ 的采样中。

为了得到对 $\theta'$ 可微的目标函数，我们对 $\eqref{loss1}$ 做两步变换。首先，$s\sim\nu^{\pi'}_{\gamma}$ 对 $\theta'$ 的依赖非常复杂，难以处理，因此做一步近似 $\nu^{\pi'}_\gamma\sim \nu^{\pi}_\gamma$ ，取消对 $\theta'$ 的依赖；然后我们用重要性采样，把 $a\sim\pi_{\theta'}(\cdot|s)$ 的采样分布变换到 $a\sim\pi_{\theta}(\cdot|s)$ ，得到
$$
L_\theta(\theta')=\mathbb{E}_{s\sim\nu^\pi_\gamma,\,a\sim\pi_{\theta}(\cdot|s)}\left[ \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}A_{\pi_\theta}(s,a) \right]
\label{loss2}
$$
这样就得到了一个对 $\theta'$ 可微的目标函数，且采样分布不依赖 $\theta'$。但近似是有代价的，只有当 $\pi(\theta)$ 和 $\pi(\theta')$ 离得不远时，该近似才成立。为了保证这一点，TRPO 引入了基于 KL-divergence (Kullback-Leibler) 的约束条件，并给出了完整的优化目标
$$
\max_{\theta'} L_\theta(\theta') \quad \text{s.t.}\,\,\mathbb{E}_{s\sim\nu^\pi_\gamma}\left[ D_{KL}\left(\pi_\theta(\cdot|s),\pi_{\theta'}(\cdot|s)\right) \right]\leqslant \delta
\label{constraint}
$$

在 TRPO 的论文中，作者证明了在该约束条件下，对 $L_\theta(\theta')$ 的优化会单调提升 $J(\theta')$ 。

> ⚠️ TRPO 算法推导过程中的下标是非常重要的，哪里是 $\pi(\theta)$ 哪里是 $\pi(\theta')$ 这都是非常讲究、非常严格的，过程差一个 $'$ 结果就完全不一样，因此必须非常小心。



#### TRPO 的物理图像

TRPO 原论文里对 $\eqref{loss1}$ 的解释很好的连接起了和 TRPO 算法最简单的 policy iteration 算法，指出 Policy iteration 是 TRPO 优化方法的一个特例。这给予该优化目标一个非常清晰的物理意义，在此我首先给出 $\eqref{loss1}$ 在论文中的形式，再引用原论文的解读。

> $$
> J(\theta')-J(\theta)\propto\sum_{s}\nu^{\pi'}_\gamma(s)\sum_a\pi_{\theta'}(a|s)A_{\pi_\theta}(s,a)
> $$
>
> This equation implies that any policy update $\pi(\theta)\rightarrow\pi(\theta')$ that has a nonegative expected advantage at *every* state $s$, i.e. $\sum_a\pi_{\theta'}(a|s)A_{\pi_\theta}(s,a)>0$ , is guaranteed to increase the policy performance $J(\theta)$, or leave it constant in the case that the expected advantage is zero everywhere. This implies the classic result that the update performed by exact policy iteration, which uses the deterministic policy $\pi'(a|s)=\arg\max_aA_\pi(s,a)$, improves the policy if there is at least one state-action pair with a positive advantage value and nonzero state visitation probability, otherwise the algorithm has converged to the optimal policy.

> 另外，本章的推导相对 TRPO 原论文做了一些简化，原论文导出了非常严格的优化目标下界。我们的简化集中在最后引入 KL-Divergence 的过程，我们直接引入了它，但并没有证明该约束就一定对前面 $\nu^{\pi'}_\gamma\sim \nu^{\pi}_\gamma$ 的近似起到有效的保障。在 TRPO 的原文中，作者证明了 $J(\theta')$ 有一个下界 $L_\theta(\theta')-CD^{\text{max}}_{\text{KL}}(\pi,\pi')$ ，这样可以把 $\max_{\theta'} J(\theta')$ 的优化问题转为 $\max_{\theta'} L_\theta(\theta')-CD^{\max}_{\text{KL}}(\pi,\pi')$ 的优化问题，而后者可以看作式 $\eqref{constraint}$ 的拉格朗日乘子形式，这样才引入了 KL-Divergence。

#### 我们为什么需要 TRPO？

Policy Gradient 系列算法（Actor-Critic 算法也算做 Policy Gradient 算法，因为策略优化目标一致）的推导非常自然，但是实践中有很多问题，在一些简单任务上甚至打不过随机搜索算法（例如 CEM 等）。这主要是由于以下几个原因：

1. 步长难以确定，一步走错可能导致整个训练崩溃。一方面，策略梯度方法的更新步长主要由学习率决定，这是一个超参数；另一方面，策略梯度方法用于更新的数据分布依赖策略本身。因此，如果某次更新步长过大，策略表现恶化，那新采到的数据也会变差，恶性循环；如果更新步长太小，那收敛又太慢。
2. 参数空间的距离并不等于策略空间的距离。确定幅度的参数更新 $\Delta\theta$ 导致的策略分布变化可能差异巨大，但我们实际关心的是动作分布而不是参数本身，这一定程度上也是上一个问题的诱因。
3. 样本效率低，梯度方差大。策略梯度是严格的 on-policy 算法，数据用一遍就得扔。而 TRPO 算法通过重要性采样，把采样分布依赖的策略与目标策略在信任区域内解耦，能够对数据做更充分的利用；同时实现了一定程度的 off-policy （在改进得到的 PPO 中有所体现），提高了数据效率。
4. 没有严格的性能单调性保证。策略梯度只保证无穷小步长下性能递增，在有限步长近似下策略的真实回报可能下降。而 TRPO 证明在信任区域内对目标函数的优化会单调提升性能。

TRPO 一定程度上解决了这些问题。总的来说， TRPO 算法解决的是每一步更新该走多远的问题，通过 KL 散度划定一个策略空间的信任区域 $\delta$ ，在该区间内可以按照目标函数放心优化，由理论保证策略性能单调递增。

但是，TRPO 是一个带约束的非线性优化问题，论文给出的近似求解实现复杂，每步更新计算量大，面临诸多实际应用问题，现代实践用的太少了，在此就不再展开具体的求解方法了。

在实践中大量使用的是下一章介绍的 PPO-Clip 。





#### 番外：(discounted) state visitation distribution & stationary distribution

对于一个马尔可夫决策过程，环境从初态 $t=0$ 时刻开始按照 $P_\pi(s'|s)=\mathbb{E}_{a\sim\pi(\cdot|s)}P(s'|s,a)$ 进行演化。定义 $\nu^\pi_t(s)$ 为 $t$ 时刻状态 $s$ 的访问分布；特别地， $\rho_0=\nu^\pi_0(s)$ 为系统的初始状态分布。那么有
$$
\nu^\pi_t(s')=\sum_sP_\pi(s'|s)\,\nu_{t-1}^\pi(s)
$$
或者写成矩阵形式
$$
\nu_t^\pi=P_\pi\nu_{t-1}^\pi=P^t_\pi\,\rho_0
\label{transition2}
$$
这就是该 MDP 的 **状态访问分布 State visitation distribution** .

特别地可以定义 **稳态分布 stationary distribution**
$$
\rho_\pi=P_\pi\rho_\pi
$$
可以看出它是状态转移方程 $\eqref{transition}$ 的不动点，且一般情况下有
$$
\lim_{t\rightarrow\infty}\nu_t^\pi=\rho_\pi
$$
即演化无穷长时间后系统稳定在 stationary distribution :
$$
\rho_0\xrightarrow{P_\pi}\nu_1^\pi\xrightarrow{P_\pi}\nu_2^\pi\rightarrow\cdots\xrightarrow{t\rightarrow\infty}\rho_\pi
$$
有时我们还考虑  **discounted state visitation distribution** ，即考虑 discounted rate $\gamma$ 之后的状态访问分布
$$
\nu_\gamma^\pi=(1-\gamma) \sum_{t=1}^\infty\gamma^tP_\pi^t\rho_0
$$
其物理意义为将每个时刻的 SVD 按照 $\gamma^t$ 进行加权平均，且一般有
$$
\lim_{\gamma\rightarrow1}\nu_\gamma^\pi=\rho_\pi
$$
有了一个策略的 SVD，还可以进一步定义该策略下 $(s,a)$ 的访问分布，被称为 **占用度量 (occupancy measure)** 
$$
\rho^\pi(s,a)=\pi(a|s)\,\nu^\pi(s)
$$
例如可以考虑 **discounted occupancy measure**
$$
\rho^\pi_\gamma(s,a)=(1-\gamma)\sum_{t=1}^\infty\gamma^tP_\pi^{(t)}(s)\,\pi(a|s)
$$
其中 $P_\pi^{(t)}(s)$ 表示 $P^t_\pi\rho_0$ 向量中状态 $s$ 对应的那一列。







