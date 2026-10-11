#### Policy Gradient

Policy Gradient 算法希望解决一个问题，那就是用一个函数（神经网络）来表达一个策略 $\pi(a|s)$ ，代替之前的表格形式，它的好处是可以直接学习和优化一个随机策略 Stocastic Policy，同时也避免了储存整个策略表格所需要的空间。

> 值得注意的是，目前为止介绍的算法（MC, Sarsa, Q-learning 以及它们对应的 value function 版本的算法）在实际的运行代码中，都不会显式维护一个策略，而是直接从 $Q$-value 得到策略，策略被内化在 $Q$-value 中了，无论是“先评估，再改进”的算法 (MC, Sarsa) 还是直接求解 optimal Q 的算法（Q-learning），策略都是依托 $Q$ 存在的。
>
> 而这种依托是有条件的，那就是最后的策略只能是一个确定性策略 Deterministic Policy，也就是在某个 state 上确定性采取 action value 最大的那个动作，这时候策略表格由二维表格退化到一维表格了。这一点限制了这些算法能应用的情况。在很多情况下 ，例如最大熵强化学习、多智能体博弈（纳什均衡）和多模态行为建模等等复杂显示问题，我们想建模或者求解一个 Stocastic Policy，这时候基于 deterministic policy 的算法失效，我们需要能够更多样的、能够表达随机策略的算法。作为 stocasitc policy 的第一步，我们来到了 Policy Gradient。
>
> 另外，虽然 MC 和 Sarsa 在探索阶段会使用 $\epsilon$-greedy 策略，这看似是一个 stocastic policy ，但其本质只是为了在算法优化过程中保证策略的探索性，确保数据采样覆盖整个分布。随着 $\epsilon$ 收敛到零，策略也会收敛到一个 determinisic policy。

Policy Gradient 依托深度学习给出一个显式求解策略 $\pi$ 的算法。

首先使用一个神经网络 $\pi_\theta:s\rightarrow \{p(a)|\forall a\in\mathcal A(s)\}$ 表达在 $s$ 上采取每个动作 $a$ 的概率。
为了优化这个神经网络策略，我们需要一个 objective function。回顾一下强化学习的终极目标是求解最优策略，而最优策略的定义是所有 state 上的 state value 都最大。因此，可以定义策略网络的优化目标是最大化所有状态上的期望 state value:
$$
J(\theta)=\mathbb E_{s\sim\rho_{\pi}(s)}[v_{\pi}(s)]
$$

> 这里的求期望 $\mathbb E_s$ 中 $s$ 所在的分布到底是什么，这是一个值得思考的事情。事实上这个分布可以是任意的，它和你想求解什么样的策略有关系。如果希望求解的是严格的全局最优策略，即在每一个状态上 state value 都是最大值，那这个分布必须覆盖到所有的 state，最简单的情况就是一个均匀分布；如果你只关心在一部分状态上的表现，希望最大化这部分状态上的 state value，那这个分布可以只在这些状态上有非零概率。还有一种特殊的分布 $\rho_\pi(s)$ 称为 **stationary distribution**，它是一个和 $\pi$ 有关的分布，即采取 $\pi$ 策略与环境交互无穷多次之后，每个状态被访问到的概率；数学上讲，它满足 $\rho_\pi=P_\pi\rho_\pi$ 其中 $P_\pi(s'|s)$ 是策略 $\pi$ 的状态转移矩阵，它告诉我们 $\rho_\pi$ 是 $P_\pi$ 的特征值为1的特征向量。

> 和 value function approximation 章节不同，在 Policy Gradient 算法中，仅仅写出 objective 并不够，不能直接拿来做深度学习梯度下降/上升，因为这个简单的目标实际优化起来却太麻烦了。为了后续算法的改进，我们在这里显式推导目标函数的梯度，加深对算法的理解，也方便后续改进。

当状态 $s$ 的分布采取 **stationary distribution** $\rho_\pi(s)$ 时，我们可以显式推导出 $J(\theta)$ 的梯度为
$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[ q_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s) \right]
$$

> 具体的推导可以查阅书籍。这里值得强调的是 "当状态 $s$ 的分布采取 $\rho_\pi(s)$ 时"这一条件，推导过程是依赖于它的，也就是说这种情况能导出一个比较好用的梯度形式；至于其他分布能不能导出很好的梯度形式则有待考察。
>
> 再一次，注意 $\mathbb E_{\pi_\theta}$ 的意义。这里如果展开来写，应该写为 $\mathbb E_{s\sim\rho(\pi_\theta),\,a\sim\pi_\theta(\cdot|s)}$，即 $s$ 应该从 $\pi_\theta$ 的 stationary distribution 中采样，而 $a$ 应该依据 $\pi_\theta(\cdot|s)$ 来采样。换句话说，这里的期望可以理解为**在"遵循 $\pi_\theta$ 所得到的 $(s,a)$ pair 的分布"上求期望**，这就说明 Policy Gradient 是一个 on-policy 的算法，因为进行策略更新所依赖的数据必须是由当前策略与环境交互产生的。后面可以看到，我们可以利用重要性采样等方法将这一要求降低，使他变成没有那么 on-policy 的算法，得到 TRPO 和 PPO 等算法。

而策略的梯度更新则可以显式写为
$$
\theta\leftarrow\theta+\alpha\,\mathbb E_{\pi_\theta}\left[ q_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s) \right]
$$

> 注意这里 $\alpha$ 前面为加号，因为我们是要最大化目标。回顾 RM & SGD 番外篇的内容。

到此为止，Policy Gradient 的核心思想就介绍完了，可以看到，为了更新策略网络，我们必须计算梯度，而为了计算梯度我们必须知道 action value $q_{\pi_\theta}(s,a)$ 。这又闭环了，回到了如何求解 action value 的问题，我们已经学过 MC, Sarsa 两种 model-free 的算法，可以用它们先求解 $q_{\pi_\theta}$ ，再计算梯度，最后更新网络。

#### REINFORCE

REINFORCE 算法就是基于 MC 的 policy gradient，思想很简单，我们直接看 pseudocode

> - for $e=1\rightarrow E:$
>
>   - 根据策略 $\pi(\theta)$ 在环境中采样得到 Trajectory $\{s_t,a_t,r_{t+1}
>     |t=1,...,T\}$
>
>   - for $t=T\rightarrow 1:$
>
>     - 计算 $q_{\pi_\theta}(s_t,a_t)$ ，即从 $(s_t,a_t)$ 出发得到的 discounted return
>       $$
>       q_{\pi_\theta}(s_t,a_t)=\sum_{i=t+1}^T\gamma^{i-t-1}r_{i}
>       $$
>
>     end for
>
>   - 进行梯度上升
>     $$
>     \theta\leftarrow\theta+\alpha\,\sum_{t=1}^{T}\left[ q_{\pi_\theta}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t|s_t) \right]
>     $$
>
> - end for

注意这里使用的是 Mini-Batch Gradient Ascent：一轮采样的若干数据合起来作为真实梯度的一次估计，进行一次梯度上升。

> 在赵老师的《强化学习的数学原理》视频中，把 REINFORCE 写成一种 off-policy 的算法，我觉得是有点奇怪的，不符合上面的理论推导。我这里的写法和《动手学强化学习》对齐。

#### Policy Gradient 的物理意义

Policy Gradient 算法中目标函数的梯度形式的物理意义值得探讨。对 $\nabla_\theta J(\theta)$ 做如下变换
$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[ q_{\pi_\theta}(s,a)\nabla_\theta\log\pi(a|s) \right]=\mathbb E_{\pi_\theta}\left[\frac{q_{\pi_\theta}(s,a)}{\pi_\theta(a|s)}\nabla_\theta\pi_\theta(a|s)\right]
$$
其中 $\nabla_\theta\pi_\theta(a|s)$ 的方向是“提高策略在 $s$ 上采取动作 $a$ 的概率”的方向，它的前面乘以系数 ${q_{\pi_\theta}(s,a)}/{\pi_\theta(a|s)}$ 表示如果该动作 $a$ 的 action value 很大，但当前策略采取它的概率又很小，那么接下来应该以比较大的步幅来增大该动作的概率；其余情况可以类推，即可以理解为某个动作的概率是否提高或降低，取决于它的 action value 和当前它被采取的概率的比值。 

#### 小结

从 REINFORCE 算法开始，强化学习算法才真正变得显式上遵循 “先评估，再改进” 的中心思想。我们有一个显式的策略函数，然后让它和环境交互，根据交互数据对策略进行评估，得到它的 action value；然后，用这个评估结果去做梯度上升，改进当前策略；循环往复。

蒙特卡洛算法的优点是可以得到梯度的无偏估计，但是缺点是涉及多步采样来估计某对 $(s,a)$ 的 action value，导致估计方差较大，导致更新不稳定。

除了 MC，我们还学过基于 TD-learning 估计 action value 的算法，在 TD-learning 的基础上还可以加入 value function approximation 用神经网络估计 action value。用 TD-learning 做“先评估”，再用 policy gradient 做”再改进“，就是接下来要介绍的 Actor-Critic 算法。

