### Actor-Critic 系列算法

Actor-Critic 一般代指一系列算法，用 TD-learning 系列算法估计 action value，再用 Policy Gradient 改进策略。特别地，Actor-Critic 已经具备现代强化学习算法的雏形，所以全都用神经网络做 value approximation 了，不再使用任何 Tabular 方法。其中，用于估计 value 的网络称为 Critic 网络，策略网络称为 Actor 网络。

将 Value function TD-learning 用于 Policy Gradient 算法中估计 action value，得到最简单的 QAC；在此基础上，在 Policy Gradient 中将 action value 替换为基于 state value 的 advantage，就得到了 A2C，从这里开始我们又开始用神经网络估计 state value 而不是 action value；在 A2C 的基础上，加入重要性采样方法，可以得到 off-policy Actor-Critic。

#### QAC

由于推导过程和 Value-function Approximation 章节的推导一致，只是把 value network 更新的部分拼到了 policy gradient 算法中，因此这里就不再详细推导，直接给出 pseudocode

> - 初始化 Critic 网络 $Q_\omega(s,a)$ 和 Actor 网络 $\pi_\theta(a|s)$
>
> - for $e=1\rightarrow E:$
>
>   - 随机选取初始状态 $s_0$，从 $\pi_\theta$ 采样动作 $a_0$
>
>   - for $t=1\rightarrow T:$
>
>     - 环境给出奖励 $r_t$；环境演化得到 $s_{t+1}$；从策略 $\pi_\theta$ 采样动作 $a_{t+1}$，得到本轮数据 $\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1} \}$
>
>     - 利用数据 $\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1} \}$ 计算 TD-target
>       $$
>       \hat y_t=r_{t+1}+ Q_\omega(s_{t+1} ,a_{t+1})
>       $$
>
>     - 利用数据 $\{s_t,a_t,\hat y_t\}$ 对如下损失做 SGD，更新 Critic 网络
>       $$
>       \mathcal L(\omega)|_{\{s_t,a_t,\hat y_t\}}=\left[\hat y_t-Q_\omega(s_t,a_t)\right]^2
>       $$
>
>     - 利用数据 $\{s_t,a_t,Q_\omega(s_t,a_t)\}$ 对如下损失做 SGD，更新 Actor 网络
>       $$
>       \tilde{J}(\theta)|_{\{s_t,a_t,Q_\omega(s_t,a_t)\}}=-Q_\omega(s_t,a_t)\log\pi_\theta(a_t|s_t)
>       $$
>
>   - end for
>
> - end for

这里对 Critic 网络的更新沿用了 DQN 中的 target-main network 设计。

另外，对 Actor 网络的更新我没有写出梯度，而是用了一个 pseudo loss，更加贴近实践，推导见 [番外：Policy Gradient 算法的梯度下降目标是什么？](#番外：Policy Gradient 算法的梯度下降目标是什么？)

#### A2C (引入 Advantage)

这一节把 policy gradient 中的 action value $q_\pi(s,a)$替换为 advantage $A_\pi(s,a)$，并利用 TD-error 近似，用 state value $v(s)$ 表达 advantage。至此，策略评估从估计 action value 又回到了估计 state value。

> 想来也挺神奇的：最开始的 model-based policy iteration 算法是先求 state value，再计算 action value，再改进策略；然后到了 model-free 算法，MC, Sarsa, Q-learning, DQN, REINFORCE, QAC 这些算法全都在从数据中直接估计 action value；现在，到了 A2C，我们又开始估计 state value 了。

##### Advantage 优势函数

优势函数 Advantage $A_\pi(s,a)$ 被定义为
$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)
$$
它表示在状态 $s$ 上采取动作 $a$ 比当前状态上所有动作的平均要好多少。

根据 Baseline Invariance property，可以证明：
$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[ q_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s) \right]=\mathbb E_{\pi_\theta}\left[ A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s) \right]
$$
即可以将 policy gradient 中的 $q_\pi(s,a)$ 直接替换为 $A_\pi(s,a)$ .

> 证明思路是证明 $E_{\pi_\theta}\left[ v_\pi(s)\nabla_\theta\log\pi_\theta(a|s) \right]=0$，见《Mathematical Foundations of Reinforcement Learning》Section 10.2.1

做这一替换的目的是为了减少 value 估计的方差，因为 $A_{\pi}(s,a)$ 的估计方差要比 $q_\pi(s,a)$ 的估计方差更小。

但这还不够，因为估计 Advantage 需要同时估计 $q_\pi(s,a)$ 和 $v_\pi(s)$ ，实践中需要两个神经网络。能否只用一个神经网络呢？可以，根据定义
$$
q_\pi(s,a)=r(s,a)+\gamma\,\mathbb E_{s'\sim P(s'|s,a)}\left[ v_\pi(s')\right]
$$
可以把 Advantage 改写为
$$
A_\pi(s,a)=r(s,a)+\gamma\,\mathbb E_{s'\sim P(s'|s,a)}\left[ v_\pi(s')\right]-v_\pi(s)
$$
在实践中可以用单次采样值作为 Advantage 的估计
$$
A_\pi(s_t,a_t)\sim r_{t+1}+\gamma v_\pi(s_{t+1})-v_\pi(s_t)
$$
而这个表达式显然就是 state value 的 TD-error $\delta_t$ . 这样，我们就可以只学习一个用于估计 state value 的神经网络就可以了。

> 一个看起来更复杂但更严谨的推导：
> $$
> \begin{align*}
> \nabla_\theta J(\theta)&=\mathbb E_{\pi_\theta}\left[ A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s) \right]\\
> &=\mathbb E_{(s,a)\sim\pi_\theta}\left[ \left(r(s,a)+\gamma\,\mathbb E_{s'\sim P(s'|s,a)}\left[ v_\pi(s')\right]-v_\pi(s)\right)\nabla_\theta\log\pi_\theta(a|s) \right]\\
> &=\mathbb E_{(s,a)\sim\pi_\theta,s'\sim P(s'|s,a)}\left[ \left(r(s,a)+\gamma v_\pi(s')-v_\pi(s)\right)\nabla_\theta\log\pi_\theta(a|s) \right]\\
> &=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1} \}\sim\pi_\theta}\left[ \left(r_{t+1} +\gamma v_\pi(s_{t+1})-v_\pi(s_t)\right)\nabla_\theta\log\pi_\theta(a_t|s_t) \right]\\
> &=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1} \}\sim\pi_\theta}\left[ \delta_t\nabla_\theta\log\pi_\theta(a_t|s_t) \right]
> \end{align*}
> $$
> 最后这个结果可以直接用于 SGD .

##### Pseudocode

给出 pseudocode

> - 初始化 Critic 网络 $V_\omega(s)$ 和 Actor 网络 $\pi_\theta(a|s)$
>
> - for $e=1\rightarrow E:$
>
>   - 初始化 $s_1$
>
>   - for $t=1\rightarrow T:$
>
>     - 根据策略 $\pi_\theta$ 采样动作 $a_t$；环境给出 $r_{t+1}$；环境演化得到 $s_{t+1}$
>
>       得到本轮数据 $\{s_t,a_t,r_{t+1},s_{t+1} \}$
>
>     - 计算 TD-target 和 TD-error
>       $$
>       \hat y_t=r_{t+1}+\gamma V_\omega(s_{t+1})
>       $$
>
>       $$
>       \delta_t=r_{t+1}+\gamma V_\omega(s_{t+1})-V_\omega(s_t)
>       $$
>
>     - 利用数据 $\{s_t, \hat y_t\}$ 对如下损失做 SGD，更新 Critic 网络
>       $$
>       \mathcal L(\omega)=\left( \hat y_t-V_\omega(s_t) \right)^2
>       $$
>
>     - 利用数据 $\{s_t,a_t,\delta_t\}$ 对如下损失做 SGD，更新 Actor 网络
>       $$
>       \tilde J(\theta)=-\delta_t\log\pi_\theta(a_t|s_t)
>       $$
>
>   - end for
>
> - end for

同样地，这里对 Critic 网络的更新沿用了 DQN 中的 target-main network 设计。

#### Off-policy actor-critic

Off-policy 是很好的性质，它使得算法可以复用之前的经验，提高数据使用效率。

利用重要性采样的技巧，我们可以把前面的 on-policy actor-critic 算法改造为 off-policy actor-critic 算法。重要性采样的内容见 [番外：重要性采样](#番外：重要性采样)，其中包括了大量我个人的思考和对相关问题的讨论，有些内容和课本上并不一致。但在这一节正文里，我还是先 follow 赵老师书里的算法。

具体的，可以利用重要性采样把 TD Advantage Policy Gradient 改写为
$$
\nabla_\theta J(\theta)=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1} \}\sim\pi_\beta}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}\delta_t\nabla_\theta\log\pi_\theta(a_t|s_t) \right]
$$
其中，$\pi_\beta$ 称为 behaviour policy，用于和环境交互采样数据；$\pi_\theta$ 称为 target policy，是我们要更新的策略。具体的推导可以参考 [番外：重要性采样](#番外：重要性采样) 和赵老师的课程。

> 赵老师课本里给出的 pseudocode 和 A2C 几乎一模一样，只是把梯度换成了重要性采样的形式，并把采样这一步所用的策略改成了 $\pi_\beta$ . 这种写法事实上完全没有利用到 off-policy 经验复用的优良性质，所以我觉得是有待考量的。

#### 番外：重要性采样

首先有如下事实
$$
\mathbb E_{X\sim p_0}[X]=\sum_x p_0(x)\cdot x=\sum_x p_1(x)\frac{p_0(x)}{p_1(x)}x=\mathbb E_{X\sim p_1}\left[ \frac{p_0(X)}{p_1(X)}X \right]=\mathbb E_{X\sim p_1}\left[f(X)\right]
$$
即某个随机变量 $X$ 在 $p_0$ 分布上的期望等于 $f(X)$ 在分布 $p_1$ 上的期望，其中 $\frac{p_0(X)}{p_1(X)}$ 被称为 importance weight。

有了重要性采样这一工具，你可以对任意期望更改它的采样分布，只需要在括号内加上 importance weight 即可。这是一个非常有用的性质，它使得你可以在另一个分布上采样，但是计算另外一个分布上的期望，换句话说，**重要性采样把 "采样分布" 和 "期望分布" 分开了。**在强化学习的语境下，这就意味着 off-policy，也就是你可以用一个策略去采样，但是去评估另一个策略的 value。

具体地，以 Advantage 形式的策略梯度为例，可以用重要性采样将其改写为
$$
\begin{align*}
\nabla_\theta J(\theta)
&=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1} \}\sim\pi_\theta}\left[A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a_t|s_t) \right]\\
&=\mathbb E_{s_t\sim \rho(\pi_\theta);\,a_t\sim \pi_\theta(\cdot|s_t); \, r_{t+1},s_{t+1}\sim P(r,s'|s_t,a_t)}\left[ A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a_t|s_t) \right]\\
&=\mathbb E_{s_t\sim \rho(\pi_\beta);\,a_t\sim \pi_\beta(\cdot|s_t); \, r_{t+1},s_{t+1}\sim P(r,s'|s_t,a_t)}\left[\frac{\rho(\pi_\theta)}{\rho(\pi_\beta)} \cdot\frac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}\cdot A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a_t|s_t) \right]
\end{align*}
$$
上述推导中，为了能够厘清到底对哪些分布应用了重要性采样，我在第二行把所有随机变量的采样分布显式地写了出来，即 $\mathbb E_{s_t\sim \rho(\pi_\theta);\,a_t\sim \pi_\theta(\cdot|s_t); \, r_{t+1},s_{t+1}\sim P(r,s'|s_t,a_t)}$ ，可以看到，只有 $s_t, a_t$ 两项的分布和 $\pi_\theta$ 有关，其中 $\rho(\pi_\theta)$ 是 $\pi_\theta$ 的 stationary ditribution，而后面两项分布只和环境有关。由于我们希望从其他策略 $\pi_{\beta}$ 中采样来估计  $\pi_\theta$ 的策略梯度，需要把 $\rho(\pi_\theta)$ 变为 $\rho(\pi_\beta)$ ，把 $\pi_\theta({\cdot|s})$ 替换为 $\pi_\beta(\cdot|s)$，同时添加对应的 importance weight ，这也是我在第三行所做的事。

第三行最后的结果可能让人有点费解，它和各种课本里给出的形式不太一样，差了一项 $\frac{\rho(\pi_\theta)}{\rho(\pi_\beta)}$ ，当两个策略差的不远时，可以认为两个策略的 stationary distribution 近似相等，这一项取 1，可以得到利用重要性采样的 Advantage 策略梯度
$$
\nabla_\theta J(\theta)=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1} \}\sim\pi_\beta}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}A_{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a_t|s_t) \right] ,\text{when} \,\pi_\beta\sim\pi_\theta
$$

> 最后这个结果吧，它和赵老师书里的推导过程和结果都不太一致，赵老师的书里并没有要求 $\pi_\beta$ 和 $\pi_\theta$ 距离不太远。我的理解是，当前推导出的这个策略梯度对应的目标函数是
> $$
> J(\theta)=\mathbb E_{s\sim\rho(\pi_\theta)}\left[v_{\pi_{\theta}}(s)\right]
> $$
> 而赵老师书里 Sec. 10.3.2, Box 10.2 里的推导似乎把目标函数定为
> $$
> J(\theta)=\mathbb E_{s\sim\rho(\pi_\beta)}\left[v_{\pi_{\beta}}(s)\right]
> $$
> 因为目标函数的分布不一样，最后推导出了不一致的结果，赵老师的做法是在一开始定义目标函数的时候就做了分布替换 $\rho(\pi_\theta)\rightarrow\rho(\pi_\beta)$ ，没有加 importance weight，目标函数被改变；而我这里的推导不改变目标函数，因此多了一项 importance weight。我觉得二者都有道理，在 policy gradient 章节我有提到过其实目标函数里的这个分布是什么并不重要，所以虽然赵老师的做法也是合理的，从算法实践的角度也更好用，因为不需要要求 $\pi_\beta\sim\pi_\theta$ .

#### Brain Storm：关于 Mini-Batch 的讨论

下面引用的部分是学习过程中的误解，所以其实是**完全错误**的，但留在这里待后续写具体算法落地实现的时候来进行对比。事实上 TD-based On-policy 算法也完全可以使用并行环境、大 batch 来进行梯度估计和网络更新，只是数据使用效率偏低，一轮数据只能使用一次。

> 目前学过的 model-free 算法在实操过程中都遵循着 **采样，评估** 两步走或者是 **采样，评估，改进** 的三步走架构，其中off-policy 算法（Q-learning, DQN）都是两步走，而 on-policy 的算法 （MC, Sarsa, REINFORCE, on-policy Actor-Critic）都是三步走。
>
> 涉及神经网络更新的系列算法（DQN, Value function MC, Value function Sarsa, REINFORCE, on-policy Actor-Critic）都需要做梯度下降/上升，这些算法的真实梯度都是一个期望值，而实际落地总是使用 SGD 或 MBGD。相比 SGD，MBGD 显然是梯度估计更加准确的一种方法，但值得注意的是，目前见过的算法有些使用 MBGD，有些却只能使用 SGD。能使用 MBGD 当然优先使用 MBGD，那很多算法使用 SGD 的原因是算法结构本身限制了无法使用 MBGD。那到底是什么原因导致了只能使用 SGD 呢？
>
> 先来总结一下那些算法使用 SGD，哪些使用 MBGD。Value function Sarsa, on-policy Actor-Critic 使用了 SGD，每次采样只采一条数据就立刻进行梯度下降；而 Value function MC, REINFORCE, DQN 则使用了 MBGD，每次梯度下降都用到了若干数据做平均。
>
> 所以结论就很清晰了：**采用 TD-learning 的 On-policy 算法普遍使用 SGD**。这是由算法结构决定的，On-policy 算法要求更新数据必须由当前策略产生，而 TD-learning 则倾向于每来一个数据就立刻更新，那么一旦更新，旧数据就无法使用，只能采新数据，采了新数据又要立刻更新，二者啮合导致 SGD。
> 反观 MC 算法本身就要求多步采样才能进行估计；而 DQN 算法则基于 Q-learning，是 off-policy 算法，可以用其他策略的经验进行学习，因此可以从 replay buffer 里采样 mini-batch 数据做梯度估计，虽然是 TD-learning，但是却可以一次采样多条数据。
>
> 所以有必要把 Actor-Critic 算法改造为 off-policy，这可以进行 MBGD。

#### 番外：Policy Gradient 算法的梯度下降目标是什么？

Policy Gradient 和 Actor-Critic 算法都显式地给出了策略梯度，在很多资料中的 pseudocode 中，梯度下降也被写为了显式的 $\theta\leftarrow\theta +\alpha\cdot(...)$ 的形式，但是正如之前提到的，实际中我们不需要也不会手动写出梯度去做更新的，而是给出一个 objective。那 policy gradient 的 objective 是什么？显然 policy gradient 的推导出发点是最大化所有状态上的期望 state value
$$
J(\theta)=\mathbb E_{s\sim\rho_{\pi}(s)}[v_{\pi}(s)]
$$
但这个 objective 是没法使用的，因为它对 $\theta$ 的依赖是通过采样分布和 state value 间接体现的，这个过程不可微（例如，要估计 $\pi_\theta$ 的 state value 你需要做很多采样，与环境进行交互，这些过程都是不可微的），因此从这个 objective 出发无法得到显式的监督数据 $(x_i,\hat y_i)$，无法直接做反向传播和梯度下降。

但我们通过数学推导得到了这个 objective 的梯度表达式
$$
\nabla_\theta J(\theta)=\mathbb E_{(s,a)\sim\pi_\theta}\left[q_{\pi_\theta}(s,a)\nabla_\theta\log \pi_\theta(a|s)\right]
\label{pg}
$$
从工程上我们可以倒推一个能得到该梯度的 pseudo objective
$$
\tilde{J}(\theta)=\mathbb E_{(s,a)\sim\pi_\theta}\left[ q_{\pi_{\theta^*}}(s,a)\log\pi_\theta(a|s) \right]
$$
注意这里 $q_{\pi_{\theta^*}}(s,a)$ 表示该项视为常数不参与反向传播。这样，上面这个 objective 的梯度就恰好等于策略梯度 Eq. $\eqref{pg}$ , 在代码实践中就可以直接用这个 objective 进行反向传播了，但它仅仅是一个工程设计，没有物理意义。注意 SGD 是最小化目标函数，因此要对上式取反。

 