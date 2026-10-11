#### Value function approximation (part 1: Introduction and MC)

值函数近似 value function approximation 用于状态空间过大甚至是连续情况下的强化学习算法。

目前为止介绍的算法都建立在离散的状态空间和动作空间上，环境模型 $r(s,a), P(s'|s,a)$、 action value $q_\pi(s,a)$ 和 policy $\pi(a|s)$ 都以表格形式储存。当状态空间和动作空间很大，甚至是连续情况时，这种方法失效。

我们先处理状态空间很大甚至是连续的情况。这种情况下我们采用函数拟合的形式去储存 $q$ ，即用一个函数 $Q_\theta^\pi(s,a):(s,a)\rightarrow q_\pi(s,a)$ 来近似估计 $q$-table，其中 $\theta$ 是该函数的参数，该函数输入 $(s,a)$ ，输出 $q_\pi(s,a)$；这个函数一般用神经网络来表达，这种方法有一定的精度损失，因此被称为 value funtion approximation。

> Value function approximation 用一个函数代替表格去储存 action value，后面介绍的 policy gradient 用函数代替表格来储存 policy，二者合起来得到 Actor-Critic 算法。

理解到这里，自然可以把 Monte Carlo, Sarsa 和 Q-learning 都用 value function approximation 来改造。从离散到连续唯一的区别就是更新方式，之前的更新方式都是更新 $Q$-table 中的某一个值，而现在需要学习和更新的是整个 $Q$-function .

学习任何函数（神经网络），都需要有一个优化目标 objective，再通过梯度下降/梯度上升来最小化/最大化这个目标。我们希望 $Q_\theta(s,a)$ 尽可能逼近 $Q$-table，那目标函数就是
$$
\mathcal L(\theta)=\mathbb E_{\pi}\left[\left(Q_\theta^\pi(s,a)-q_\pi(s,a)\right)^2\right]
$$

> 这里 $\mathbb E_\pi$ 表示 $\mathbb E_{(s,a)\sim\pi}$，即遵循策略 $\pi$ 的情况下每个状态动作对 $(s,a)$ 的概率分布。

直接最小化这个目标函数，就得到了对应的 MC 算法。具体地，通过 MC 章节介绍的算法采样得到一大堆 $(s,a)$ 的 action value 的估计值，这样你就得到了一大堆监督数据：$\{(s,a), q_\pi(s,a)\}$ ，用这一堆监督数据进行梯度下降，更新 $Q_\theta^\pi(s,a)$ 的参数，然后改进策略，循环往复。

> 这里可能有点奇怪：得到监督数据 $\{(s,a), q_\pi(s,a)\}$ 的过程看起来就是传统的 MC 算法，那似乎还是需要存一个表格，完全没有任何改进。但实则不然，你不需要维护一个所有 $(s,a)$ 都存在的表格，而是只需要保存那些采样过程中出现过的 $(s,a)$ ，并只估计他们的 action value，得到一轮监督数据，然后进行梯度下降。在进行过梯度下降并更新策略之后，这些数据就可以扔了，最终维护的只有函数 $Q_\theta^\pi(s,a)$

> 关于记号：表格形式的 action value 我记为 $q_\pi(s,a)$ ，这是为了和最初的贝尔曼方程里的记号一致；Q-learning 中迭代过程的 Q 值记为 $Q$ ；Action value funtion 记为 $Q_\theta^\pi(s,a)$，表示以 $\theta$ 为参数的、用于估计策略 $\pi$ 的 action value 的函数。

#### Sarsa with value approximation

MC with value funtion 算法的目标函数非常直观，但是更新太慢，计算复杂。我们期待使用 TD-learning 来加快学习效率；换句话说，其实就是希望使用 Sarsa 算法做 value function 的更新。

那么如何通过修改目标函数从而在 Sarsa 中引入 value funtion 呢？根据 action value 的贝尔曼期望方程 $q_\pi(s,a)=\mathbb E_\pi[r_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1})|s_t=s,a_t=a]$ ，可以认为一个良好的 value funtion 也应该满足
$$
Q_\theta^\pi(s,a)=\mathbb E_\pi[r_{t+1}+\gamma Q_\theta^\pi(s_{t+1},a_{t+1})|s_t=s,a_t=a]
$$
对应地，可以写出 Bellman error 形式的 objective loss
$$
\mathcal L(\theta)=\mathbb E_\pi\left[\left( r_{t+1}+\gamma Q_\theta^\pi(s_{t+1},a_{t+1})-Q_\theta^\pi(s_t,a_t)\right)^2 \right]
$$
伪代码

> - 初始化策略表格 $\pi(a|s)$ 和神经网络 $Q_\theta^\pi(s,a)$
>
> - for $e=1\rightarrow E:$
>
>   - 随机选择初始状态 $s_0$ ，采样初始动作 $a_0$，得到初始奖励 $r_0$
>
>   - for $t=0,...,T-1:$
>
>     - 环境给出 $r_{t+1}=r(s_t,a_t)$；环境从 $(s_t,a_t)$ 出发演化得到 $s_{t+1}$；遵循策略 $\pi$ 采样得到 $a_{t+1}$ 
>
>     - 根据本轮数据 $\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\}$ ，对如下损失进行一次 SGD
>       $$
>       \mathcal L(\theta)|_{\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\}}=\left( r_{t+1}+\gamma Q_\theta^\pi(s_{t+1},a_{t+1})-Q_\theta^\pi(s_t,a_t)\right)^2
>       $$
>
>     - 根据 $\epsilon$-greedy 更新策略表格 $\pi(a|s)$
>
>   - end for
>
> - end for

这里，关于 “根据本轮数据 $\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\}$ ，对如下损失进行一次 SGD” 这句话具体是什么意思，可以查看 [番外：实践中具体如何进行梯度下降？](#番外：实践中具体如何进行梯度下降？)。

#### DQN (Deep Q Network)

DQN 在 Q-learning 中引入 value function，并引入了很多重要的工程设计，例如 **经验回放池 Replay buffer**，**目标网络-主网络** 等。Replay buffer 只能用在 offpolicy 算法中，而目标网络-主网络设计则是通用的解决梯度问题的重要设计。

先仿照上一节的 Sarsa with value function 写出 Q-learning with value function 算法的目标 loss
$$
\mathcal L(\theta)=\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1}\}}\left[\left( r_{t+1}+\gamma \max_{a_{t+1}}Q_\theta(s_{t+1},a_{t+1})-Q_\theta(s_t,a_t)\right)^2 \right]
$$

> 和 Tabular Q-learning 类似，这里的期望 $\mathbb E_{\{s_t,a_t,r_{t+1},s_{t+1}\}}$ 所在的分布和策略是无关的，只需要在所有可能发生的 state transitions $\{s_t,a_t,r_{t+1},s_{t+1}\}$ 上求期望，这只和环境有关，不要求具体遵循哪个策略。因此 DQN 显然也是 off-policy 算法。

此时，只需要把 Tabular Q-learning 算法中的 TD 更新改为基于上述 objective 的 SGD 更新，就得到了 Q-learning with value function。

具体到真正实用的、由 DeepMind 团队发布的 Deep Q Network 算法，还采用了两项工程设计 Target Network-Main Network 和 Replay buffer，这两项都是由于引入了深度神经网络梯度下降之后而采取的必要优化。

##### Target/Main Network (梯度下降的具体做法)

先介绍 Target Network-Main Network. 这个技巧的名字我觉得起的不好，听起来太过 tricky，难以让人捕捉到具体的操作是什么样子的，Google DeepMind 原论文里也没有这样的表述。它实际上就是一个简化梯度计算的技巧，我这里从实操的角度出发来介绍这个技巧。

由于 DQN 在更新中使用了梯度下降，实际计算中必然涉及对 $\mathcal L(\theta)$ 求梯度，但是对其中的 $\max_{a'}Q_\theta(s',a')$ 这一项求 $\theta$ 的梯度是比较费劲的，所以干脆就不算它的梯度了，只算后面 $Q_\theta(s,a)$ 的梯度。具体实操起来就是，先用当前的网络计算 $\hat y=r_{t+1}+\gamma\max_{a_{t+1}}Q_\theta(s_{t+1},a_{t+1})$ 的具体数值，然后把 $\{s_t,a_t,\hat y\}$ 当成一份监督数据，去优化 $(\hat y-Q_\theta(s_t,a_t))^2$ ，这时候反向传播自然就只对后面那一项求梯度了。

所以这个思想真的很简单，根本不需要显式分出来两个网络，其实就是先固定网络算数据，算完数据再更新网络，用 $\{s_t,a_t,\hat y\}$ 去优化 $(\hat y-Q_\theta(s_t,a_t))^2$，
而不是直接用 $\{s_t,a_t,r_{t+1},s_{t+1}\}$ 去优化 $\left( r_{t+1}+\gamma \max_{a_{t+1}}Q_\theta(s_{t+1},a_{t+1})-Q_\theta(s_t,a_t)\right)^2$.

##### Replay buffer (随机采样和 Mini-Batch)

这个也很简单，就是把所有采样的经验全部存起来，维护一个 buffer $\mathcal B=\{(s,a,r,s')_i|i=1,2,...\}$ ，然后每次做梯度下降的时候随机采样出一个 mini-batch 来计算梯度，使用 Mini-Batch Gradient Desent。

这里有两个要点，随机采样 和 Mini-batch。随机采样保证了监督数据 i.i.d，Mini-batch 降低了梯度估计的方差，这两点合力一起保证了梯度估计的准确性。具体分析见后续章节 [番外：为什么需要使用经验回放池 Replay buffer？](#番外：为什么需要使用经验回放池 Replay buffer？).

##### 伪代码

由于 DQN 是 off-policy 算法，它既采用 on-policy 更新（即边改进策略，边采样），也可以采用 off-policy 更新（即使用其他策略采样得到的结果不断更新），二者均可。off-policy 算法不在乎数据是如何得到的，只要有足够的数据就行。

[DQN原论文](https://arxiv.org/pdf/1312.5602) 给出的是 on-policy 更新的算法

> - 初始化 state value function $Q_\theta(s,a)$ 和经验回放池 $\mathcal B$
>
> - for $e=1\rightarrow E:$
>
>   - 随机选取初始状态 $s_0$ 
>
>   - for $t=1\rightarrow T$:
>
>     - 用 $\epsilon$-greedy 策略根据 $Q_\theta(s_{t+1},a)$ 采样动作 $a_{t}$；环境给出 $r_{t+1}$；环境演化得到 $s_{t+1}$；
>
>     - 将这条经验加入经验回放池: $\mathcal B\leftarrow\{s_t,a_t,r_{t+1},s_{t+1}\}$
>
>     - 从 $\mathcal B$ 中采样 $M$ 条经验 $\{(s,a,r,s')_i|i=1,2,...,M\}$，计算
>       $$
>       \hat y_i=r_{i}+\gamma\max_{a'}Q_\theta(s'_i,a
>       ')
>       $$
>       得到本轮监督数据 $\{(s,a,\hat y)_i|i=1,...,M\}$；
>
>     - 对损失
>       $$
>       L=\frac1M\sum_{i=1}^{M}\left( \hat y_i-Q_\theta(s_i,a_i) \right)^2
>       $$
>       做一步 Mini-Batch Gradient Descent，此时 $\hat y_i$ 视为常数，与 $\theta$ 无关；
>
>   - end for
>
> - end for

但需要注意，这里的 on-policy 仅仅是指更新方式，这个算法本质上还是 off-policy 的，因为它在更新策略时使用了经验回放池里的数据，这些数据是由之前的策略的产生的。

也可以给出更贴近 Tabular Q-learning 章节的 off-policy 更新方式

> - 遵循一个随机策略 $\pi_b$ （称为 behaviour policy）在环境中进行大量采样，得到经验回放池 $\mathcal B$
>
> - 初始化 state value function $Q_\theta(s,a)$ 
>
> - for $i=1\rightarrow I:$
>
>   - 从 $\mathcal B$ 中采样 $M$ 条经验 $\{(s,a,r,s')_i|i=1,2,...,M\}$，计算
>     $$
>     \hat y_i=r_{i}+\gamma\max_{a'}Q_\theta(s'_i,a
>     ')
>     $$
>     得到本轮监督数据 $\{(s,a,\hat y)_i|i=1,...,M\}$
>
>   - 对损失
>     $$
>     L=\frac1M\sum_{i=1}^{M}\left( \hat y_i-Q_\theta(s_i,a_i) \right)^2
>     $$
>     做一步 Mini-Batch Gradient Descent，此时 $\hat y_i$ 视为常数，与 $\theta$ 无关
>
> - end for

 注意，DQN 算法不是显式维护一个策略，而是只维护 $Q_\theta$， 最终的最优策略就躺在 $Q_\theta(s,a)$ 里。

#### 番外：为什么需要使用经验回放池 Replay buffer？

注意到在 Tabular Q-learning 中我们从来没有提到 replay buffer 的概念，为什么到了 DQN 要引入这样一种设计呢？在《强化学习的数学原理》第8课 6:00 左右的地方，赵老师给出了解释，这里结合《动手学强化学习》7.3.1部分的内容再进行整理和讲解。

首先，需要意识到一件事情，在 Tabular Q-learning 中，我们事实上是在对每一对 $(s,a)$ 分别求解它的贝尔曼期望方程
$$
Q(s,a)=\mathbb E[r_{t+1}+\gamma\max_{a_{t+1}}Q(s_{t+1},a_{t+1})|s_t=s,a_t=a]
$$

> 也就是说，如果系统存在一百个 (s,a) 状态，那就有100个这样的方程待求解，你需要采样足够足够多的 state transitions，确保每个 (s,a) 上发生的 state transitions 都能被访问到，才能有效求到全局最优。

并且，对 $Q$-table 中每个 $(s,a)$ 的值，都有 TD 更新公式
$$
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
$$
这就可以注意到，事实上 $Q$-table 中的每个格子值的单次更新是互相独立的，互不干扰，更新一个值不会改变其他的值。这种情况下你可以一个数据一个数据地进行更新。

但是 DQN 是一种深度学习方法，它是在所有的 $\{s,a,r,s'\}$ 的分布上最小化一个期望的目标损失 $\mathcal L(\theta)$，这个目标损失的梯度也是建立在整个分布上的一个期望，这时的更新牵一发而动全身，更新一次 $Q_\theta^\pi$ 网络的参数，所有 $q(s,a)$ 的估计值或多或少都会受到影响。

由于 SGD 使用单个数据点上的梯度作为整体期望梯度的估计，方差比较大。为了保证梯度稳定，或者说为了保证每次更新使用的梯度是真实梯度的一个良好估计，一般需要使用 Mini-Batch Gradient 来降低梯度估计的方差，且这个 Mini-Batch 的采样需要尽可能需要保持独立同分布 i.i.d，这样才能用有限的数据对整体期望梯度进行相对准确的估计。

一句话总结，**DQN 的 replay buffer 保证了监督数据 i.i.d. , 而 Mini-Batch 则降低了梯度估计的方差，这两点合力保证了梯度估计的准确性和稳定性，这是深度学习的要求。**

因此，Replay buffer 的使用是我们从强化学习迈入深度强化学习的一个标志。

最后，Tabular Q-learning 也可以使用 replay buffer 随机采样的方式来进行更新，这样一个样本可能被多次使用，提高样本效率。我在 Q-learning 章节中给出的伪代码事实上已经使用了 replay buffer。

#### 番外：实践中具体如何进行梯度下降？

在本章 Value approximation 系列的算法中，我的推导和《强化学习的数学原理》中不太一样，课程中给出的这一部分算法的 pseudocode 在每一步都显示地写出了梯度的具体形式，然后手动在做梯度下降，但我这里所有的推导都只给到 loss function，pseudocode 也只写到 loss function，至于具体如何做梯度下降，那是深度学习的事情，可以通过很成熟的反向传播、自动求导机制高效完成。因此，我们理解算法只需要明白它的 loss 是什么、如何收集监督数据即可，因为你有了loss和监督数据，就可以直接套用深度学习的梯度下降、反向传播等工具进行神经网络的更新了，这一点在后续更加现代的算法例如 TRPO、PPO 中更加得到体现。

但有些时候显式地写出梯度的具体形式有助于我们理解算法的本质，例如后面的 Policy Gradient 算法，但要记得在实际使用中采用 loss function + 反向传播 的方式进行更新。

#### 番外：$Q_\theta$ 网络的具体形式

注意到在 DQN 中我们总是需要求解 $\max_aQ_\theta(s,a)$ 这样一个量，如果按照这一章最开始的想法，把神经网络设计为 $(s,a)\rightarrow q_\pi(s,a)$ 即输入 $s$ 和 $a$ 输出一个值，那将非常麻烦。[DQN原论文](https://arxiv.org/pdf/1312.5602) Section 4.1 明确提到他们的设计是 $Q_\theta:s\rightarrow \{q_\pi(s,a)|\forall a\in\mathcal A(s)\}$，即输入 $s$ 输出所有动作的 action value，这样只需要一次 forward 就可以得到所有动作的 action value 并求解 $\max_aQ_\theta(s,a)$ 了。

#### 番外：Online/Offline vs. OnPolicy/OffPolicy

Online/Offline 取决于在策略更新的过程中是否能和环境交互产生新的数据。

OnPolicy/OffPolicy 取决于策略更新所用的数据是不是来自于当前策略和环境交互得到的。


