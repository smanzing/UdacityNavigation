# Report

This report summarizes the implemented methods for the project **Navigation** of the Udacity course 
**Deep Reinforcement Learning**. 

## 1 Introduction

The goal of the project **Navigation** is to train an agent to collect yellow bananas while avoiding blue bananas
in a large, square world. Below, we give a description of the problem statement.

### Problem Statement

We train an agent using reinforcement learning. 
Reinforcement learning is modeled as a Markov Decision Process, in which we have a 

- set of states $\mathcal{S}$,
- set of actions $\mathcal{A}$,
- a set of rewards,
- the one-step dynamics of the environment.

In the project **Navigation**, the state space has 37 dimensions which consist of the agent's velocity 
and a ray-based perception of objects in front of the agent. The action space consists of 4 discrete actions that
are moving forward, moving backward, turning left, and turning right. The agent receives a reward of +1 
for collecting yellow bananas and a reward of -1 for collecting blue bananas.

We aim to train an agent to interact with its environment such that the expected return, i.e., 
the (discounted) cumulative reward, is maximized. Thus, the agent must collect as many yellow bananas as possible, while 
avoiding blue bananas.

The task is episodic and terminates after the agent has taken a maximum number of actions. The environment is solved, 
if the agent achieves an average score of +13 over 100 consecutive episodes.

## 2 Method

We aim to learn the best action for each state of the environment. We refer to the mapping from states 
to actions as policy. A policy can be derived from an action-value function $Q: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{R}$:

$$a_t = \arg \max_{a \in \mathcal{A}} Q(s, a),$$

where the action-value function $Q$ yields the expected return for a state $s \in \mathcal{S}$ and action $a \in \mathcal{A}$ 
under the assumption that the agent takes action $a$ in state $s$ and follows the policy for all future time steps. 
The action-value function is also known as Q-value function.

We can learn the optimal Q-value function using Q-Learning and variants thereof. 
In this project, we use a combination of the algorithms presented in [[1]](#1), [[2]](#2), and [[3]](#3), to solve the 
reinforcement learning problem. Before reviewing the main findings of [[1]](#1), [[2]](#2), and [[3]](#3), we
briefly explain the basics of Q-learning.

### 2.1 Q-Learning

In Q-learning, we iteratively update the estimated Q-values over consecutive time steps until convergence. At each time 
step $t$, the agent takes an action $a_t$ in state $s_t$. Then, the agent receives a reward $r_{t+1}$ and reaches the 
next state $s_{t+1}$. Before taking the next action $a_{t+1}$, the information of the tuple $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$
--also referred to as experience--is used to update the current estimate of the Q-value function. The update rule is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime) - Q(s_t, a_t)),$$

where $\alpha \in ]0, 1]$ is the learning rate, $\gamma\in [0, 1[$ is the discount factor, and

$$\Delta = (r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime) - Q(s_t, a_t))$$

is the temporal difference error. Note that we assume that we take the action $a^\prime$ that maximizes the Q-value 
function for the next state $s_{t+1}$ for computing the time difference error; however, other rules can be applied as well.

For discrete environments with a small, finite number of states and actions, the Q-value function can be represented as table. 
However, this becomes computationally intractable for large state/action spaces and continuous state/action spaces. 
Instead of using a finite structure to represent the Q-value function, we can use function approximation to approximate 
the Q-value function, i.e., we try to approximate the real Q-value function with a linear or non-linear function.
In this project, we use neural networks to approximate the Q-value function. This leads to *Deep Q-Learning* [[1]](#1), 
which is explained next.


### 2.1 Deep Q-Learning Algorithm

In [[1]](#1), neural networks are used as function approximator for the Q-value function. Let us refer to the neural network
with weights $\theta$ as Q-network $Q(s, a; \theta)$. Using a Q-network as function approximator reduces to problem of 
finding the optimal Q-value function to finding a set of weights $\theta$ that yield an optimal Q-value function. 
To this end, we can directly use the temporal difference error to adjust the weights of the Q-network and define the following 
loss function for training:

$$L = \alpha \big(r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta^-) - Q(s_t, a_t; \theta)\big)^2,$$

where $r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta^-)$ is a replacement for the true Q-value
function and $Q(s_t, a_t; \theta)$ is the current Q-value given the weights $\theta$. Using gradient descent, we obtain 
the following update rule for the weights $\theta$:

$$\Delta \theta = \alpha \big(r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta^-) - Q(s_t, a_t; \theta)\big) \nabla_{\theta}  Q(s_t, a_t; \theta).$$

Using vanilla Q-Learning with Q-networks as function approximator can potentially lead to harmful correlations. There are 
two issues, namely:

1. Learning from each experience tuple sequentially can lead to oscillating or diverging Q-values, since sequential 
  experiences can be highly correlated,
2. Our estimated target $r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta^-)$ is depending on
the Q-network's weights which can also lead to correlations that prevent convergence of the Q-values. 

In [[1]](#1), it is suggested to keep track of experiences and to store them in a replay buffer to tackle issue 1. We then randomly sample small batches
of experiences from the replay buffer to train the Q-network. To avoid issue 2., it is suggested to keep the weights $\theta^-$ of 
the target Q-values fixed for $N$-steps and are then updated with the current Q-network weights $\theta$.

### 2.2 Double Q-Learning 

Particularly in early iterations of learning, the Q-value target $r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta^-)$
may overestimate the true Q-value due to the max-operator. In [[2]](#2), Double Q-Learning is suggested to overcome the 
overestimation issue. To this end, the Q-value target is changed to


$$r_{t+1} + \gamma Q(s_{t+1}, \arg \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta); \theta^-).$$

### 2.3 Prioritized Experience Replay

the work of [[3]](#3) proposes to improve the utilization of experiences. There may be some experiences that are more important,
i.e., the agent can learn more from them, as other experiences. The priority $p_i$ of the $i$-th experience is the magnitude of the
temporal difference error:

$$p_i = |\Delta_i|+ E,$$ 

with $E > 0$ to ensure that experiences can also be sampled in case $|\Delta_i| = 0$. A greater temporal difference error 
corresponds to a higher priority. The priority is used to compute the sampling probability for each experience when:

$$P_i = \frac{p_i^a}{\sum_k p_k^a},$$

where the denominator is the sum of priorities of all experiences in the replay buffer and $a$ is a parameter that prevents 
overfitting to a small set of experiences, see [[3]](#3). Basically, higher values of $a$ mean that the priority of 
experiences has high influence on their sampling probability. In contrast, $a=0$ corresponds to uniform sampling.
To compensate for non-uniform sampling probabilities, the update rule for the Q-network weights is changed to

$$\Delta \theta = \sum_i \frac{(K P_i)^{-b}}{\max_j (K P_j)^{-b}} \Delta_i \nabla_{\theta_i}  Q(s_t, a_t; \theta_i),$$

where $(K P_i)^{-b}$ is the importance-sampling weight, $K$ is the length of the replay buffer, and $\frac{1}{\max_j (K P_j)^{-b}}$ 
normalizes the importance-sampling weights with $j$ iterating over all experiences in the current batch.

### 2.4 Implementation Details

The implementation follows Algorithm 1 in [[3]](#3), which is Double Q-Learning with prioritized experience replay. 
Following steps are executed for each episode:

- Create a new experience $e_t$ by interacting with the environment
- Store $e_t$ in the replay buffer
- Every $c$-th step
  - Sample batch from replay buffer
  - Compute importance-sampling weight for each experience $e_i$ in batch
  - Compute temporal difference error $\Delta_i$ for all $e_i$ in batch
  - Update priorities $p_i$ for all $e_i$ in batch
  - Compute loss function $L$ and update Q-network weights $\theta$
- Every $l$-th step
  - Update Q-value target weights $\theta^-$ using a soft update: $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$ 
- Select epsilon-greedy action

After each episode, we decay $\epsilon$ for the epsilon-greedy action selection. The most important parameters are stated in the table below.

**Replay Buffer Parameters**:

| Parameter   | Value |
|-------------|-------|
| buffer size | 1e5   |
| batch size  | 64    | 
| a           | 0.7   |
| b           | 0.5   | 
| E           | 1e-6  | 

**Double Q-Learning Parameters**:

| Parameter                | Value |
|--------------------------|-------|
| discount factor $\gamma$ | 0.99  |
| soft update $\tau$       | 1e-3  | 
| learning rate $\alpha$   | 5e-4  |
| Update step $c$          | 2     | 
| Update step $l$          | 4     | 

**$\epsilon$ Decay**

| Parameter     | Value |
|---------------|-------|
| Start value   | 1.0   |
| Minimum value | 0.01  | 
| Decay rate    | 0.995 |

The maximum number of time steps for each episode is limited to 1000. 

The Q-network architecture consists of three linear layers. The parameters are given in the table below.

**Q-Network Parameters**:

| Layer Nr. | Input Size | Output Size | Activation Function |
|-----------|------------|-------------|---------------------|
| 1         | state size | 64          | ReLu                |
| 2         | 64         | 64          | ReLu                |
| 3         | 64         | action size |                     |


## 3 Evaluation

## 4 Future Work


## References

<a id="1">[1]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* 
**Human-level control through deep reinforcement learning.**
*Nature* **518**, 529–533 (2015).

<a id="1">[2]</a>
Hasselt, H. van and Guez, A. and Silver, D.
**Deep Reinforcement Learning with Double Q-Learning**
*Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence*, 2094–2100 (2016).

<a id="1">[3]</a>
Schaul, T. and Quan, J. and Antonoglou, I. and Silver, D.
**Prioritized Experience Replay**
*arXiv* **10.48550/ARXIV.1511.05952**, (2015).
