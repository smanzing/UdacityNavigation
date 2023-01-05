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

$$a_t = \argmax_{a \in \mathcal{A}} Q(s, a),$$

where the action-value function $Q$ yields the expected return for a state $s \in \mathcal{S}$ and action $a \in \mathcal{A}$ 
under the assumption that the agent takes action $a$ in state $s$ and follows the policy for all future time steps. 
The action-value function is also known as Q-value function.

We can learn the optimal Q-value function using Q-Learning and variants thereof. 
In this project, we use a combination of the algorithms presented in [[1]](#1), [[2]](#2), and [[3]](#3), to solve the 
reinforcement learning problem. Before reviewing the the main findings of [[1]](#1), [[2]](#2), and [[3]](#3), we
briefly explain the basics of Q-learning.

### 2.1 Q-Learning

In Q-learning, we iteratively update the estimated Q-values over consecutive time steps until convergence. At each time 
step $t$, the agent takes an action $a_t$ in state $s_t$. Then, the agent receives a reward $r_{t+1}$ and reaches the 
next state $s_{t+1}$. Before taking the next action $a_{t+1}$, the information of the tuple $e_t = (s_t, a_t, r_{t+1}, s_{t+1})$
--also referred to as experience--is used to update the current estimate of the Q-value function. The update rule is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime) - Q(s_t, a_t)),$$

where $\alpha \in ]0, 1]$ is the learning rate, $\gamma\in [0, 1[$ is the discount factor, and 
$(r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime) - Q(s_t, a_t))$ is the temporal difference error.
Note that we assume that we take the action $a^\prime$ that maximizes the Q-value function for the next state $s_{t+1}$ for
computing the time difference error; however, other rules can be applied as well.

For discrete environments with a small, finite number of states and actions, the Q-value function can be represented as table. 
However, this becomes computationally intractable for large state/action spaces and continuous state/action spaces. 
Instead of using a finite structure to represent the Q-value function, we can use function approximation to approximate 
the Q-value function, i.e., we try to approximate the real Q-value function with a linear or non-linear function.
In this project, we use neural networks to approximate the Q-value function. This leads to *Deep Q-Learning* [[1]](#1), 
which is explained next.


### 2.1 Deep Q-Learning Algorithm
In  [[1]](#1), neural networks are used as function approximator for the Q-value function. Let us refer to the neural network
with weights $\theta$ as Q-network $Q(s, a; \theta)$. We can then rewrite the temporal difference error as:

$$\delta_t = r_{t+1} + \gamma \max_{a^\prime \in \mathcal{A}} Q(s_{t+1}, a^\prime; \theta) - Q(s_t, a_t; \theta)$$

Thus, we need to find a set of weights $\theta$ that yield an optimal Q-value function. 


### 2.2 Double Deep Q-Learning 

### 2.3 Prioritized Experience Replay

### 2.4 Implementation Details

**Neural Network Architecture**

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