# Report

This report summarizes the implemented methods for the project **Navigation** of the Udacity course 
**Deep Reinforcement Learning**. 

## 1 Introduction

The goal of the project **Navigation** is to train an agent to collect yellow bananas while avoiding blue bananas
in a large, square world. Below, we first give a description of the environment, followed by a short introduction to 
reinforcement learning.

### Problem Statement

We train the agent using reinforcement learning. 
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
avoiding blue bananas in the world.

The task is episodic and terminates after the agent has taken a maximum number of actions. The environment is solved, 
if the agent achieves an average score of +13 over 100 consecutive episodes.

## 2 Method

The agent needs to learn the best action to take in each state of the environment. We refer to the mapping from states 
to actions as policy. For each policy, we have a action-value function. The action-value function yields the expected
return for a state $s \in \mathcal{S}$ and action $a \in \mathcal{A}$ under the assumption that the agent takes action $a$ 
in state $s$ and follows the policy for all future time steps. The action-value function is also known as Q-value function.

For discrete environments with a small, finite number of states, the Q-value function can be represented as table. 
However, in large state spaces this becomes computationally intractable. Instead, function approximation can be used to 
approximate the Q-value function. In this project, we use neural networks to approximate the Q-value function. 

Using neural networks for approximating the Q-value function leads to deep reinforcement learning.
A well-known algorithm is *Deep Q-Learning* to learn the Q-value function, see [[1]](#1). 


Below, we review the main findings of [[1]](#1), [[2]](#2), and [[3]](#3), followed by clarifying implementation details.

### 2.1 Deep Q-Learning Algorithm

Typically, with large state/action spaces,  

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