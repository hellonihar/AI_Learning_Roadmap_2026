# 07.1.3 — Dynamic Programming

## Overview

Dynamic Programming (DP) refers to a set of algorithms for computing optimal policies in an MDP when the model (transition probabilities $P$ and reward function $R$) is **fully known**. DP is the theoretical foundation for all modern RL — every subsequent algorithm can be understood as an approximation to DP when the model is unknown.

## Policy Evaluation

Given a policy $\pi$, policy evaluation computes its value function $V^\pi$ using the Bellman expectation equation as an update rule:

$$
V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V_k(s') \right]
$$

This is a system of $|S|$ linear equations. Iterative application converges to $V^\pi$ as $k \to \infty$. In matrix form:

$$
V_{k+1} = R^\pi + \gamma P^\pi V_k
$$

## Policy Improvement

Once we have $V^\pi$, we can improve the policy by acting greedily with respect to it:

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
$$

The **policy improvement theorem** guarantees $\pi' \ge \pi$ (pointwise).

## Policy Iteration

Alternate between evaluation and improvement until convergence:

1. Evaluate $V^{\pi_k}$
2. Improve: $\pi_{k+1} = \text{greedy}(V^{\pi_k})$
3. Repeat

Policy iteration converges in finite time (at most $|A|^{|S|}$ iterations, but typically far fewer).

## Value Iteration

A special case where evaluation is truncated to a single sweep before improvement:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V_k(s') \right]
$$

Value iteration is an application of **contraction mapping** — the Bellman optimality operator $\mathcal{T}^*$ is a $\gamma$-contraction in the sup norm:

$$
\| \mathcal{T}^* V - \mathcal{T}^* V' \|_\infty \le \gamma \| V - V' \|_\infty
$$

## Practical Context

DP requires a complete model of the environment, which is rarely available in real-world problems. However, the ideas of **bootstrapping** (estimating values using other value estimates) and **generalised policy iteration** (GPI: the interaction between evaluation and improvement) are the backbone of every RL algorithm from DQN to PPO.

## Connection to LLMs

While LLM training doesn't use DP directly, the concept of **iterative improvement** mirrors RLHF: the reward model evaluates (critic), the LLM improves (actor), and the loop repeats. Understanding GPI helps reason about convergence in alignment pipelines.
