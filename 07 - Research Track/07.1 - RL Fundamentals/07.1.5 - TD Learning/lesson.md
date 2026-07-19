# 07.1.5 — Temporal Difference Learning

## Overview

Temporal Difference (TD) learning combines ideas from Monte Carlo (model-free, learns from experience) and Dynamic Programming (bootstraps from current estimates). TD methods update value estimates **online**, after every timestep, without waiting for the episode to finish. This makes them the most widely used class of RL algorithms.

## TD(0) — The Simplest TD Algorithm

TD(0) updates the value of a state using the observed reward and the estimated value of the next state:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right)
$$

The term $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**. It measures how "surprising" the transition was. TD(0) converges to $V^\pi$ for fixed $\pi$ under standard stochastic approximation conditions.

## SARSA — On-Policy TD Control

SARSA learns the action-value function $Q(s,a)$ while following policy $\pi$:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right)
$$

The name comes from the tuple $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$. SARSA is **on-policy**: it evaluates and improves the same policy it uses for exploration.

## Q-Learning — Off-Policy TD Control

Q-learning directly approximates the optimal action-value function $Q^*$, independent of the policy being followed:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right)
$$

Q-learning is **off-policy**: it learns about the greedy (optimal) policy while following any exploratory policy. This separation makes it more sample-efficient and is the foundation of Deep Q-Networks (DQN).

## TD($\lambda$) — Eligibility Traces

Eligibility traces bridge the gap between TD(0) (one-step bootstrapping) and Monte Carlo (full-return). The TD($\lambda$) algorithm maintains an **eligibility trace** $e_t(s)$ for each state, decayed by $\lambda\gamma$:

$$
e_t(s) = \begin{cases}
\gamma \lambda e_{t-1}(s) + 1 & \text{if } s = s_t \\
\gamma \lambda e_{t-1}(s) & \text{otherwise}
\end{cases}
$$

The update becomes:

$$
V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s)
$$

When $\lambda = 0$, this reduces to TD(0). When $\lambda = 1$, it approaches MC. Intermediate values offer a bias-variance trade-off.

## Practical Context

TD methods are the workhorse of modern deep RL. DQN (Q-learning with neural networks) achieved human-level Atari play. SARSA and Q-learning variants underpin most model-free RL algorithms.

## Connection to LLMs

Q-learning-inspired approaches are being explored for **process reward models** in LLM reasoning. TD methods also appear in **stepwise reward models** for chain-of-thought, where partial credit is assigned to intermediate reasoning steps rather than only the final answer.
