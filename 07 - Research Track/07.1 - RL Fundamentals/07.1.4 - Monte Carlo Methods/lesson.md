# 07.1.4 — Monte Carlo Methods

## Overview

Monte Carlo (MC) methods learn from complete **episodes** of experience — no model required. The agent simply plays through an entire trajectory, observes the actual returns $G_t$, and updates its value estimates toward those observed returns. MC methods are the first class of **model-free** RL algorithms.

## MC Prediction (Evaluating a Policy)

To estimate $V^\pi(s)$, MC methods average the observed returns after each visit to $s$:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left( G_t - V(s_t) \right)
$$

There are two variants:

- **First-visit MC**: only the first occurrence of $s$ in each episode is used. This is unbiased and has better theoretical properties.
- **Every-visit MC**: every occurrence is used. Lower variance but biased.

MC returns are unbiased estimates of $V^\pi$ — they do not use bootstrapping, unlike DP or TD methods.

## MC Control (Finding the Optimal Policy)

To find an optimal policy, combine MC prediction with **policy improvement**. The key challenge is ensuring sufficient exploration. Two approaches:

### On-Policy MC Control

Learn about the policy you're currently executing. Use **exploring starts** (every state-action pair has non-zero probability of being the start) or $\varepsilon$-soft policies:

$$
\pi(a \mid s) \ge \frac{\varepsilon}{|A|}
$$

The canonical algorithm is MC $\varepsilon$-greedy improvement: after each episode, update $Q(s,a)$ toward $G_t$, then set the policy to be $\varepsilon$-greedy with respect to $Q$.

### Off-Policy MC Control

Learn about a **target policy** $\pi$ while following a different **behaviour policy** $b$. This enables learning from data collected by any source. Importance sampling corrects the distribution mismatch:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k \mid s_k)}{b(a_k \mid s_k)}
$$

The off-policy MC update becomes:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left( \rho_{t:T-1} G_t - V(s_t) \right)
$$

Importance sampling introduces high variance, especially for long episodes.

## Practical Context

MC methods are conceptually simple and unbiased, but their variance is high and they cannot learn online (must wait for episode completion). They are most useful when episodes are short and the environment is stochastic. Modern RL relies more on TD methods, but MC ideas persist in **Monte Carlo Tree Search** (MCTS), used in AlphaGo and AlphaZero.

## Connection to LLMs

Off-policy learning is crucial for RLHF: the behaviour policy (the SFT model generating responses) differs from the target policy (the aligned model). Off-policy corrections help the reward model learn from stale data.
