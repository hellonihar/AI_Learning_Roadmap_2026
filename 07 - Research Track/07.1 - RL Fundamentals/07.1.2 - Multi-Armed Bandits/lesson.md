# 07.1.2 — Multi-Armed Bandits

## Overview

Multi-armed bandits are the simplest RL setting: a single state, multiple actions (arms), and each action yields a reward drawn from a fixed (but unknown) distribution. There is no sequential state transition — only repeated independent decisions. This isolates the **exploration vs exploitation** dilemma in its purest form.

## Problem Formulation

You have $k$ arms. Pulling arm $i$ at time $t$ yields reward $r_t \sim \mathcal{D}_i$ where $\mathcal{D}_i$ is unknown. The goal is to minimise **regret**:

$$
R_T = T \cdot \mu^* - \sum_{t=1}^T \mu_{a_t}
$$

where $\mu^* = \max_i \mu_i$ is the expected reward of the optimal arm, and $\mu_{a_t}$ is the expected reward of the chosen arm.

## Algorithms

### $\varepsilon$-Greedy

With probability $\varepsilon$, explore (pick a random arm); with probability $1-\varepsilon$, exploit (pick the arm with highest empirical mean). Simple but effective. Decaying $\varepsilon$ over time provably converges.

$$\hat{\mu}_i = \frac{1}{n_i} \sum_{t: a_t = i} r_t$$

### Upper Confidence Bound (UCB)

UCB selects arms optimistically by adding an uncertainty bonus:

$$a_t = \arg\max_i \left( \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}} \right)$$

This implements "optimism in the face of uncertainty." UCB1 achieves logarithmic regret: $R_T = O(\log T)$.

### Thompson Sampling

A Bayesian approach: maintain a posterior distribution $p(\mu_i \mid \text{data})$ for each arm (typically Beta for Bernoulli rewards). At each step, sample $\tilde{\mu}_i \sim p(\mu_i)$, then pick $a_t = \arg\max_i \tilde{\mu}_i$. Thompson Sampling naturally balances exploration and exploitation and matches UCB's theoretical guarantees while often performing better in practice.

$$
\text{Prior: } \mu_i \sim \text{Beta}(\alpha_i, \beta_i) \quad
\text{Posterior: } \mu_i \mid \text{data} \sim \text{Beta}(\alpha_i + s_i, \beta_i + f_i)
$$

## Practical Context

Bandits are used in A/B testing, ad placement, recommendation systems, and hyperparameter tuning. They are also the foundation for understanding exploration in deep RL.

## Connection to LLMs

Bandit algorithms are used for **prompt selection** in LLM chains, **model routing** (choosing between a small and large model per query), and **active learning** for RLHF data collection. The exploration-exploitation trade-off is central to RLHF reward modelling.
