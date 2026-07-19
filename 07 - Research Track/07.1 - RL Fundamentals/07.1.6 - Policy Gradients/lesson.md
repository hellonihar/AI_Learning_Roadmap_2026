# 07.1.6 — Policy Gradients

## Overview

Policy gradient methods directly parameterise and optimise the policy $\pi_\theta(a \mid s)$ without learning a value function (though many use a critic). The gradient of expected return with respect to $\theta$ is estimated from samples, enabling learning in continuous action spaces and stochastic policies.

## The Policy Gradient Theorem

The expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$ has gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, Q^{\pi_\theta}(s, a) \right]
$$

This is remarkable: the gradient depends only on the score function $\nabla_\theta \log \pi_\theta$ and the action-value $Q$, not on the unknown environment dynamics.

## REINFORCE — Monte Carlo Policy Gradient

REINFORCE uses the complete return $G_t$ as an unbiased estimate of $Q$:

$$
\nabla_\theta J(\theta) \approx \sum_t G_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

The update increases the log-probability of actions that led to high returns and decreases it for low-return actions. REINFORCE is unbiased but has high variance.

## Actor-Critic

Adding a critic (a learned value function $V_\phi(s)$) reduces variance. The **advantage function** $A(s,a) = Q(s,a) - V(s)$ measures how much better an action is than average:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, A(s, a) \right]
$$

The actor (policy) and critic (value function) are learned jointly. The critic provides a baseline that reduces variance without introducing bias.

## A2C / A3C

**A2C** (Advantage Actor-Critic) computes advantages as $A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k}) - V(s_t)$. A3C uses asynchronous parallel workers; A2C uses synchronous workers with comparable performance and simpler implementation.

## PPO — Proximal Policy Optimization

PPO constrains policy updates to prevent catastrophic collapse. It clips the probability ratio to stay within $[1-\varepsilon, 1+\varepsilon]$:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$. PPO is the default RL algorithm for LLM alignment (RLHF).

## DDPG — Deep Deterministic Policy Gradient

For continuous control, DDPG maintains a deterministic policy $\mu_\theta(s)$ and learns $Q$ via DQN-style updates. The policy gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_a Q(s,a) \big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]
$$

DDPG uses experience replay and target networks for stability.

## Practical Context

Policy gradients are the foundation of modern RL applications: robotics (continuous control with SAC, TD3), games (PPO in Dota 2), and crucially, **LLM alignment** (RLHF uses PPO to fine-tune language models based on human preferences).

## Code Example

See `code_walkthrough.md` in the Research Track root for a numpy implementation of REINFORCE on a Gym environment.
