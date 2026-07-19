# 07.1.1 — Reinforcement Learning Setup

## Overview

Reinforcement Learning (RL) is a paradigm where an **agent** learns to make sequential decisions by interacting with an **environment**. Unlike supervised learning, the agent receives no explicit labels; instead, it gets a **reward signal** that tells it how well it is doing. The goal is to learn a **policy** that maximizes cumulative reward over time.

## The Markov Decision Process (MDP)

All of RL is built on the MDP formalism. An MDP is defined by the tuple $(S, A, P, R, \gamma)$:

- **$S$**: set of states (what the agent observes about the environment)
- **$A$**: set of actions the agent can take
- **$P(s' \mid s, a)$**: transition probability — the dynamics of the environment
- **$R(s, a)$**: reward function — the scalar feedback signal
- **$\gamma \in [0, 1)$**: discount factor — how much we care about future rewards

At each timestep $t$, the agent observes $s_t$, chooses $a_t$, receives $r_t = R(s_t, a_t)$, and transitions to $s_{t+1} \sim P(\cdot \mid s_t, a_t)$. The return is the discounted sum:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

## Key Components

**Policy $\pi(a \mid s)$**: The agent's behaviour — a mapping from states to actions (deterministic or stochastic). The goal is to find an optimal policy $\pi^*$ that maximises expected return.

**Value Function $V^\pi(s)$**: The expected return starting from state $s$ and following policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]
$$

**Action-Value Function $Q^\pi(s, a)$**: The expected return starting from $s$, taking action $a$, then following $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

These functions satisfy recursive Bellman equations:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
$$

## Exploration vs Exploitation

A central tension in RL: the agent must **exploit** known high-reward actions while **exploring** unknown actions that might lead to even higher reward. This trade-off appears in every RL algorithm.

## Types of RL Agents

| Type | What it learns | Examples |
|------|---------------|----------|
| **Value-based** | $Q(s,a)$, derive policy implicitly | DQN |
| **Policy-based** | $\pi(a \mid s)$ directly | REINFORCE, PPO |
| **Actor-Critic** | Both $\pi$ and $V$ | A2C, SAC |

## Practical Context

Modern RL powers breakthroughs from game-playing (AlphaGo, Dota 5v5) to robotics (dexterous manipulation) to LLM alignment (RLHF). Frameworks like Gymnasium, Stable-Baselines3, and Ray RLlib make it easy to experiment. Understanding the MDP formalism is the first step toward building systems that learn from interaction.

## Connection to LLMs

RL is crucial for aligning large language models. Reinforcement Learning from Human Feedback (RLHF) treats the LLM as a policy that generates text, and a learned reward model provides the reward signal. The policy is optimised with PPO to produce outputs that humans prefer — directly applying the MDP framework to language generation.
