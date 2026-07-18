# Reinforcement Learning (High-Level)

An **agent** learns by interacting with an **environment**, receiving rewards or penalties for its actions.

## Core Concepts

- **Agent** — the decision-maker (e.g., game-playing AI)
- **Environment** — the world the agent acts in (e.g., chess board, robot's physical space)
- **Action** — what the agent does (e.g., move piece, spin motor)
- **State** — current situation (e.g., board position, sensor readings)
- **Reward** — feedback signal (+1 for win, -1 for loss)
- **Policy** — the strategy the agent follows (mapping state → action)

The goal: maximize **cumulative reward** over time.

## Examples

1. **Game playing (AlphaGo)**: Agent plays Go against itself millions of times. Reward = +1 for winning, -1 for losing. Learns strategies no human ever discovered.
2. **Robot navigation**: Robot in a warehouse. Actions = move forward, turn left, turn right. Reward = +10 for reaching destination, -1 per collision, -0.01 per step (penalize dawdling). Learns to navigate efficiently.
3. **Recommendation systems (limited)**: Agent recommends articles; reward = click (positive) or bounce (negative). Learns to balance showing familiar content (exploit) vs trying new content (explore).

## Exploration vs Exploitation

The fundamental RL dilemma:
- **Exploit**: Take actions you know give high reward (safe, but may miss better options)
- **Explore**: Try new actions to discover potentially higher rewards (costs short-term reward)

Too much exploit → stuck in local optima. Too much explore → never accumulates reward.

## Key Distinction

RL is different from supervised learning:
- No labeled dataset of "correct actions"
- Learning from delayed reward (a chess move 20 turns before checkmate may be brilliant or terrible)
- Agent's actions affect future data it sees (non-i.i.d. data)
