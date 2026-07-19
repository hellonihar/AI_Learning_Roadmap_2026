# Exercises — Research Track

## Exercise 1: MDP Formulation

A robot navigates a 3×3 grid world. The robot starts at (0,0) and must reach (2,2). Each step moves up/down/left/right with cost -1. Reaching (2,2) gives +10 and terminates. Walls block movement.

**(a)** Define the MDP tuple $(S, A, P, R, \gamma)$ for this environment.  
**(b)** If $\gamma = 0.9$, what is the optimal value of the start state?  
**(c)** How does $V^*(s)$ change if $\gamma = 0.5$?

<details>
<summary>Answer</summary>

**(a)**
- $S$: 9 grid positions (excluding walls if any)
- $A$: {up, down, left, right}
- $P(s' \mid s, a)$: deterministic — 1.0 for the intended move, 0 otherwise (bump into wall = stay)
- $R(s,a) = -1$ for all non-terminal transitions, $+10$ at (2,2)
- $\gamma = 0.9$

**(b)** With $\gamma=0.9$, the shortest path is 4 steps: $V^*((0,0)) = \sum_{k=0}^3 -0.9^k + 10\cdot 0.9^4 \approx -3.44 + 6.56 = 3.12$

**(c)** With $\gamma=0.5$, future rewards are discounted more heavily: $V^*((0,0)) = -(1 + 0.5 + 0.25 + 0.125) + 10\cdot 0.0625 \approx -1.875 + 0.625 = -1.25$. The start state now has negative value because the agent cares more about immediate step costs.
</details>

---

## Exercise 2: Bandit Regret

Consider a 3-armed bandit with Bernoulli rewards: $\mu = [0.1, 0.5, 0.8]$. After 1000 pulls using $\varepsilon$-greedy with $\varepsilon=0.1$, the algorithm chose arm 3 (optimal) 850 times, arm 2 100 times, and arm 1 50 times.

**(a)** Compute the cumulative regret.  
**(b)** What would UCB1's regret look like asymptotically?  
**(c)** If Thompson Sampling uses Beta(1,1) priors, derive the posterior after observing 80 successes and 20 failures on arm 3.

<details>
<summary>Answer</summary>

**(a)** $\mu^* = 0.8$. Expected reward per pull = $850/1000 \cdot 0.8 + 100/1000 \cdot 0.5 + 50/1000 \cdot 0.1 = 0.68 + 0.05 + 0.005 = 0.735$. Optimal expected reward = $0.8$. Regret = $(0.8 - 0.735) \times 1000 = 65$.

**(b)** UCB1 achieves $O(\log T)$ regret. After 1000 steps, regret would be roughly $O(\log 1000) \approx O(7)$, significantly lower than $\varepsilon$-greedy's linear regret. UCB is more sample-efficient because it explores systematically rather than randomly.

**(c)** Beta prior (1,1) + Binomial likelihood (80,20) → Beta posterior (1+80, 1+20) = Beta(81, 21). Expected value: $81/(81+21) = 81/102 \approx 0.794$.
</details>

---

## Exercise 3: Policy vs Value Iteration

A 2-state MDP: State A transitions to B (reward +1) with probability 0.5, else stays in A (reward 0). State B is terminal with reward 0. Discount $\gamma=0.9$.

**(a)** Run one sweep of value iteration starting from $V_0 = [0, 0]$.  
**(b)** Compare with one iteration of policy evaluation for the policy that always takes action "go to B".

<details>
<summary>Answer</summary>

**(a)** Value iteration:
$V_1(A) = \max\{0.5(1+0.9\cdot 0) + 0.5(0+0.9\cdot 0), 0\} = \max\{0.5, 0\} = 0.5$
$V_1(B) = 0$ (terminal)
$V_1 = [0.5, 0]$

**(b)** Policy evaluation for $\pi(A)=B$:
$V^\pi(A) = 0.5(1 + 0.9\cdot 0) + 0.5(0 + 0.9 V^\pi(A))$
$V^\pi(A) = 0.5 + 0.45 V^\pi(A)$
$0.55 V^\pi(A) = 0.5$
$V^\pi(A) \approx 0.909$

Value iteration converges faster per iteration but both converge to $V^*$ eventually.
</details>

---

## Exercise 4: Q-Learning Update

Given $Q(s, a) = [2.0, 3.0, 1.5]$ for three actions. After taking action $a=1$ (index 1), receiving reward $r=1$, and reaching $s'$ where $\max_{a'} Q(s', a') = 4.0$, with $\alpha = 0.1, \gamma = 0.9$:

**(a)** Compute the TD target and TD error.  
**(b)** What is the new $Q(s, 1)$?  
**(c)** How would the update differ if this were SARSA and $a' = 2$ with $Q(s', 2) = 3.0$?

<details>
<summary>Answer</summary>

**(a)** TD target: $r + \gamma \max_{a'} Q(s', a') = 1 + 0.9 \cdot 4.0 = 4.6$  
TD error: $4.6 - Q(s, 1) = 4.6 - 3.0 = 1.6$

**(b)** $Q(s,1) \leftarrow 3.0 + 0.1 \cdot 1.6 = 3.16$

**(c)** SARSA uses the actual next action: TD target = $1 + 0.9 \cdot 3.0 = 3.7$, TD error = $3.7 - 3.0 = 0.7$. Q-learning updates toward the optimal (max), SARSA updates toward the actual policy. SARSA is safer (on-policy) while Q-learning is more aggressive (off-policy).
</details>

---

## Exercise 5: PPO Clipping

A PPO update has $r_t(\theta) = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} = 1.7$ and advantage $\hat{A}_t = -0.5$ with $\varepsilon = 0.2$.

**(a)** Compute the clipped surrogate objective contribution for this token.  
**(b)** Explain why clipping matters here.  
**(c)** What if $\hat{A}_t = +0.5$ instead?

<details>
<summary>Answer</summary>

**(a)** $r_t = 1.7$, clipped to $[0.8, 1.2]$, so $r_t^{\text{clip}} = 1.2$.
$L = \min(1.7 \cdot (-0.5), 1.2 \cdot (-0.5)) = \min(-0.85, -0.6) = -0.85$

**(b)** The large ratio (1.7) means the new policy significantly increased this action's probability. Since the advantage is negative (bad action), we want to decrease its probability. Clipping at 1.2 prevents an overly large update that could destabilise the policy.

**(c)** If $\hat{A}_t = +0.5$: $\min(1.7 \cdot 0.5, 1.2 \cdot 0.5) = \min(0.85, 0.6) = 0.6$. Clipping still limits the update, but now it limits how much we increase a good action's probability. The clip prevents the policy from changing too much in one step in either direction.
</details>

---

## Exercise 6: KV Cache Memory

A 70B model has 80 layers, GQA with 8 KV-heads, head dimension 128, FP16.

**(a)** Compute the KV cache size per token.  
**(b)** For a batch of 4 sequences, each with 16K context, how much GPU memory is needed for the KV cache?  
**(c)** How much memory does INT4 KV cache quantisation save?

<details>
<summary>Answer</summary>

**(a)** Per token: $80 \times 8 \times 128 \times 2 \times 2 = 327,680$ bytes = 320 KB. (K and V each, 2 bytes per FP16 element)

**(b)** $4 \times 16,384 \times 320 \text{ KB} = 20.48 \text{ GB}$

**(c)** INT4 uses 0.5 bytes per element instead of 2: $\frac{20.48}{4} = 5.12 \text{ GB}$. Savings: 15.36 GB. This can mean the difference between fitting on one A100-80GB or needing multiple GPUs.
</details>

---

## Exercise 7: Quantization Error

A weight vector $w = [4.2, -1.1, 2.8, -3.5, 0.9, -4.0]$ is quantised to INT4.

**(a)** Compute the symmetric INT4 quantisation scale $\Delta$ for the full vector.  
**(b)** Quantise and dequantise each value. Compute the mean squared error (MSE).  
**(c)** How does block-wise quantisation (block size 3) reduce the error?

<details>
<summary>Answer</summary>

**(a)** INT4 range: [-8, 7]. $\max(|w|) = 4.2$. $\Delta = 4.2 / 7 = 0.6$.  
$w_q = \text{round}(w / 0.6) = [7, -2, 5, -6, 2, -7]$  
$\hat{w} = w_q \cdot 0.6 = [4.2, -1.2, 3.0, -3.6, 1.2, -4.2]$

**(b)** Errors: $[0, -0.1, -0.2, 0.1, -0.3, 0.2]$. MSE = $(0^2 + 0.01 + 0.04 + 0.01 + 0.09 + 0.04) / 6 = 0.19 / 6 \approx 0.032$

**(c)** Block 1: $[4.2, -1.1, 2.8]$, $\max=4.2$, $\Delta=0.6$. Quantised: $[7, -2, 5]$, $\hat{w}=[4.2, -1.2, 3.0]$.  
Block 2: $[-3.5, 0.9, -4.0]$, $\max=4.0$, $\Delta\approx 0.571$. Quantised: $[-6, 2, -7]$, $\hat{w}\approx[-3.43, 1.14, -4.0]$.  
Errors: block 1: $[0, -0.1, -0.2]$, block 2: $[-0.07, -0.24, 0]$.  
MSE lower because $\Delta$ adapts per block.
</details>

---

## Exercise 8: Speculative Decoding

A draft model has acceptance probability $p_{\text{accept}}(x) = \min(1, p(x)/q(x))$. The target model produces $p = [0.4, 0.3, 0.2, 0.1]$ and the draft model produces $q = [0.3, 0.4, 0.1, 0.2]$.

**(a)** For each token, compute the acceptance probability.  
**(b)** What is the expected number of accepted tokens?  
**(c)** If the target model takes 100ms per forward pass and the draft model takes 10ms per pass, what is the expected speedup for $K=4$?

<details>
<summary>Answer</summary>

**(a)** Acceptance probs: $\min(1, 0.4/0.3)=1.0$, $\min(1, 0.3/0.4)=0.75$, $\min(1, 0.2/0.1)=1.0$, $\min(1, 0.1/0.2)=0.5$.

**(b)** $E[\text{accepted}] = 1.0 + 0.75 + 1.0 = 2.75$ (the last token is always accepted if we reach it; otherwise expected accepted before rejection). More precisely: the probability of accepting all 4 is $1.0 \cdot 0.75 \cdot 1.0 \cdot 0.5 = 0.375$. Expected accepted = $\sum_{k=0}^4 P(\text{accepted} \ge k) = 1 + 0.75 + 0.75\cdot 1.0 + 0.75\cdot 1.0 \cdot 0.5 + 0.375 = 1 + 0.75 + 0.75 + 0.375 + 0.375 = 3.25$ tokens.

**(c)** Without speculation: 4 tokens × 100ms = 400ms. With speculation: 1 target pass (100ms) + 1 draft pass (10ms) per iteration. Expected iterations = $4 / 3.25 \approx 1.23$. Average time per token = $(100 + 10) / 3.25 \approx 33.8\text{ms}$. Speedup = $100 / 33.8 \approx 2.96\times$.
</details>
