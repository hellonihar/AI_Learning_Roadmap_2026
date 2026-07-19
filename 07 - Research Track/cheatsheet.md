# Research Track — Cheatsheet

## 07.1 — RL Fundamentals

### MDP Components
| Symbol | Meaning |
|--------|---------|
| $S$ | State space |
| $A$ | Action space |
| $P(s'\|s,a)$ | Transition dynamics |
| $R(s,a)$ | Reward function |
| $\gamma$ | Discount factor |

### Bellman Equations
$$V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s') \mid s, \pi]$$
$$Q^\pi(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^\pi(s',a') \mid s,a]$$

### Multi-Armed Bandits
- **$\varepsilon$-Greedy**: explore with prob $\varepsilon$, exploit otherwise
- **UCB**: $a_t = \arg\max (\hat{\mu}_i + \sqrt{2\ln t / n_i})$
- **Thompson Sampling**: sample from posterior $p(\mu_i \mid \text{data})$, pick max

### Dynamic Programming
- **Policy Evaluation**: $V_{k+1} = R^\pi + \gamma P^\pi V_k$
- **Policy Iteration**: evaluate $\to$ improve $\to$ repeat
- **Value Iteration**: $V_{k+1}(s) = \max_a \sum_{s'} P[\cdot](R + \gamma V_k(s'))$

### TD Learning
- **TD(0)**: $V(s) \leftarrow V(s) + \alpha(r + \gamma V(s') - V(s))$
- **SARSA** (on-policy): $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma Q(s',a') - Q(s,a))$
- **Q-Learning** (off-policy): $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$
- **TD($\lambda$)**: $V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$ with eligibility trace $e_t$

### Policy Gradients
- **REINFORCE**: $\nabla J(\theta) = \mathbb{E}[G_t \nabla \log \pi_\theta(a_t\|s_t)]$
- **Actor-Critic**: $\nabla J(\theta) = \mathbb{E}[A(s,a) \nabla \log \pi_\theta(a\|s)]$
- **PPO Clipped**: $\mathcal{L} = \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\varepsilon) \hat{A}_t)]$
- **DDPG**: $\nabla J(\theta) = \mathbb{E}[\nabla_a Q(s,a) \nabla_\theta \mu_\theta(s)]$

---

## 07.2 — LLM Foundations

### GPT Evolution
| Model | Params | Key Innovation |
|-------|--------|---------------|
| GPT-1 | 117M | Generative pre-training |
| GPT-2 | 1.5B | Zero-shot transfer at scale |
| GPT-3 | 175B | In-context learning |
| GPT-3.5 | 175B | RLHF + instruction tuning |
| GPT-4 | ~1.8T MoE | Multimodal, long context |

### Scaling Laws
- **Kaplan**: $N \propto C^{3.3}, D \propto C^{2.8}$
- **Chinchilla**: $N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$
- **Power law**: $L(N) \propto N^{-\alpha_N}, L(D) \propto D^{-\alpha_D}$

### Alignment
- **RLHF**: SFT $\to$ Reward Model $\to$ PPO with KL penalty
- **DPO**: $\mathcal{L} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$
- **Constitutional AI**: self-critique + revision from principles
- **KTO**: binary quality signal, no pairwise preferences

---

## 07.3 — LLM Engineering

### Efficient Attention
- **Vanilla**: $O(L^2)$ time and memory
- **FlashAttention**: IO-aware tiled algorithm, $O(L)$ memory
- **Sliding Window**: $O(LW)$ with window $W$
- **GQA**: $g$ KV-heads shared across query heads

### KV Cache
- Caches $(K, V)$ for all previous tokens during decode
- Memory: $2 \times L \times \text{layers} \times \text{heads} \times d_{\text{head}} \times \text{dtype}$
- **PagedAttention**: page-based allocation, eliminates fragmentation

### Quantization
| Method | Bits | Technique |
|--------|------|-----------|
| GPTQ | 4 | OBQ approximation, column-wise |
| AWQ | 4 | Activation-aware scaling |
| bitsandbytes | 4/8 | Block-wise NF4, double quant |
| GGUF | 2-8 | CPU-optimised (llama.cpp) |

### Speculative Decoding
- Draft model proposes $K$ tokens, target verifies in parallel
- Rejection sampling ensures exact output distribution
- Medusa: draft heads instead of separate draft model
- Speedup: 1.5-3x depending on acceptance rate

---

## 07.4 — Deployment

### Serving Frameworks
| Framework | Key Feature |
|-----------|-------------|
| **vLLM** | PagedAttention, continuous batching |
| **TGI** | Hugging Face integration, streaming |
| **Triton+TRT-LLM** | Multi-framework, TensorRT kernels |

### Inference Optimizations
- **Kernel Fusion**: combine small ops into one GPU kernel
- **Tensor Parallelism**: shard layers across GPUs (all-reduce)
- **Pipeline Parallelism**: split layers across GPUs (P2P)
- **Continuous Batching**: dynamic add/remove sequences

### Cost Optimization
- **Spot instances**: 60-90% discount vs on-demand
- **Model routing**: small model for easy queries
- **Caching**: exact match, semantic, prefix (KV cache)
- **GPU selection**: match GPU to model size/quantisation
