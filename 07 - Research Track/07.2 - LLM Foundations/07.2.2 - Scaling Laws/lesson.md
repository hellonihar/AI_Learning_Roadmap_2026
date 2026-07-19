# 07.2.2 — Scaling Laws

## Overview

Scaling laws describe how model performance improves as a function of model size, dataset size, and compute budget. They provide a scientific basis for resource allocation in LLM development, enabling practitioners to predict optimal model configurations before training.

## Kaplan Scaling Laws (2020)

**Paper**: *Scaling Laws for Neural Language Models*

Kaplan et al. studied autoregressive Transformers ranging from 768 to 1.5B parameters and discovered **power-law scaling** of cross-entropy loss with each axis:

$$
L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}
$$

where $N$ is the number of parameters, $D$ is the dataset size in tokens, and $C$ is the compute budget in FLOPs.

Key findings:

- **Larger models are more sample-efficient**: increasing $N$ reduces the loss more than increasing $D$ for the same compute.
- **Performance scales smoothly**: there is no sharp threshold for "good" performance — improvement is continuous with scale.
- **Optimal allocation**: given a compute budget $C$, most should go to model size, not data. Specifically, $N \propto C^{3.3}$ and $D \propto C^{2.8}$ for early stopping.

The compute-optimal frontier follows:

$$
L(C) \approx \left( \frac{C}{C_{\min}} \right)^{-\alpha_C}
$$

## Chinchilla Scaling Laws (2022)

**Paper**: *Training Compute-Optimal Large Language Models*

Hoffmann et al. (DeepMind) revisited the question with a more rigorous methodology. Their conclusion was striking: **Kaplan's laws significantly overestimated the optimal model size**. For a given compute budget, you should train a smaller model on more data.

Chinchilla found:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

This 1:1 allocation means for every doubling of compute, both model size and training tokens should double. For example, a 70B model should be trained on ~1.4T tokens — far more than GPT-3's 300B tokens.

The Chinchilla model (70B parameters, 1.4T tokens) outperformed GPT-3 (175B parameters, 300B tokens) across many benchmarks, proving **data quality and quantity matter as much as model size**.

## Practical Implications

- **Kaplan regime** (pre-2022): most LLMs were undertrained (too many parameters, too few tokens).
- **Chinchilla regime** (post-2022): models like Llama, MPT, and Falcon were trained on 1-2T tokens, following the 1:1 compute allocation.
- **Continued scaling**: newer models push beyond Chinchilla-optimal because more data continues to improve performance even if it's suboptimal for the exact compute budget.

## Emergent Abilities and Inverse Scaling

Scaling also revealed **emergent abilities** — tasks where performance jumps sharply at a critical scale (e.g., arithmetic, code generation, chain-of-thought reasoning). Conversely, some tasks show **inverse scaling** where larger models perform worse (e.g., certain bias benchmarks).

## Practical Context

Scaling laws guide decisions like: "Should I train a 7B model on 2T tokens or a 13B model on 1T tokens?" Understanding the compute-optimal frontier helps you get the best performance for your training budget.
