# 07.3.4 — Speculative Decoding

## Overview

Autoregressive decoding generates one token at a time, and each step requires a full forward pass. Speculative decoding accelerates this by using a fast **draft model** to propose multiple tokens, which are then verified by a large **target model** in parallel. The result is mathematically identical to the target model's output, but generated much faster.

## The Core Idea

Instead of generating tokens one-by-one with the large model:

1. The **draft model** (small, e.g., 100M params) generates $K$ tokens quickly.
2. The **target model** (large, e.g., 70B params) processes all $K$ tokens in a single forward pass.
3. **Rejection sampling** determines which draft tokens to accept.

If the target model agrees with the draft model's token distribution, all $K$ tokens are accepted in one step — effectively a K× speedup.

## Rejection Sampling

Given draft distribution $q(x)$ and target distribution $p(x)$:

- Sample $x \sim q(x)$
- Accept $x$ with probability $\min\left(1, \frac{p(x)}{q(x)}\right)$
- If rejected, resample from the residual distribution: $p'(x) \propto \max(0, p(x) - q(x))$

This rejection sampling scheme guarantees the output distribution matches $p(x)$ exactly. The expected accepted tokens per step is:

$$
\mathbb{E}[\text{accepted}] = \sum_x \min(p(x), q(x)) = 1 - D_{\text{TV}}(p \| q)
$$

where $D_{\text{TV}}$ is the total variation distance. The better the draft model matches the target model, the more tokens are accepted per step.

## Medusa

**Paper**: *Medusa: Simple LLM Inference Acceleration Framework without Quality Degradation*

Medusa eliminates the need for a separate draft model by adding multiple **draft heads** (lightweight prediction heads) on top of the target model. Each head predicts the next $i$ tokens:

- Head 1 predicts token $t+1$
- Head 2 predicts token $t+2$ given head 1's prediction
- ...
- Head $K$ predicts token $t+K$

The draft heads are trained with minimal additional compute (fine-tuning the heads only). During inference, the model generates $K$ draft tokens per step, which are verified via tree attention.

**Typical speedup**: 2-3x on standard LLM workloads.

## Parallel Decoding

The verification step processes all $K$ draft tokens in a single forward pass using a special **tree attention** mask. This mask ensures each token only attends to its prefix in the draft tree, not to tokens from alternative branches:

$$
\text{Attention}_{ij} = 0 \quad \text{if } \text{token}_j \text{ is not in the draft prefix of } \text{token}_i
$$

This parallel verification is the source of the speedup: one forward pass of the large model validates $K$ tokens instead of one.

## Practical Considerations

- **Draft model quality**: the speedup depends on the acceptance rate, which depends on how well the draft model matches the target model.
- **Batch size**: speculative decoding is most beneficial at small batch sizes (low utilisation of GPU compute). At large batch sizes, the target model is already saturated and the overhead of running the draft model may not be worth it.
- **Hardware**: draft models may run on CPU while the target model runs on GPU, enabling heterogeneous execution.

## Practical Context

Speculative decoding is a standard optimisation in production LLM systems. Together with KV cache optimisation and quantisation, it forms the trifecta of inference acceleration. vLLM, TensorRT-LLM, and llama.cpp all support speculative decoding.
