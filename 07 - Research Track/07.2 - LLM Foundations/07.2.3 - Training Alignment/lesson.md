# 07.2.3 — Training & Alignment

## Overview

Alignment ensures LLMs produce outputs that are **helpful, honest, and harmless**. Pre-trained models simply predict the next token — they don't inherently follow instructions, refuse harmful requests, or express uncertainty. Alignment techniques add these capabilities after pre-training.

## Instruction Tuning

The simplest alignment method: fine-tune the pre-trained model on a dataset of (instruction, response) pairs. FLAN (Fine-tuned LAnguage Net) and FLAN-T5 showed that instruction tuning dramatically improves zero-shot performance on unseen tasks.

Datasets like OpenAssistant, ShareGPT, and Dolly provide instruction-response pairs. The loss is standard supervised fine-tuning (SFT):

$$
\mathcal{L}_{\text{SFT}} = -\sum_t \log \pi_\theta(y_t \mid x, y_{<t})
$$

## RLHF — Reinforcement Learning from Human Feedback

RLHF is a three-stage pipeline:

1. **SFT**: instruction-tune the model on high-quality demonstrations.
2. **Reward modelling**: train a reward model $R_\phi(x, y)$ on human preference comparisons. Given two outputs $y_1, y_2$, the Bradley-Terry preference model gives:

$$
P(y_1 \succ y_2) = \frac{\exp(R_\phi(x, y_1))}{\exp(R_\phi(x, y_1)) + \exp(R_\phi(x, y_2))}
$$

3. **RL fine-tuning**: optimise the policy $\pi_\theta$ with PPO to maximise reward while staying close to the SFT model via a KL penalty:

$$
\mathcal{L}_{\text{RLHF}} = \mathbb{E}_{y \sim \pi_\theta} \left[ R_\phi(x, y) - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}}) \right]
$$

The KL penalty prevents the model from exploiting the reward model.

## DPO — Direct Preference Optimization

DPO eliminates the explicit reward model by reparameterising the RLHF objective in terms of the policy itself:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

where $y_w$ is the preferred response and $y_l$ is the dispreferred one. DPO is simpler (no reward model, no RL loop) and often matches or exceeds RLHF in practice.

## Constitutional AI (CAI)

Anthropic's approach uses a **constitution** (a set of principles) to supervise the model via self-critique. A two-stage process:

1. **Red-team**: generate harmful outputs, then have the model critique and revise them according to the constitution.
2. **Preference learning**: train on (harmful, revised) pairs using DPO or RLHF.

CAI reduces reliance on human annotators and improves harmlessness.

## KTO — Kahneman-Tversky Optimisation

KTO derives from prospect theory: humans evaluate outcomes relative to a reference point. Instead of pairwise preferences, KTO uses a binary signal ("good" vs "bad" output):

$$
\mathcal{L}_{\text{KTO}} = -\mathbb{E}_{x,y} \left[ \lambda - \text{sigmoid}\left( \beta \left( \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}} \right) \right) \right]
$$

KTO requires only per-output quality labels, not pairwise preferences, simplifying data collection.

## Practical Context

Alignment is one of the most active research areas. The choice between RLHF, DPO, and KTO depends on data availability, compute budget, and whether you have pairwise comparisons or individual quality ratings. Modern models (GPT-4, Claude, Gemini) use variants of these techniques.
