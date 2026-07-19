# Diffusion Models

## Overview

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. Inspired by non-equilibrium thermodynamics, they have rapidly become the state of the art in image generation, surpassing GANs on many benchmarks. They are the foundation behind DALL-E 2, Stable Diffusion, Imagen, and Midjourney.

## Forward Diffusion Process

The forward process is a fixed Markov chain that gradually adds Gaussian noise to the data over $T$ timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)$$

where $\beta_t$ is a variance schedule that controls the amount of noise added at each step. Starting from a clean data point $x_0$, after $T$ steps the distribution $q(x_T)$ approaches an isotropic Gaussian $\mathcal{N}(0, I)$.

A useful property is that we can sample $x_t$ directly at any timestep $t$ without iterating:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

where $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, and $\epsilon \sim \mathcal{N}(0, I)$.

## Reverse Diffusion Process

The generative model learns to reverse the forward process. Starting from pure noise $x_T \sim \mathcal{N}(0, I)$, it iteratively denoises:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Training involves predicting the noise $\epsilon$ added at each step (or predicting $x_0$). The simplified training objective (Ho et al., 2020) is:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

## Connection to Score Matching and VAE

Diffusion models are closely related to **score-based generative models** (Song & Ermon, 2019). The score function is the gradient of the log probability density: $\nabla_x \log p(x)$. The noise prediction objective is equivalent to learning the score function scaled by a factor:

$$\epsilon_\theta(x_t, t) \propto -\sqrt{1 - \bar{\alpha}_t}\; \nabla_x \log p(x_t)$$

The connection to VAEs becomes clear when viewing the diffusion process as a hierarchical VAE with $T$ latent variables, where the encoder $q(x_1, \ldots, x_T | x_0)$ is fixed (not learned) and the decoder $p_\theta(x_0, \ldots, x_T)$ is learned to reverse the process. The ELBO for this hierarchical VAE reduces to the denoising objective.

## DDPM vs. DDIM

**DDPM (Denoising Diffusion Probabilistic Models):** The standard formulation (Ho et al., 2020). Generative process is a Markov chain. Requires many steps (~1000) for high-quality generation.

**DDIM (Denoising Diffusion Implicit Models):** (Song et al., 2020). A non-Markovian variant that allows:
- **Fewer sampling steps** (10–50 vs. 1000) by defining a different reverse process that skips timesteps.
- **Deterministic generation** (the mapping from noise to data is deterministic given the noise), enabling interpolation in latent space.
- **Trade-off:** Faster sampling at the cost of slightly lower quality per step.

## Why Diffusion Models Outperform GANs for Image Generation

1. **Stable training:** Diffusion models do not involve adversarial training. The loss function is a simple MSE on noise prediction, which converges reliably without mode collapse or vanishing gradient issues.

2. **Coverage:** Diffusion models produce highly diverse outputs and do not suffer from mode collapse. They effectively model the full data distribution.

3. **Scalability:** Diffusion models scale well with model capacity and dataset size. Larger models and more compute consistently improve results.

4. **Mathematical foundation:** The connection to score matching and the ELBO provides a principled optimization framework with clear convergence properties.

5. **Comparable or better FID:** Modern diffusion models achieve state-of-the-art FID scores on class-conditional and unconditional generation benchmarks.

6. **Controllable generation:** Classifier-free guidance allows trading off diversity for fidelity by amplifying the conditional signal during sampling.

## Limitations

- **Slow sampling:** Even with DDIM, diffusion models require multiple forward passes (10–50 steps) compared to GANs (single forward pass).
- **High compute cost:** Training requires many steps and large batch sizes.
- **Less latent space interpretability:** Compared to GANs, the latent space of diffusion models is less well understood for semantic editing.

## Summary

Diffusion models have revolutionized generative AI by offering a stable, principled alternative to GANs. The forward-reverse process framework, combined with connections to score matching, provides a robust training paradigm. While slower at inference than GANs, their superior sample quality, diversity, and training stability have made them the dominant approach for image generation as of 2024–2026.
