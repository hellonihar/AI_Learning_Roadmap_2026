# Generative Adversarial Networks (GANs)

## Overview

Generative Adversarial Networks, introduced by Ian Goodfellow et al. in 2014, represent a paradigm shift in generative modeling. Instead of a single network learning to model the data distribution directly, GANs pit two neural networks against each other in a game-theoretic framework: a **generator** that produces fake data and a **discriminator** that tries to distinguish real from fake. Through this adversarial process, the generator learns to produce increasingly realistic samples.

## The Two Players

### Generator ($G$)

The generator takes a random noise vector $z \sim p(z)$ (typically sampled from a standard Gaussian or uniform distribution) and maps it to the data space:

$$G(z; \theta_G) : z \in \mathbb{R}^d \to x_{\text{fake}} \in \mathbb{R}^D$$

The goal of the generator is to produce samples that are indistinguishable from real data, thereby "fooling" the discriminator.

### Discriminator ($D$)

The discriminator takes a data point $x$ (real or fake) and outputs a scalar probability that $x$ came from the real data distribution rather than from the generator:

$$D(x; \theta_D) : x \in \mathbb{R}^D \to [0, 1]$$

The goal of the discriminator is to correctly classify real vs. fake samples.

## The Minimax Game

GANs are trained via a two-player minimax game with the value function $V(D, G)$:

$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

The training alternates between:
1. **Updating the discriminator:** Maximize $\log D(x)$ for real samples and $\log(1 - D(G(z)))$ for fake samples.
2. **Updating the generator:** Minimize $\log(1 - D(G(z)))$ — equivalently, maximize $\log D(G(z))$ (the "non-saturating" heuristic, which provides stronger gradients early in training).

### Nash Equilibrium

The optimal solution to this game is a Nash equilibrium where:
- The discriminator cannot distinguish real from fake: $D(x) = \frac{1}{2}$ for all $x$.
- The generator perfectly matches the real data distribution: $p_G = p_{\text{data}}$.

In theory, this equilibrium is reached when the generator recovers the true data distribution. In practice, GAN training rarely reaches a true equilibrium and requires careful tuning.

## Training Dynamics

Training a GAN is notoriously challenging due to the adversarial nature of the optimization:

- **Alternating updates:** One step of discriminator update for every $k$ steps of generator update (typically $k = 1$).
- **Learning rate balance:** If the discriminator becomes too strong, the generator gradients vanish. If the generator becomes too strong, the discriminator collapses.
- **Loss landscapes:** The generator's loss $\log(1 - D(G(z)))$ saturates when $D$ confidently rejects fake samples. The non-saturating variant $\max \log D(G(z))$ provides better gradients.

## DCGAN: Deep Convolutional GAN

The DCGAN (Radford et al., 2015) architecture established key design principles for stable GAN training with convolutional neural networks:

- Replace pooling layers with strided convolutions (discriminator) and transposed convolutions (generator).
- Use batch normalization in both networks (except the generator output and discriminator input).
- Remove fully connected hidden layers.
- Use ReLU in the generator (except the output layer, which uses Tanh).
- Use leaky ReLU in the discriminator (except the output layer, which uses Sigmoid).

DCGAN demonstrated that GANs could generate realistic images of bedrooms, faces, and other objects, and that the learned latent space supports vector arithmetic (e.g., "smiling woman minus neutral woman plus neutral man" yields a smiling man).

## Challenges

### Mode Collapse

Mode collapse occurs when the generator learns to produce only a few distinct outputs (or a single output) that fool the discriminator. The generator exploits a weakness in the discriminator rather than learning the full diversity of the data distribution. The discriminator sees these repeated samples and learns to reject them, causing the generator to switch to a different mode — a process that can cycle indefinitely.

**Mitigations:** Minibatch discrimination, unrolled GANs, Wasserstein loss, and spectral normalization.

### Non-Convergence

The minimax game is a non-convex optimization problem with complex dynamics. The two networks may oscillate without ever reaching equilibrium. The loss curves often do not indicate convergence in the traditional supervised learning sense.

**Mitigations:** Two-timescale update rule (TTUR), where the discriminator and generator use different learning rates.

### Vanishing Gradients

When the discriminator is too good at distinguishing real from fake, the generator receives near-zero gradients and cannot learn. This is especially problematic early in training when the generator produces obviously fake samples.

**Mitigations:** The non-saturating loss ($\max \log D(G(z))$), Wasserstein loss (which provides meaningful gradients even when the discriminator is strong), and feature matching.

### Evaluation

Unlike supervised models, GANs lack a single objective metric that correlates perfectly with visual quality. Common evaluation metrics include:

- **Inception Score (IS):** Measures both the quality and diversity of generated images using a pre-trained Inception network.
- **Fréchet Inception Distance (FID):** Compares the statistics of real and generated image features in the Inception embedding space. Lower FID is better.
- **Precision and Recall:** Decompose FID into precision (how realistic individual samples are) and recall (how well the generator covers the data distribution).

## Summary

GANs introduced a powerful adversarial framework for generative modeling. The generator-discriminator minimax game drives the generator to produce increasingly realistic data. Despite significant challenges — mode collapse, non-convergence, and vanishing gradients — architectural innovations like DCGAN and improved training techniques have made GANs one of the most impactful generative modeling paradigms. Understanding the game-theoretic foundation is essential before exploring GAN variants and alternatives.
