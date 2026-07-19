# Resources: Unsupervised Deep Learning

---

## Foundational Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| **Auto-Encoding Variational Bayes** (Kingma & Welling) | 2013 | VAE, reparameterization trick, ELBO |
| **Generative Adversarial Nets** (Goodfellow et al.) | 2014 | GAN framework, minimax game |
| **Unsupervised Representation Learning with Deep Convolutional GANs** (Radford et al.) | 2015 | DCGAN architecture guidelines |
| **Conditional Generative Adversarial Nets** (Mirza & Osindero) | 2014 | cGAN, conditioning on auxiliary info |
| **Wasserstein GAN** (Arjovsky, Chintala, Bottou) | 2017 | WGAN, Wasserstein distance |
| **Improved Training of Wasserstein GANs** (Gulrajani et al.) | 2017 | WGAN-GP, gradient penalty |
| **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks** (Zhu et al.) | 2017 | CycleGAN |
| **A Style-Based Generator Architecture for GANs** (Karras et al.) | 2019 | StyleGAN, AdaIN, mapping network |
| **Denoising Diffusion Probabilistic Models** (Ho, Jain, Abbeel) | 2020 | DDPM, simplified training objective |
| **Denoising Diffusion Implicit Models** (Song, Meng, Ermon) | 2020 | DDIM, accelerated sampling |
| **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al.) | 2021 | Unified score-based + diffusion framework |

### How to Read These Papers

- Start with Kingma & Welling (VAE) — it's the most accessible.
- Then Goodfellow et al. (GAN) — the original paper is well-written.
- Read Ho et al. (DDPM) for the modern diffusion perspective.
- Skip to Arjovsky et al. (WGAN) for understanding GAN training difficulties.

---

## Courses

| Course | Provider | Relevance |
|--------|----------|-----------|
| **CS231n: CNNs for Visual Recognition** | Stanford | GANs, VAEs, generative modeling for vision |
| **CS236: Deep Generative Models** | Stanford | Comprehensive — VAEs, GANs, normalizing flows, diffusion |
| **Probabilistic Machine Learning** (Spring 2023) | NYU / Philippe Hennet | VAE theory, latent variable models |
| **Deep Unsupervised Learning** | UC Berkeley / Pieter Abbeel | Advanced topics in generative modeling |
| **Fast.ai: Practical Deep Learning** | fast.ai | Hands-on implementations of GANs, autoencoders |

---

## Books

| Book | Author(s) | Focus |
|------|-----------|-------|
| **Deep Learning** | Goodfellow, Bengio, Courville | Chapters 14 (Autoencoders), 20 (Generative Models) |
| **Probabilistic Machine Learning: Advanced Topics** | Murphy | Chapters on VAEs, normalizing flows, diffusion |
| **Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play** | Foster | Practical implementations in TensorFlow/Keras |

---

## Blog Posts & Tutorials

- **"Understanding Variational Autoencoders (VAEs)"** — Joseph Rocca, Towards Data Science
- **"From Autoencoder to Beta-VAE"** — Lilian Weng (lilianweng.github.io)
- **"Generative Adversarial Networks: The Math Behind It"** — Joseph Rocca
- **"The GAN Landscape: Losses, Architectures, and Training"** — Casper Hansen
- **"What are Diffusion Models?"** — Lilian Weng (lilianweng.github.io)
- **"The Annotated Diffusion Model"** — Niels Rogge, Kashif Rasul (Hugging Face blog)
- **"Diffusion Models: A Comprehensive Guide"** — Ryan O'Connor, Towards AI
- **"Understanding the Reparameterization Trick"** — Jake Tae
- **"Intuitively Understanding Variational Autoencoders"** — Irhum Shafkat, Towards Data Science
- **"A Beginner's Guide to Variational Methods"** — Erik Sudderth (video lecture series)

---

## Video Lectures

- **"Autoencoders"** — Hugo Larochelle, Neural Networks course (YouTube)
- **"Variational Autoencoders"** — Arnaud Driesse, ML Explained
- **"Generative Adversarial Networks (GANs)"** — Ian Goodfellow, Stanford CS231n
- **"Diffusion Models"** — CVPR 2022 Tutorial by Karsten Kreis et al.
- **"Score-Based Generative Modeling"** — Yang Song, ICLR 2021 Tutorial

---

## Implementations & Code

| Repository | Description |
|------------|-------------|
| **pytorch/examples/vae** | Official PyTorch VAE implementation |
| **eriklindernoren/PyTorch-GAN** | Collection of GAN implementations in PyTorch |
| **lucidrains/denoising-diffusion-pytorch** | DDPM/DDIM implementation |
| **NVlabs/stylegan3** | Official StyleGAN3 code |
| **huggingface/diffusers** | Production-grade diffusion model library |
| **junyanz/pytorch-CycleGAN-and-pix2pix** | Official CycleGAN implementation |

---

## Key People to Follow

| Researcher | Known For |
|------------|-----------|
| Diederik Kingma | VAE, Adam optimizer |
| Max Welling | VAE, Bayesian deep learning |
| Ian Goodfellow | GAN inventor |
| Alec Radford | DCGAN, GPT |
| Martin Arjovsky | WGAN, theoretical GAN analysis |
| Tero Karras | StyleGAN series |
| Jonathan Ho | DDPM, diffusion models |
| Yang Song | Score-based generative modeling, DDIM |
| Dina Bashkirova | Unsupervised domain adaptation |
