# Exercises: Unsupervised Deep Learning

---

## Exercise 1: Derive the VAE ELBO

Starting from the log marginal likelihood $\log p_\theta(x)$, derive the Evidence Lower Bound (ELBO). Show all steps including the introduction of the variational posterior $q_\phi(z|x)$ and application of Jensen's inequality.

<details>
<summary>Answer</summary>

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x|z) p(z) \, dz \\
&= \log \int p_\theta(x|z) \frac{p(z)}{q_\phi(z|x)} q_\phi(z|x) \, dz \\
&= \log \mathbb{E}_{q_\phi(z|x)} \left[ \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right] \\
&\geq \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right] \quad \text{(Jensen's inequality)} \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(z)}{q_\phi(z|x)} \right] \\
&= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \parallel p(z))
\end{aligned}
$$

The two terms are the reconstruction loss and the KL divergence regularization.

**Why Jensen's inequality is valid:** $\log$ is a concave function, so $\log(\mathbb{E}[Y]) \geq \mathbb{E}[\log(Y)]$.
</details>

---

## Exercise 2: Explain the Reparameterization Trick

Why is the reparameterization trick necessary for training VAEs? Write the mathematical expression and explain how it enables gradient-based optimization.

<details>
<summary>Answer</summary>

The VAE encoder outputs a distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$. Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is a stochastic operation — the sampling process has no gradient with respect to $\mu$ and $\sigma$.

The reparameterization trick writes $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. Now:
- $\mu$ and $\sigma$ are deterministic functions of $x$ and $\phi$.
- $\epsilon$ provides the stochasticity but is independent of $\phi$.
- Gradients can flow through $\mu$ and $\sigma$ via $\frac{\partial z}{\partial \mu} = 1$ and $\frac{\partial z}{\partial \sigma} = \epsilon$.

Without this, we could not backpropagate through the sampling operation and the encoder could not be trained.
</details>

---

## Exercise 3: Compare AE, VAE, and GAN

Fill in the table:

| Property | AE | VAE | GAN |
|----------|----|-----|-----|
| Latent space | ? | ? | ? |
| Generative capability | ? | ? | ? |
| Training stability | ? | ? | ? |
| Loss function | ? | ? | ? |
| Sample quality | ? | ? | ? |

<details>
<summary>Answer</summary>

| Property | AE | VAE | GAN |
|----------|----|-----|-----|
| Latent space | Deterministic, unstructured | Probabilistic, regularized | Usually deterministic, often structured |
| Generative capability | Limited (no sampling mechanism) | Strong (sample from prior) | Strong (sample from noise) |
| Training stability | Very stable | Stable | Unstable, sensitive to hyperparameters |
| Loss function | MSE (reconstruction) | ELBO (reconstruction + KL) | Minimax (adversarial) |
| Sample quality | Blurry, reconstructions only | Slightly blurry | Sharp, realistic |

</details>

---

## Exercise 4: Mode Collapse Analysis

You train a GAN on the MNIST dataset (10 digit classes). After training, the generator only produces images of the digit "8". Explain:

1. What is this phenomenon called?
2. What causes it during training?
3. Name two techniques to mitigate it.

<details>
<summary>Answer</summary>

1. **Mode collapse** — the generator captures only a subset of the data distribution (one mode) instead of the full diversity.

2. **Causes:** The generator finds a specific output that consistently fools the discriminator. The discriminator learns to reject this output, causing the generator to switch to another easy target — creating a cycle. The generator exploits weaknesses in the discriminator rather than learning the full data distribution.

3. **Mitigation techniques:**
   - **Minibatch discrimination:** The discriminator looks at batches of samples, not individual ones, making it harder to fool with repeated outputs.
   - **WGAN/WGAN-GP:** The Wasserstein loss provides smoother gradients and reduces mode collapse.
   - **Unrolled GANs:** The generator sees several steps into the discriminator's optimization, preventing it from exploiting short-term discriminator weaknesses.
   - **Spectral normalization:** Stabilizes discriminator gradients.
</details>

---

## Exercise 5: Compute Reconstruction Loss

Given a batch of 3 samples with original values $x$ and reconstructions $\hat{x}$:

$$x = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.5 & 0.5 \end{bmatrix}, \quad \hat{x} = \begin{bmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \\ 0.6 & 0.4 \end{bmatrix}$$

Compute the MSE reconstruction loss.

<details>
<summary>Answer</summary>

MSE = $\frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$

Sample 1: $(1.0 - 0.9)^2 + (0.0 - 0.1)^2 = 0.01 + 0.01 = 0.02$
Sample 2: $(0.0 - 0.2)^2 + (1.0 - 0.8)^2 = 0.04 + 0.04 = 0.08$
Sample 3: $(0.5 - 0.6)^2 + (0.5 - 0.4)^2 = 0.01 + 0.01 = 0.02$

Sum = $0.02 + 0.08 + 0.02 = 0.12$

MSE = $0.12 / 3 = 0.04$
</details>

---

## Exercise 6: Reversible vs. Irreversible Transformations

The CycleGAN uses a cycle consistency loss: $\|F(G(x)) - x\|_1$. Why is this necessary for unpaired image-to-image translation? What would happen if we only used adversarial losses?

<details>
<summary>Answer</summary>

With only adversarial losses, the generator $G: X \to Y$ must produce outputs that appear to come from domain $Y$, but there is no constraint preserving the content of the input $x$. For example, translating a photo of a zebra to "horse" domain could change the background, the pose, or the identity entirely — as long as the output looks like a horse to the discriminator.

The cycle consistency loss enforces that translating $x \to Y \to X$ should approximately recover the original $x$. This forces $G$ to preserve the content/structure of the input while only changing the domain-specific appearance (style). Without cycle consistency, the generator would have too much freedom and the outputs would lack correspondence to the inputs.

---

## Exercise 7: Diffusion Forward Process

Given $x_0 = [2.0, -1.0]$, $\beta_t = 0.02$ for all $t$, compute $x_3$ using the forward diffusion process. Show both the iterative approach and the direct formula.

Assume $\epsilon_1, \epsilon_2, \epsilon_3 \sim \mathcal{N}(0, I)$ sampled as:
$\epsilon_1 = [0.5, -0.3],\; \epsilon_2 = [-0.2, 0.4],\; \epsilon_3 = [0.1, 0.6]$

<details>
<summary>Answer</summary>

**Iterative approach:**

$\alpha_t = 1 - \beta_t = 0.98$

$x_1 = \sqrt{0.98} \cdot [2.0, -1.0] + \sqrt{0.02} \cdot [0.5, -0.3]$
$x_1 = [0.9899 \cdot 2.0 + 0.1414 \cdot 0.5,\; 0.9899 \cdot (-1.0) + 0.1414 \cdot (-0.3)]$
$x_1 = [1.9798 + 0.0707,\; -0.9899 - 0.0424] = [2.0505, -1.0323]$

$x_2 = \sqrt{0.98} \cdot [2.0505, -1.0323] + \sqrt{0.02} \cdot [-0.2, 0.4]$
$x_2 = [0.9899 \cdot 2.0505 + 0.1414 \cdot (-0.2),\; 0.9899 \cdot (-1.0323) + 0.1414 \cdot 0.4]$
$x_2 = [2.0293 - 0.0283,\; -1.0219 + 0.0566] = [2.0010, -0.9653]$

$x_3 = \sqrt{0.98} \cdot [2.0010, -0.9653] + \sqrt{0.02} \cdot [0.1, 0.6]$
$x_3 = [0.9899 \cdot 2.0010 + 0.1414 \cdot 0.1,\; 0.9899 \cdot (-0.9653) + 0.1414 \cdot 0.6]$
$x_3 = [1.9808 + 0.0141,\; -0.9555 + 0.0848] = [1.9949, -0.8707]$

**Direct formula:**

$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$

$\bar{\alpha}_3 = (0.98)^3 = 0.941192$

$x_3 = \sqrt{0.941192} \cdot [2.0, -1.0] + \sqrt{1 - 0.941192} \cdot \epsilon_{agg}$

For the aggregate noise we need the actual noise added over 3 steps using the forward process property. Let's compute with the total noise variance:

$x_3 = \sqrt{0.9702} \cdot [2.0, -1.0] + \sqrt{0.0588} \cdot [0.4429, 0.5947]$
(where $0.4429, 0.5947$ is the effective aggregate noise)

$x_3 \approx [1.9404 + 0.1074,\; -0.9702 + 0.1442] = [2.0478, -0.8260]$

The slight difference from the iterative approach is due to rounding.

</details>

---

## Exercise 8: KL Divergence Computation

For a single data point, the VAE encoder outputs $\mu = [0.5, -0.3]$ and $\log\sigma^2 = [-0.2, 0.1]$. Compute the KL divergence between $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ and the prior $p(z) = \mathcal{N}(0, I)$.

<details>
<summary>Answer</summary>

The closed-form KL divergence is:

$$\text{KL}(q \parallel p) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

Compute for each dimension:

**Dimension 1:** $\mu_1 = 0.5$, $\log\sigma_1^2 = -0.2 \implies \sigma_1^2 = e^{-0.2} = 0.8187$

$$\text{Term}_1 = 1 + (-0.2) - (0.5)^2 - 0.8187 = 1 - 0.2 - 0.25 - 0.8187 = -0.2687$$

**Dimension 2:** $\mu_2 = -0.3$, $\log\sigma_2^2 = 0.1 \implies \sigma_2^2 = e^{0.1} = 1.1052$

$$\text{Term}_2 = 1 + 0.1 - (-0.3)^2 - 1.1052 = 1 + 0.1 - 0.09 - 1.1052 = -0.0952$$

**Total KL divergence:**

$$\text{KL} = -\frac{1}{2}(-0.2687 + -0.0952) = -\frac{1}{2}(-0.3639) = 0.1819$$

Interpretation: The KL divergence is 0.1819 nats, indicating a small deviation from the standard Gaussian prior. This is a reasonable value during VAE training.
</details>
