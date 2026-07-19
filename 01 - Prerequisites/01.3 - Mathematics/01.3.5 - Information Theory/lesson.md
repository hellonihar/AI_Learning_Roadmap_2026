# Information Theory for Machine Learning

Information theory provides the mathematical framework for quantifying information, uncertainty, and the relationship between random variables. Its concepts are deeply embedded in modern machine learning — from loss functions to model evaluation.

---

## 1. Entropy

**Entropy** measures the average amount of information (or uncertainty) in a random variable. For a discrete random variable $X$ with PMF $P(X)$:

$$
H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

With base-2 logarithms, entropy is measured in **bits**; with natural logs, in **nats**.

### Properties
- $H(X) \geq 0$ with equality iff $X$ is deterministic
- Maximum entropy for a given support size $n$ is achieved by the uniform distribution: $H_{\max} = \log n$
- Example: A fair coin ($p = 0.5$) has $H = 1$ bit. A biased coin ($p = 0.9$) has $H \approx 0.47$ bits (less uncertainty).

### Differential Entropy (Continuous)

For continuous random variables:

$$
h(X) = -\int p(x) \log p(x) \, dx
$$

Unlike discrete entropy, differential entropy can be negative.

---

## 2. Joint and Conditional Entropy

**Joint entropy** measures uncertainty of two variables together:

$$
H(X, Y) = -\sum_{x, y} P(x, y) \log P(x, y)
$$

**Conditional entropy** measures uncertainty remaining in $Y$ after knowing $X$:

$$
H(Y \mid X) = -\sum_{x, y} P(x, y) \log P(y \mid x) = H(X, Y) - H(X)
$$

Key identity: $H(X, Y) = H(X) + H(Y \mid X)$ (chain rule for entropy).

---

## 3. Cross-Entropy

**Cross-entropy** measures the average number of bits needed to encode samples from distribution $P$ using a code optimized for distribution $Q$:

$$
H(P, Q) = -\sum_{x} P(x) \log Q(x)
$$

### Relationship to KL Divergence

$$
H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)
$$

Cross-entropy = entropy of $P$ + KL divergence from $P$ to $Q$.

### Application: Loss Function

Cross-entropy is the standard loss function for classification:

$$
L = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})
$$

Minimizing cross-entropy is equivalent to minimizing KL divergence between the true distribution and the predicted distribution.

---

## 4. KL Divergence (Kullback-Leibler Divergence)

KL divergence measures how one probability distribution diverges from another:

$$
D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

### Properties
- $D_{\text{KL}}(P \parallel Q) \geq 0$ with equality iff $P = Q$ (Gibbs' inequality)
- **Not symmetric**: $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$
- **Not a metric** (does not satisfy triangle inequality)

### Forward vs. Reverse KL

- **Forward KL** $D_{\text{KL}}(P \parallel Q)$: Averaging over $P$, encourages $Q$ to cover all modes of $P$
- **Reverse KL** $D_{\text{KL}}(Q \parallel P)$: Averaging over $Q$, encourages $Q$ to fit a single mode of $P$

This distinction matters in variational inference: minimizing reverse KL gives a mode-seeking approximation.

---

## 5. Mutual Information

**Mutual information** measures how much knowing one variable reduces uncertainty about another:

$$
I(X; Y) = D_{\text{KL}}(P_{XY} \parallel P_X P_Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}
$$

### Relationship to Entropy

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)
$$

- $I(X; Y) \geq 0$ with equality iff $X$ and $Y$ are independent
- $I(X; Y) \leq \min(H(X), H(Y))$

### Application: Feature Selection

Mutual information can rank features by their relevance to the target variable. A feature $X_i$ with high $I(X_i; Y)$ is more informative for predicting $Y$.

---

## 6. Jensen-Shannon Divergence

The **Jensen-Shannon divergence** (JSD) is a symmetric, bounded version of KL divergence:

$$
D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}\left(P \parallel \frac{P+Q}{2}\right) + \frac{1}{2} D_{\text{KL}}\left(Q \parallel \frac{P+Q}{2}\right)
$$

### Properties
- $0 \leq D_{\text{JS}} \leq \log 2$ (for base-2 log, max is 1 bit)
- Symmetric: $D_{\text{JS}}(P \parallel Q) = D_{\text{JS}}(Q \parallel P)$
- $\sqrt{D_{\text{JS}}}$ is a metric
- Used in GANs: the original GAN loss relates to the JSD between real and generated distributions

---

## 7. Applications in AI/ML

| Concept | Application |
|---------|------------|
| Entropy | Decision tree splitting (information gain), uncertainty quantification |
| Cross-entropy | Classification loss function, language model training |
| KL divergence | Variational autoencoders (VAEs), policy gradients in RL |
| Mutual information | Feature selection, representation learning (InfoNCE) |
| Jensen-Shannon divergence | Generative Adversarial Networks (GANs) |
| Conditional entropy | Model evaluation, Bayesian experimental design |

### Information Gain in Decision Trees

The **information gain** of splitting on feature $A$ is:

$$
\text{IG}(Y, A) = H(Y) - H(Y \mid A)
$$

This is exactly the mutual information between the target and the feature. Decision trees select the split that maximizes information gain.

### Cross-Entropy in Language Models

LLMs trained on next-token prediction minimize the cross-entropy loss:

$$
L = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})
$$

This is also called **perplexity** when exponentiated: $\text{PPL} = \exp(L)$.

### Variational Autoencoders (VAEs)

VAEs optimize the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L} = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - D_{\text{KL}}(q(z \mid x) \parallel p(z))
$$

- First term: reconstruction loss
- Second term: KL divergence between approximate posterior and prior

### Model Compression

Knowledge distillation trains a smaller "student" model to match a larger "teacher" model by minimizing KL divergence between their output distributions:

$$
L_{\text{KD}} = D_{\text{KL}}(P_{\text{teacher}} \parallel P_{\text{student}})
$$

---

## Summary

Information theory gives us principled ways to measure and manipulate information. Cross-entropy and KL divergence are the workhorses of modern deep learning loss functions, while mutual information provides a foundation for understanding what models learn.
