# Information Theory — Exercises

## Conceptual Questions

1. **Entropy Intuition**: A random variable $X$ takes values $\{a, b, c, d\}$ with probabilities $[0.5, 0.25, 0.125, 0.125]$. Compute its entropy. Then design an optimal binary code for these symbols. How does the average code length relate to the entropy?

2. **Cross-Entropy Minimization**: In classification, the true labels follow $P(y \mid x)$ and we predict $Q(y \mid x)$. Explain why minimizing $H(P, Q)$ is equivalent to minimizing $D_{\text{KL}}(P \parallel Q)$. What term is constant and can be ignored?

3. **KL Divergence Asymmetry**: Give a concrete example where $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$. Explain what each divergence "cares about" in terms of the distributions.

4. **Mutual Information and Correlation**: Two random variables $X$ and $Y$ have correlation $\rho = 0$. Does this imply $I(X; Y) = 0$? Explain with an example.

## Coding Exercises

5. **Entropy of a Dataset**: Write a function `dataset_entropy(y)` that computes the entropy of class labels. Test on balanced (50/50) and imbalanced (90/10) binary datasets. Then write `information_gain(X, y, feature_idx)` that splits a numerical feature at the median and computes information gain.

6. **KL Divergence for Model Comparison**: Train two simple classifiers on synthetic 2D data:
   - A Gaussian Naive Bayes model
   - A logistic regression model
   For each model, compute the KL divergence between the predicted class probabilities and the true class distribution (approximated via histogram binning). Which model's predictions are closer to the true distribution?

7. **VAE Loss Components**: Implement a function that computes the ELBO loss components for a simple VAE with 1D latent space:
   - Prior: $p(z) = \mathcal{N}(0, 1)$
   - Posterior: $q(z \mid x) = \mathcal{N}(\mu(x), \sigma^2(x))$
   - Compute the KL term $D_{\text{KL}}(q(z \mid x) \parallel p(z))$ analytically
   - The analytical formula is: $-\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$
   - Generate random $\mu$ and $\log \sigma^2$ values and verify your implementation matches numerical integration

---

## Answers

**Answer 1**: $H(X) = -(0.5\log_2 0.5 + 0.25\log_2 0.25 + 0.125\log_2 0.125 + 0.125\log_2 0.125) = 0.5 + 0.5 + 0.375 + 0.375 = 1.75$ bits. Optimal code: a→0, b→10, c→110, d→111. Average length = $0.5(1) + 0.25(2) + 0.125(3) + 0.125(3) = 1.75$ bits = entropy. This is Shannon's source coding theorem in action.

**Answer 2**: $H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$. The entropy $H(P)$ is constant (determined by true labels), so minimizing $H(P, Q)$ is equivalent to minimizing $D_{\text{KL}}(P \parallel Q)$.

**Answer 3**: Let $P = [1, 0]$, $Q = [0.5, 0.5]$. $D_{\text{KL}}(P \parallel Q) = 1 \cdot \log(1/0.5) + 0 = 1$ bit. $D_{\text{KL}}(Q \parallel P) = 0.5 \cdot \log(0.5/1) + 0.5 \cdot \log(0.5/0) = \infty$ (since $Q$ assigns probability where $P$ is zero). Forward KL is infinite whenever $Q$ has probability where $P$ doesn't; reverse KL forces $Q$ to avoid areas where $P$ is near zero.

**Answer 4**: No. Mutual information detects non-linear relationships. Example: $X \sim \mathcal{U}(-1, 1)$, $Y = X^2$. $\text{Cov}(X, Y) = \mathbb{E}[X^3] = 0$ (zero correlation), but $I(X; Y) > 0$ since $Y$ is determined by $X$.

**Answer 5**:
```python
import numpy as np

def dataset_entropy(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

balanced = np.array([0]*50 + [1]*50)
imbalanced = np.array([0]*90 + [1]*10)
print(f"Balanced entropy: {dataset_entropy(balanced):.4f}")
print(f"Imbalanced entropy: {dataset_entropy(imbalanced):.4f}")

def information_gain(X, y, feat_idx):
    # H(y) before split
    H_total = dataset_entropy(y)
    # Split at median
    threshold = np.median(X[:, feat_idx])
    left = y[X[:, feat_idx] <= threshold]
    right = y[X[:, feat_idx] > threshold]
    H_cond = (len(left)/len(y))*dataset_entropy(left) + (len(right)/len(y))*dataset_entropy(right)
    return H_total - H_cond
```

**Answer 6**:
```python
# Train both classifiers, get predicted probabilities
# For each model, compute KL(true_dist || pred_dist) via histogram binning
# The model with lower KL divergence better matches the true distribution
# Logistic regression often has lower KL on well-separated data
# Naive Bayes may have higher KL due to its independence assumption
```

**Answer 7**:
```python
import numpy as np

def kl_vae_analytical(mu, log_var):
    """D_KL(q(z|x) || p(z)) for Gaussian q and N(0,1) prior"""
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

# Numerical verification
def kl_vae_numerical(mu, log_var, n_samples=50000):
    sigma = np.exp(0.5 * log_var)
    z = np.random.normal(mu, sigma, size=(n_samples, len(mu)))
    log_q = -0.5 * np.log(2*np.pi) - np.log(sigma) - 0.5 * ((z - mu) / sigma)**2
    log_p = -0.5 * np.log(2*np.pi) - 0.5 * z**2
    return np.mean(np.sum(log_q - log_p, axis=1))

np.random.seed(42)
mu = np.random.randn(3)
log_var = np.random.randn(3)
kl_a = kl_vae_analytical(mu, log_var)
kl_n = kl_vae_numerical(mu, log_var, 100000)
print(f"Analytical KL: {kl_a:.4f}")
print(f"Numerical KL: {kl_n:.4f}")
print(f"Match: {np.isclose(kl_a, kl_n, atol=0.05)}")
```
