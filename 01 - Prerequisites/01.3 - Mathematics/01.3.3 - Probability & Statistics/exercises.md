# Probability & Statistics — Exercises

## Conceptual Questions

1. **Bayes' Theorem in Medical Testing**: A disease affects 1% of the population. A test is 95% accurate for both infected and healthy people. If a person tests positive, what is the probability they actually have the disease? (Compute using Bayes' theorem.)

2. **MLE Intuition**: Why does MLE for a Gaussian distribution give the sample mean as $\hat{\mu}_{\text{MLE}}$? Derive $\hat{\mu}_{\text{MLE}}$ by maximizing $\log P(\mathcal{D} \mid \mu, \sigma^2)$.

3. **Central Limit Theorem**: You have 1000 samples from an exponential distribution with $\lambda = 0.5$. You compute the mean of each sample (size 50). What distribution does the collection of sample means approximately follow? What are its parameters?

4. **Variance vs. Bias**: Explain the bias-variance tradeoff in terms of MLE and MAP. Which estimator might have lower variance but higher bias? Why?

## Coding Exercises

5. **MLE for Poisson Distribution**: Generate $n=200$ samples from $\text{Pois}(\lambda=3)$. Write code to compute the MLE for $\lambda$ (both analytically and numerically via optimization). Verify the answer.

6. **Bayesian Updating (Beta-Bernoulli)**: Implement sequential Bayesian inference for a Bernoulli parameter:
   - Prior: $\text{Beta}(2, 2)$
   - For each new data point, update the posterior: $\text{Beta}(\alpha + \text{successes}, \beta + \text{failures})$
   - Track how the posterior mean converges to the true $p$ as more data arrives
   - Plot posterior distributions after 1, 5, 20, 100 observations (use `np.linspace` to evaluate the Beta PDF)
   - True $p = 0.7$

7. **A/B Test Analyzer**: Write a function `ab_test(conversions_A, n_A, conversions_B, n_B)` that:
   - Computes conversion rates and their difference
   - Performs a z-test and returns the p-value
   - Computes a 95% confidence interval for the difference
   Also write a bootstrap version that resamples both groups and computes the CI for the difference.
   Test with: A (120 conversions / 1500 visitors), B (180 / 1500).

---

## Answers

**Answer 1**: Let $D$ = has disease, $T+$ = positive test.

$$
P(D \mid T+) = \frac{P(T+ \mid D) P(D)}{P(T+ \mid D) P(D) + P(T+ \mid \neg D) P(\neg D)}
= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} \approx 0.161
$$

Only ~16%! This illustrates why screening rare conditions requires extremely accurate tests.

**Answer 2**: For $X_i \sim \mathcal{N}(\mu, \sigma^2)$:

$$
\log L = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2
$$

Taking $\partial/\partial \mu$: $\frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \implies \hat{\mu} = \frac{1}{n}\sum_i x_i$.

**Answer 3**: By CLT, the sample means follow approximately $\mathcal{N}(\mu, \sigma^2/n)$ where $\mu = 1/\lambda = 2$, $\sigma^2 = 1/\lambda^2 = 4$, so $\mathcal{N}(2, 4/50) = \mathcal{N}(2, 0.08)$.

**Answer 4**: MLE has low bias but high variance (overfits). MAP with a prior introduces bias toward the prior but reduces variance. This is the bias-variance tradeoff: regularization (L2) corresponds to a Gaussian prior on weights, reducing variance at the cost of increased bias.

**Answer 5**:
```python
import numpy as np

np.random.seed(42)
true_lambda = 3.0
n = 200
samples = np.random.poisson(true_lambda, size=n)
lambda_mle = samples.mean()
print(f"MLE for lambda: {lambda_mle:.4f} (true={true_lambda})")
```

**Answer 6**:
```python
import numpy as np

np.random.seed(42)
true_p = 0.7
alpha, beta = 2, 2
data = np.random.binomial(1, true_p, size=100)

# Beta PDF function
def beta_pdf(x, a, b):
    # unnormalized beta PDF (proportional)
    return x**(a-1) * (1-x)**(b-1)

for n_obs in [1, 5, 20, 100]:
    successes = data[:n_obs].sum()
    failures = n_obs - successes
    a_post, b_post = alpha + successes, beta + failures
    posterior_mean = a_post / (a_post + b_post)
    print(f"n={n_obs:3d}: posterior Beta({a_post},{b_post}), mean={posterior_mean:.4f}")
```

**Answer 7**:
```python
import numpy as np

def ab_test_z(cA, nA, cB, nB):
    pA, pB = cA/nA, cB/nB
    diff = pB - pA
    p_pool = (cA + cB) / (nA + nB)
    se = np.sqrt(p_pool * (1-p_pool) * (1/nA + 1/nB))
    z = diff / se
    p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z)/np.sqrt(2))))
    ci = diff + np.array([-1.96, 1.96]) * se
    return diff, p_value, ci

def ab_test_bootstrap(cA, nA, cB, nB, n_iter=10000):
    dataA = np.array([1]*cA + [0]*(nA-cA))
    dataB = np.array([1]*cB + [0]*(nB-cB))
    diffs = []
    for _ in range(n_iter):
        bootA = np.random.choice(dataA, nA, replace=True).mean()
        bootB = np.random.choice(dataB, nB, replace=True).mean()
        diffs.append(bootB - bootA)
    ci = np.percentile(diffs, [2.5, 97.5])
    return np.mean(diffs), ci

diff, pval, ci_z = ab_test_z(120, 1500, 180, 1500)
diff_boot, ci_boot = ab_test_bootstrap(120, 1500, 180, 1500)
print(f"Diff: {diff:.4f}, p-value: {pval:.4f}")
print(f"Z-test CI: {ci_z}")
print(f"Bootstrap CI: {ci_boot}")
```
