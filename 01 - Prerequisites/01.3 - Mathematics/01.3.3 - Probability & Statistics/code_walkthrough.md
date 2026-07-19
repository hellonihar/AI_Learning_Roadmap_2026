# Probability & Statistics with NumPy

This walkthrough demonstrates probability and statistics concepts using only NumPy.

```python
import numpy as np
```

## 1. Random Variables and Distributions

```python
np.random.seed(42)

# Bernoulli(p)
p = 0.7
bernoulli_samples = np.random.binomial(1, p, size=1000)
print(f"Bernoulli({p}) mean: {bernoulli_samples.mean():.3f} (expected {p})")

# Binomial(n, p)
n, p = 10, 0.3
binomial_samples = np.random.binomial(n, p, size=5000)
print(f"Binomial mean: {binomial_samples.mean():.3f} (expected {n*p})")
print(f"Binomial var: {binomial_samples.var():.3f} (expected {n*p*(1-p)})")

# Poisson(lambda)
lam = 4.0
poisson_samples = np.random.poisson(lam, size=5000)
print(f"Poisson mean: {poisson_samples.mean():.3f} (expected {lam})")

# Normal(mu, sigma)
mu, sigma = 5.0, 2.0
normal_samples = np.random.normal(mu, sigma, size=5000)
print(f"Normal mean: {normal_samples.mean():.3f} (expected {mu})")
print(f"Normal std: {normal_samples.std():.3f} (expected {sigma})")

# Uniform(a, b)
a, b = 0.0, 10.0
uniform_samples = np.random.uniform(a, b, size=5000)
print(f"Uniform mean: {uniform_samples.mean():.3f} (expected {(a+b)/2})")

# Exponential(lambda)
scale = 1.0
exp_samples = np.random.exponential(scale, size=5000)
print(f"Exponential mean: {exp_samples.mean():.3f} (expected {scale})")
```

## 2. PDF and CDF for Normal Distribution

```python
# Normal PDF manually
def normal_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Normal CDF (using error function)
def normal_cdf(x, mu=0, sigma=1):
    return 0.5 * (1 + np.math.erf((x - mu) / (sigma * np.sqrt(2))))

x_vals = np.linspace(-4, 4, 9)
for x in x_vals:
    pdf = normal_pdf(x)
    cdf = normal_cdf(x)
    print(f"x={x:+.1f}: PDF={pdf:.4f}, CDF={cdf:.4f}")

# Verify: CDF'(x) ≈ PDF(x)
h = 1e-6
x = 1.0
cdf_deriv = (normal_cdf(x + h) - normal_cdf(x - h)) / (2 * h)
print(f"\nAt x=1: dCDF/dx ≈ {cdf_deriv:.6f}, PDF = {normal_pdf(x):.6f}")
```

## 3. Law of Large Numbers & Central Limit Theorem

```python
# Law of Large Numbers: sample mean converges to true mean
def demonstrate_lln(distribution, params, true_mean, n_samples=10000):
    means = []
    cumulative = 0
    for i in range(1, n_samples + 1):
        cumulative += distribution(**params)
        if i % 500 == 0:
            means.append(cumulative / i)
    return means

# CLT: sum of i.i.d. random variables → Normal
def demonstrate_clt(distribution, params, n_samples=10000, sample_size=30):
    sample_means = []
    for _ in range(n_samples):
        sample = np.array([distribution(**params) for _ in range(sample_size)])
        sample_means.append(sample.mean())
    return np.array(sample_means)

# Demonstrate with Exponential(1) — skewed distribution
exp_sample = lambda **_: np.random.exponential(1.0)
clt_means = demonstrate_clt(exp_sample, {}, n_samples=5000, sample_size=30)
print(f"CLT demo (Exp(1) means): mean={clt_means.mean():.3f}, std={clt_means.std():.3f}")
print(f"Expected mean=1.0, std=1/sqrt(30)={1/np.sqrt(30):.3f}")

# Compare histogram of CLT means to Normal approximation
z_scores = (clt_means - 1.0) / (1.0 / np.sqrt(30))
within_2sigma = np.sum(np.abs(z_scores) < 2) / len(z_scores)
print(f"Proportion within 2σ: {within_2sigma:.3f} (expected ~0.954)")
```

## 4. MLE: Estimating Bernoulli Parameter

```python
# MLE for Bernoulli(p): p_hat = (1/n) * sum(x_i)
true_p = 0.35
n_trials = [10, 50, 200, 1000]

print("MLE for Bernoulli parameter:")
for n in n_trials:
    samples = np.random.binomial(1, true_p, size=n)
    p_mle = samples.mean()
    print(f"  n={n:4d}: p_hat={p_mle:.4f} (true p={true_p})")

# MLE for Normal(mu, sigma^2)
true_mu, true_sigma = 3.0, 1.5
samples = np.random.normal(true_mu, true_sigma, size=500)
mu_mle = samples.mean()
sigma_mle = np.sqrt(((samples - mu_mle)**2).mean())
print(f"\nMLE for Normal: mu_hat={mu_mle:.3f}, sigma_hat={sigma_mle:.3f}")
print(f"True: mu={true_mu}, sigma={true_sigma}")
```

## 5. Naive Bayes Classifier

```python
# Simple Gaussian Naive Bayes implementation
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'prior': len(X_c) / len(X),
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0) + 1e-9  # avoid division by zero
            }

    def _log_likelihood(self, x, mean, std):
        # log PDF of normal distribution
        return -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((x - mean)**2) / (std**2))

    def predict(self, X):
        preds = []
        for x in X:
            scores = {}
            for c in self.classes:
                p = self.params[c]
                log_prior = np.log(p['prior'])
                log_lik = self._log_likelihood(x, p['mean'], p['std'])
                scores[c] = log_prior + log_lik
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

# Test on synthetic data
np.random.seed(42)
n_per_class = 100
X0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_per_class)
X1 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], n_per_class)
X = np.vstack([X0, X1])
y = np.array([0]*n_per_class + [1]*n_per_class)

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Naive Bayes accuracy: {accuracy:.3f}")
```

## 6. Hypothesis Testing: A/B Test Simulation

```python
# Simulate A/B test for conversion rates
np.random.seed(42)
n_A, n_B = 1000, 1000
p_A_true, p_B_true = 0.08, 0.10

# Generate data
conversions_A = np.random.binomial(n_A, p_A_true)
conversions_B = np.random.binomial(n_B, p_B_true)

# Conversion rates
p_A_hat = conversions_A / n_A
p_B_hat = conversions_B / n_B
print(f"Conversion A: {p_A_hat:.4f}, B: {p_B_hat:.4f}")

# Z-test for two proportions
# H0: p_A = p_B, H1: p_A != p_B
p_pool = (conversions_A + conversions_B) / (n_A + n_B)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z_stat = (p_B_hat - p_A_hat) / se

# Two-tailed p-value
p_value = 2 * (1 - normal_cdf(abs(z_stat)))
print(f"Z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")
print(f"Reject H0 at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
```

## 7. Bootstrap Confidence Intervals

```python
# Bootstrap estimate of mean and confidence interval
data = np.random.exponential(scale=5.0, size=200)
n_bootstrap = 10000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means[i] = sample.mean()

ci_low = np.percentile(bootstrap_means, 2.5)
ci_high = np.percentile(bootstrap_means, 97.5)
print(f"Sample mean: {data.mean():.3f}")
print(f"95% Bootstrap CI: ({ci_low:.3f}, {ci_high:.3f})")
```

These simulations demonstrate the core probabilistic concepts that underpin machine learning algorithms.
