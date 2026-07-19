# Probability & Statistics for Machine Learning

Probability provides the mathematical foundation for reasoning under uncertainty, which is fundamental to machine learning. Every ML model makes predictions with some degree of confidence, and probabilistic thinking helps us understand, evaluate, and improve these predictions.

---

## 1. Probability Axioms

Three axioms define probability:

1. **Non-negativity**: $P(A) \geq 0$ for any event $A$.
2. **Normalization**: $P(\Omega) = 1$ where $\Omega$ is the sample space.
3. **Additivity**: For mutually exclusive events $A_1, A_2, \ldots$, $P(\bigcup_i A_i) = \sum_i P(A_i)$.

### Key Rules

- **Complement**: $P(A^c) = 1 - P(A)$
- **Union**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Intersection** (for independent events): $P(A \cap B) = P(A) P(B)$

---

## 2. Conditional Probability and Bayes' Theorem

### Conditional Probability

The probability of $A$ given $B$:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

Two events are **independent** if $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$.

### Bayes' Theorem

The most important theorem in ML:

$$
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
$$

In ML notation:

$$
P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}
$$

- **Posterior**: $P(\theta \mid \mathcal{D})$ — belief about parameters after seeing data
- **Likelihood**: $P(\mathcal{D} \mid \theta)$ — how likely the data is given parameters
- **Prior**: $P(\theta)$ — initial belief about parameters
- **Evidence**: $P(\mathcal{D})$ — probability of the data

**Application — Naive Bayes Classifier**: Assumes features are conditionally independent given the class:

$$
P(y \mid x_1, \ldots, x_d) \propto P(y) \prod_{i=1}^d P(x_i \mid y)
```

---

## 3. Random Variables

A **random variable** assigns a numerical value to each outcome of a random process.

### Discrete Random Variables

Take countable values. Characterized by a **probability mass function (PMF)** $P(X = x)$.

**Bernoulli**: $X \in \{0, 1\}$, $P(X=1) = p$

$$
P(X = x) = p^x (1-p)^{1-x}
$$

**Binomial**: $n$ independent Bernoulli trials, $X =$ number of successes

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

**Poisson**: Models count of rare events over a fixed interval

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

### Continuous Random Variables

Take uncountable values. Characterized by a **probability density function (PDF)** $f(x)$ where $P(a \leq X \leq b) = \int_a^b f(x) dx$.

**Uniform**: $X \sim \mathcal{U}(a, b)$

$$
f(x) = \frac{1}{b-a}, \quad a \leq x \leq b
$$

**Normal (Gaussian)**: $X \sim \mathcal{N}(\mu, \sigma^2)$

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

The **standard normal** has $\mu = 0$, $\sigma = 1$. The **central limit theorem** states that the sum of many i.i.d. random variables converges to a normal distribution.

**Exponential**: Models waiting times

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

---

## 4. Expectation and Variance

### Expectation (Mean)

The average value weighted by probability:

$$
\mathbb{E}[X] = \begin{cases}
\sum_x x P(X=x) & \text{discrete} \\
\int x f(x) dr & \text{continuous}
\end{cases}
$$

**Linearity**: $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$

### Variance

Measures spread around the mean:

$$
\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

Standard deviation: $\sigma(X) = \sqrt{\text{Var}(X)}$

### Covariance and Correlation

For two random variables:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
$$

Correlation (normalized covariance):

$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

In ML, the covariance matrix captures pairwise feature relationships and is central to PCA, Gaussian processes, and multivariate normal distributions.

---

## 5. Maximum Likelihood (MLE) and Maximum A Posteriori (MAP)

### MLE

Find parameters $\theta$ that maximize the likelihood of observed data:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D} \mid \theta) = \arg\max_\theta \sum_{i=1}^n \log P(x_i \mid \theta)
$$

**Example**: For Bernoulli $X_1, \ldots, X_n \sim \text{Bern}(p)$:

$$
\hat{p}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i
$$

### MAP

Incorporates a prior distribution:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta \left[ \log P(\mathcal{D} \mid \theta) + \log P(\theta) \right]
$$

MLE is a special case of MAP with a uniform prior. MAP provides regularization — for example, a Gaussian prior on weights corresponds to L2 regularization.

---

## 6. Hypothesis Testing

Used to make decisions from data:

1. **Null hypothesis** $H_0$: default assumption
2. **Alternative hypothesis** $H_1$: what we want to prove
3. **Test statistic**: computed from data
4. **p-value**: probability of observing the statistic (or more extreme) under $H_0$
5. **Significance level** $\alpha$: reject $H_0$ if $p < \alpha$

### Application: A/B Testing

Compare two versions (A and B) to see if there is a statistically significant difference. Typically uses a t-test or z-test on conversion rates.

---

## 7. Applications in AI/ML

| Concept | Application |
|---------|------------|
| Conditional probability | Naive Bayes, Markov chains, Bayesian networks |
| Bayes' Theorem | Bayesian inference, probabilistic programming |
| Normal distribution | Assumption in linear regression, VAEs, GANs |
| Bernoulli/Binomial | Binary classification, dropout regularization |
| Expectation | Loss functions, risk minimization |
| MLE | Training most ML models (minimizing NLL) |
| MAP | Regularized regression (Ridge, Lasso) |
| Hypothesis testing | Feature selection, model comparison, A/B testing |

---

## Summary

Probability and statistics give us the language to express uncertainty, quantify confidence, and make principled decisions from data. MLE and MAP directly connect probabilistic reasoning to model training, making this section essential for understanding how and why machine learning algorithms work.
