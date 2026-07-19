# Information Theory with NumPy

This walkthrough demonstrates information theory concepts using only NumPy.

```python
import numpy as np
```

## 1. Entropy

```python
def entropy(p, base=2):
    """Compute entropy H(P) for discrete distribution p"""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # avoid log(0)
    if base == 2:
        return -np.sum(p * np.log2(p))
    else:
        return -np.sum(p * np.log(p))

# Fair coin
fair = np.array([0.5, 0.5])
print(f"Fair coin entropy: {entropy(fair):.4f} bits")

# Biased coin
biased = np.array([0.9, 0.1])
print(f"Biased coin entropy: {entropy(biased):.4f} bits")

# Almost deterministic
det = np.array([0.999, 0.001])
print(f"Almost deterministic: {entropy(det):.4f} bits")

# Uniform distribution over 8 outcomes
uniform_8 = np.ones(8) / 8
print(f"Uniform(8) entropy: {entropy(uniform_8):.4f} bits (max = log2(8) = 3)")

# 3-sided die
p = np.array([0.2, 0.3, 0.5])
print(f"3-sided die: {entropy(p):.4f} bits")

# Differential entropy for Gaussian N(μ, σ²)
def gaussian_differential_entropy(sigma, base=2):
    if base == 2:
        return 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
    else:
        return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

for sigma in [0.5, 1.0, 2.0]:
    print(f"N(0, {sigma**2}) differential entropy: {gaussian_differential_entropy(sigma):.4f} nats")
```

## 2. Cross-Entropy and KL Divergence

```python
def cross_entropy(p, q, base=2):
    """H(P, Q) = -sum p(x) log q(x)"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if base == 2:
        return -np.sum(p * np.log2(q))
    else:
        return -np.sum(p * np.log(q))

def kl_divergence(p, q, base=2):
    """D_KL(P || Q) = sum p(x) log(p(x)/q(x))"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if base == 2:
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    else:
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# True distribution P, approximate Q
p_true = np.array([0.7, 0.2, 0.1])
q_approx = np.array([0.6, 0.3, 0.1])

print(f"P = {p_true}")
print(f"Q = {q_approx}")
print(f"Cross-entropy H(P,Q): {cross_entropy(p_true, q_approx):.4f} bits")
print(f"Entropy H(P): {entropy(p_true):.4f} bits")
print(f"KL D_KL(P||Q): {kl_divergence(p_true, q_approx):.4f} bits")
print(f"Verify: H(P) + KL = {entropy(p_true) + kl_divergence(p_true, q_approx):.4f} bits")

# Asymmetric: D_KL(P||Q) != D_KL(Q||P)
print(f"\nD_KL(P||Q) = {kl_divergence(p_true, q_approx):.4f}")
print(f"D_KL(Q||P) = {kl_divergence(q_approx, p_true):.4f}")
print(f"Note: KL is NOT symmetric!")

# KL(P||Q) = 0 when P = Q
p_same = p_true.copy()
print(f"\nD_KL(P||P) = {kl_divergence(p_true, p_same):.4f}")
```

## 3. Mutual Information

```python
def mutual_information(joint, base=2):
    """I(X;Y) from joint distribution P(X,Y)"""
    joint = np.asarray(joint, dtype=float)
    # Marginal distributions
    p_x = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    product = p_x @ p_y  # P(X) * P(Y)
    
    mi = 0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:
                if base == 2:
                    mi += joint[i, j] * np.log2(joint[i, j] / product[i, j])
                else:
                    mi += joint[i, j] * np.log(joint[i, j] / product[i, j])
    return mi

# Independent X and Y
joint_indep = np.array([[0.2, 0.3], [0.2, 0.3]])
print(f"Independent: I(X;Y) = {mutual_information(joint_indep):.4f} bits")

# Dependent X and Y
joint_dep = np.array([[0.4, 0.1], [0.1, 0.4]])
print(f"Dependent: I(X;Y) = {mutual_information(joint_dep):.4f} bits")

# Deterministic relationship
joint_det = np.array([[0.5, 0.0], [0.0, 0.5]])
print(f"Deterministic: I(X;Y) = {mutual_information(joint_det):.4f} bits (max = H(X) = 1)")

# Verify: I(X;Y) = H(X) + H(Y) - H(X,Y)
def entropy_2d(joint):
    flat = joint.flatten()
    flat = flat[flat > 0]
    return -np.sum(flat * np.log2(flat))

p_x = joint_dep.sum(axis=1)
p_y = joint_dep.sum(axis=0)
h_x = entropy(p_x)
h_y = entropy(p_y)
h_xy = entropy_2d(joint_dep)
mi_via_entropy = h_x + h_y - h_xy
print(f"\nI(X;Y) via H(X)+H(Y)-H(X,Y): {mi_via_entropy:.4f} bits")
```

## 4. Jensen-Shannon Divergence

```python
def js_divergence(p, q, base=2):
    """Jensen-Shannon divergence D_JS(P || Q)"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    if base == 2:
        return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)
    else:
        return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)

# Compare KL vs JSD
p = np.array([1.0, 0.0, 0.0])
q = np.array([0.0, 0.5, 0.5])
print(f"P = {p}")
print(f"Q = {q}")
print(f"KL(P||Q): {kl_divergence(p, q):.4f} bits")
print(f"KL(Q||P): {kl_divergence(q, p):.4f} bits")
print(f"JSD(P||Q): {js_divergence(p, q):.4f} bits")
print(f"JSD is symmetric and bounded (0 to {np.log2(2):.0f} bit)")

# JSD between two Gaussians (via sampling)
np.random.seed(42)
samples_p = np.random.normal(0, 1, 10000)
samples_q = np.random.normal(1, 1, 10000)

# Discretize into bins
bins = np.linspace(-4, 5, 50)
hist_p, _ = np.histogram(samples_p, bins=bins, density=True)
hist_q, _ = np.histogram(samples_q, bins=bins, density=True)
# Normalize
hist_p /= hist_p.sum()
hist_q /= hist_q.sum()

jsd = js_divergence(hist_p, hist_q)
print(f"\nJSD between N(0,1) and N(1,1): {jsd:.4f} bits")
```

## 5. Information Gain for Decision Trees

```python
def information_gain(y, feature_values):
    """Compute information gain of splitting y on feature"""
    # H(Y) before split
    total_entropy = entropy(np.bincount(y) / len(y))
    
    # Weighted entropy after split
    unique_vals = np.unique(feature_values)
    conditional_entropy = 0
    for v in unique_vals:
        mask = feature_values == v
        y_subset = y[mask]
        if len(y_subset) > 0:
            p_subset = np.bincount(y_subset) / len(y_subset)
            conditional_entropy += (len(y_subset) / len(y)) * entropy(p_subset)
    
    return total_entropy - conditional_entropy

# Simple example: predict whether to play tennis
# Feature: "Outlook" with values [Sunny, Overcast, Rainy]
# Target: [Play, Don't Play]
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
outlook = np.array([0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2])
# 0=Sunny, 1=Overcast, 2=Rainy

ig = information_gain(y, outlook)
print(f"Information gain (Outlook): {ig:.4f} bits")

# Compare with a random split
random_feat = np.random.randint(0, 3, len(y))
ig_random = information_gain(y, random_feat)
print(f"Information gain (Random): {ig_random:.4f} bits (should be lower)")
```

## 6. Cross-Entropy Loss for Classification

```python
# Cross-entropy as a loss function
def cross_entropy_loss(y_true, y_pred):
    """y_true: one-hot or class indices, y_pred: predicted probabilities"""
    n = len(y_true)
    if y_pred.ndim == 1:
        # Binary classification
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    else:
        # Multi-class
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

# Binary classification
y_true_bin = np.array([1, 0, 1, 0, 1])
y_pred_bin = np.array([0.9, 0.2, 0.8, 0.1, 0.95])
loss = cross_entropy_loss(y_true_bin, y_pred_bin)
print(f"Cross-entropy (binary): {loss:.4f}")

# Perfect predictions
y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
print(f"Cross-entropy (perfect): {cross_entropy_loss(y_true_bin, y_pred_perfect):.4f}")

# Multi-class
y_true_mc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_mc = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
print(f"Cross-entropy (multi-class): {cross_entropy_loss(y_true_mc, y_pred_mc):.4f}")
```

## 7. KL Divergence in Variational Inference

```python
# Approximate a complex distribution P with a simpler Q
# P is a mixture of two Gaussians, Q is a single Gaussian
def mixture_pdf(x, means, stds, weights):
    """Mixture of Gaussians PDF"""
    pdf = np.zeros_like(x)
    for mu, sigma, w in zip(means, stds, weights):
        pdf += w * np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# P: mixture of N(-3, 1) and N(3, 1)
x_grid = np.linspace(-8, 8, 1000)
p_pdf = mixture_pdf(x_grid, [-3, 3], [1, 1], [0.5, 0.5])

# Approximate Q with N(0, 3) (covers both modes)
q1_pdf = gaussian_pdf(x_grid, 0, 3)
# Approximate Q with N(3, 1) (mode-seeking)
q2_pdf = gaussian_pdf(x_grid, 3, 1)

# KL divergence (using numerical integration)
def kl_numerical(p, q, dx):
    mask = (p > 1e-12) & (q > 1e-12)
    return np.sum(p[mask] * np.log(p[mask] / q[mask])) * dx

dx = x_grid[1] - x_grid[0]
p /= np.sum(p) * dx
q1_pdf /= np.sum(q1_pdf) * dx
q2_pdf /= np.sum(q2_pdf) * dx

kl_forward_q1 = kl_numerical(p, q1_pdf, dx)
kl_forward_q2 = kl_numerical(p, q2_pdf, dx)
print(f"KL(P || Q_wide): {kl_forward_q1:.4f} (mass-covering)")
print(f"KL(P || Q_narrow): {kl_forward_q2:.4f} (mode-seeking: lower KL!)")

# Note: minimizing KL(P||Q) would pick Q2 (mode-seeking)
# Minimizing KL(Q||P) would pick Q1 (mass-covering)
kl_reverse_q1 = kl_numerical(q1_pdf, p, dx)
kl_reverse_q2 = kl_numerical(q2_pdf, p, dx)
print(f"KL(Q_wide || P): {kl_reverse_q1:.4f}")
print(f"KL(Q_narrow || P): {kl_reverse_q2:.4f}")
print("Forward KL (P||Q) is mode-seeking; Reverse KL (Q||P) is mass-covering")
```

These implementations show how information-theoretic quantities are computed and used in ML contexts.
