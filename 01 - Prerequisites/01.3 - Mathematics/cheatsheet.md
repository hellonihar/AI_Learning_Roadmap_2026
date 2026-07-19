# Mathematics for AI — Cheatsheet

## 01.3.1 — Linear Algebra

| Concept | Formula |
|---------|---------|
| Dot product | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ |
| L2 norm | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ |
| L1 norm | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ |
| Matrix mult. | $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$ |
| Transpose | $(\mathbf{A}^T)_{ij} = A_{ji}$ |
| Inverse | $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ |
| Determinant | $\det(\mathbf{A})$ = volume scaling factor |
| Eigendecomp | $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ |
| SVD | $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ |
| Trace | $\text{tr}(\mathbf{A}) = \sum_i A_{ii} = \sum_i \lambda_i$ |
| Frobenius norm | $\|\mathbf{A}\|_F = \sqrt{\sum_i \sum_j A_{ij}^2}$ |
| Matrix-vector grad | $\nabla_{\mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$ |
| Quadratic form grad | $\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |

---

## 01.3.2 — Calculus

| Concept | Formula |
|---------|---------|
| Derivative | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ |
| Power rule | $\frac{d}{dx} x^n = n x^{n-1}$ |
| Chain rule | $\frac{d}{dx} f(g(x)) = f'(g(x)) g'(x)$ |
| Gradient | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^T$ |
| Gradient descent | $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \nabla f(\mathbf{x}^{(t)})$ |
| Jacobian | $\mathbf{J}_{ij} = \partial f_i / \partial x_j$ |
| Hessian | $\mathbf{H}_{ij} = \partial^2 f / \partial x_i \partial x_j$ |
| Taylor (1st order) | $f(x) \approx f(a) + f'(a)(x-a)$ |
| Taylor (2nd order) | $f(\mathbf{x}+\Delta) \approx f(\mathbf{x}) + \nabla f^T \Delta + \frac{1}{2} \Delta^T \mathbf{H} \Delta$ |
| Sigmoid derivative | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| Softplus | $\frac{d}{dx} \log(1+e^x) = \sigma(x)$ |

---

## 01.3.3 — Probability & Statistics

| Concept | Formula |
|---------|---------|
| Conditional prob. | $P(A \mid B) = P(A \cap B) / P(B)$ |
| Bayes' Theorem | $P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}$ |
| Bernoulli PMF | $P(X=x) = p^x (1-p)^{1-x}$ |
| Binomial PMF | $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$ |
| Poisson PMF | $P(X=k) = \lambda^k e^{-\lambda} / k!$ |
| Normal PDF | $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ |
| Uniform PDF | $f(x) = 1/(b-a), \; a \leq x \leq b$ |
| Exponential PDF | $f(x) = \lambda e^{-\lambda x}, \; x \geq 0$ |
| Expectation | $\mathbb{E}[X] = \sum x P(x) \; \text{or} \; \int x f(x) dx$ |
| Variance | $\text{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ |
| Covariance | $\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ |
| MLE | $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \log P(\mathcal{D} \mid \theta)$ |
| MAP | $\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log P(\mathcal{D} \mid \theta) + \log P(\theta)]$ |
| Law of total prob. | $P(B) = \sum_i P(B \mid A_i) P(A_i)$ |
| Central Limit Thm | $\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu, \sigma^2/n)$ |

---

## 01.3.4 — Optimization

| Concept | Formula |
|---------|---------|
| Convex function | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ |
| Batch GD | $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L(\mathbf{w}^{(t)})$ |
| SGD | $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L_i(\mathbf{w}^{(t)})$ |
| Momentum | $\mathbf{v}^{(t+1)} = \beta \mathbf{v}^{(t)} + \nabla L^{(t)}$; $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \mathbf{v}^{(t+1)}$ |
| Adam | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$; $\hat{m}_t = m_t/(1-\beta_1^t)$; $\hat{v}_t = v_t/(1-\beta_2^t)$; $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$ |
| Lagrangian | $\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$ |
| L1 penalty | $\tilde{L} = L + \lambda \|\mathbf{w}\|_1$ |
| L2 penalty | $\tilde{L} = L + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$ |
| Weight decay | $\mathbf{w}^{(t+1)} = (1 - \eta\lambda)\mathbf{w}^{(t)} - \eta \nabla L^{(t)}$ |
| Softmax | $\sigma(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$ |

---

## 01.3.5 — Information Theory

| Concept | Formula |
|---------|---------|
| Entropy | $H(X) = -\sum_x P(x) \log P(x)$ |
| Joint entropy | $H(X,Y) = -\sum_{x,y} P(x,y) \log P(x,y)$ |
| Conditional entropy | $H(Y \mid X) = H(X,Y) - H(X)$ |
| Cross-entropy | $H(P,Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{\text{KL}}(P \parallel Q)$ |
| KL divergence | $D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ |
| Mutual information | $I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} = H(X) - H(X \mid Y)$ |
| JS divergence | $D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$, $M = \frac{P+Q}{2}$ |
| Chain rule (entropy) | $H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_{<i})$ |
| Information gain | $\text{IG}(Y, A) = H(Y) - H(Y \mid A) = I(Y; A)$ |
| Perplexity | $\text{PPL} = \exp(H(P, \hat{P}))$ |
| ELBO (VAE) | $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \parallel p(z))$ |
| KL for Gaussians | $D_{\text{KL}}(\mathcal{N}(\mu_1,\sigma_1^2) \parallel \mathcal{N}(\mu_2,\sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$ |
| Data processing ineq. | $I(X; Z) \leq I(X; Y)$ if $X \to Y \to Z$ |

---

## Common NumPy Functions

```python
np.dot(a, b)          # dot product
np.linalg.norm(x)     # L2 norm
np.linalg.inv(A)      # matrix inverse
np.linalg.det(A)      # determinant
np.linalg.eig(A)      # eigenvalues/eigenvectors
np.linalg.svd(A)      # SVD
np.linalg.solve(A, b) # solve Ax = b
np.linalg.matrix_rank(A) # rank of matrix
```

## Common Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ |
| ReLU | $\max(0, x)$ | $1_{x>0}$ |
| Tanh | $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$ | $1 - \tanh^2(x)$ |
| Softmax | $\sigma(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$ | $\sigma_i(\delta_{ij} - \sigma_j)$ |
