# Linear Algebra for Machine Learning

Linear algebra is the mathematical foundation of几乎所有 machine learning and deep learning. From representing data as vectors and matrices to understanding how neural networks transform inputs, linear algebra provides the language and tools for modern AI.

---

## 1. Vectors

A **vector** is an ordered tuple of numbers. In ML, vectors represent data points, features, or weights.

### Dot Product (Inner Product)

The dot product of two vectors $\mathbf{a} = [a_1, a_2, \ldots, a_n]$ and $\mathbf{b} = [b_1, b_2, \ldots, b_n]$ is:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^T \mathbf{b}
$$

Geometrically, $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta$, where $\theta$ is the angle between vectors.

In ML, the dot product measures similarity (e.g., cosine similarity for word embeddings) and computes the weighted sum in a neuron: $z = \mathbf{w}^T \mathbf{x} + b$.

### Vector Norm

A norm measures the magnitude (length) of a vector. The most common is the **Euclidean norm** ($L_2$):

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

The $L_1$ norm sums absolute values: $\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$. $L_1$ and $L_2$ norms are used for regularization (see Optimization section).

### Linear Independence

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is **linearly independent** if no vector can be written as a linear combination of the others. Formally:

$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \ldots + c_k \mathbf{v}_k = 0 \implies c_1 = c_2 = \ldots = c_k = 0
$$

Linear independence is crucial for understanding the dimensionality of data and when a matrix has an inverse.

---

## 2. Matrices

A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a rectangular array of numbers with $m$ rows and $n$ columns.

### Matrix Multiplication

If $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$, their product $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times p}$ is:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

Matrix multiplication is **not commutative** ($\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ generally). In neural networks, each layer computes $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$, where $\mathbf{W}$ is a weight matrix.

### Transpose

The transpose $\mathbf{A}^T$ flips rows and columns: $(\mathbf{A}^T)_{ij} = A_{ji}$.

Properties: $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$, $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$.

### Matrix Inverse

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the inverse $\mathbf{A}^{-1}$ satisfies $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}_n$.

A matrix must be **non-singular** (determinant $\neq 0$, full rank) to be invertible. The inverse solves linear systems: $\mathbf{A}\mathbf{x} = \mathbf{b} \implies \mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$.

### Determinant

The determinant $\det(\mathbf{A})$ is a scalar that encodes the volume scaling factor of the linear transformation. Key properties:
- $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A}) \det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = 1 / \det(\mathbf{A})$
- If $\det(\mathbf{A}) = 0$, the matrix is singular (non-invertible).

---

## 3. Eigenvalues and Eigenvectors

An **eigenvector** $\mathbf{v}$ of a square matrix $\mathbf{A}$ satisfies:

$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$

where $\lambda$ is the corresponding **eigenvalue**. Intuitively, applying $\mathbf{A}$ to $\mathbf{v}$ stretches or shrinks it without changing direction.

Eigenvalues are found by solving $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$ (the characteristic polynomial).

### Properties
- A symmetric matrix ($\mathbf{A} = \mathbf{A}^T$) has real eigenvalues and orthogonal eigenvectors.
- The sum of eigenvalues equals the trace: $\sum \lambda_i = \text{tr}(\mathbf{A})$.
- The product equals the determinant: $\prod \lambda_i = \det(\mathbf{A})$.

### Application: Principal Component Analysis (PCA)

PCA finds directions of maximum variance in data. The covariance matrix $\mathbf{\Sigma}$ of the data is computed, and its eigenvectors give the principal components. The eigenvalues indicate the variance explained by each component.

---

## 4. Singular Value Decomposition (SVD)

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal matrix of left singular vectors
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: orthogonal matrix of right singular vectors

SVD generalizes eigendecomposition to non-square matrices. It is the workhorse of:
- **Dimensionality reduction**: Truncating small singular values gives a low-rank approximation (used in PCA, recommendation systems).
- **Matrix pseudo-inverse**: $\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T$ for solving least-squares problems.
- **Image compression**: Keep only the top $k$ singular values.

---

## 5. Matrix Calculus

Matrix calculus extends calculus to functions of vectors and matrices.

### Gradient

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is a vector of partial derivatives:

$$
\nabla_{\mathbf{x}} f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]^T
$$

### Useful Identities

- $\nabla_{\mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
- $\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}$
- $\nabla_{\mathbf{X}} \|\mathbf{X}\|_F^2 = 2\mathbf{X}$ (for Frobenius norm)

Matrix calculus is essential for deriving gradient updates in neural network backpropagation.

---

## 6. Real-World AI Applications

| Concept | Application |
|---------|------------|
| Dot product | Attention mechanism in Transformers: $\text{score} = \mathbf{Q}\mathbf{K}^T$ |
| Matrix multiplication | Fully connected layers: $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$ |
| Eigendecomposition | PCA for dimensionality reduction, Spectral clustering |
| SVD | Collaborative filtering (Netflix prize), Latent Semantic Analysis |
| Vector norms | Regularization (L1/L2 weight decay), gradient clipping |
| Linear independence | Determining model parameter identifiability |

Word embeddings (Word2Vec, GloVe) represent words as dense vectors. The dot product between word vectors reflects semantic similarity: $\cos(\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}, \mathbf{v}_{\text{queen}}) \approx 1$.

In neural networks, each layer applies an affine transformation $\mathbf{W}\mathbf{x} + \mathbf{b}$ followed by a nonlinear activation. The weight matrix $\mathbf{W}$ learns to project inputs into progressively more abstract feature spaces.

---

## Summary

Linear algebra provides the foundational toolkit for representing and transforming data in AI. Mastering vectors, matrices, eigendecomposition, and SVD unlocks a deep understanding of how machine learning algorithms work under the hood.
