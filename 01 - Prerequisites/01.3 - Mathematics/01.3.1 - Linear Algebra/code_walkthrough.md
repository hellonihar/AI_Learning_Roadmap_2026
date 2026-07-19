# Linear Algebra with NumPy

This walkthrough demonstrates key linear algebra concepts using only NumPy.

```python
import numpy as np
```

## 1. Vectors

```python
# Create vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)  # or a @ b
print(f"Dot product: {dot}")  # 1*4 + 2*5 + 3*6 = 32

# Norms
l2_norm = np.linalg.norm(a)  # sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.74
l1_norm = np.linalg.norm(a, ord=1)  # |1| + |2| + |3| = 6
print(f"L2 norm: {l2_norm:.4f}, L1 norm: {l1_norm}")

# Cosine similarity
cos_sim = dot / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.4f}")

# Linear independence check via rank
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
matrix = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(matrix)
print(f"Vectors are {'linearly independent' if rank == 3 else 'dependent'} (rank={rank})")

# Dependent vectors
v3_dep = np.array([2, 3, 0])  # 2*v1 + 3*v2
matrix_dep = np.column_stack([v1, v2, v3_dep])
rank_dep = np.linalg.matrix_rank(matrix_dep)
print(f"Dependent set rank: {rank_dep}")
```

## 2. Matrices

```python
# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication (note: not element-wise)
C = A @ B  # or np.matmul(A, B)
print(f"A @ B =\n{C}")
# [[1*5 + 2*7, 1*6 + 2*8],   = [[19, 22],
#  [3*5 + 4*7, 3*6 + 4*8]]      [43, 50]]

# Transpose
print(f"A^T =\n{A.T}")

# Identity matrix
I = np.eye(3)
print(f"I_3 =\n{I}")

# Inverse
A_inv = np.linalg.inv(A)
print(f"A^{-1} =\n{A_inv}")
print(f"A @ A^{-1} =\n{A @ A_inv}")  # Should be ~identity

# Determinant
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.4f}")

# Solving linear system: Ax = b
b_vec = np.array([1, 2])
x = np.linalg.solve(A, b_vec)
print(f"Solution to Ax = b: x = {x}")
print(f"Check: A @ x = {A @ x}")
```

## 3. Eigenvalues and Eigenvectors

```python
# Symmetric matrix (guarantees real eigenvalues)
M = np.array([[2, 1], [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: M @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = M @ v
    rhs = lam * v
    print(f"λ = {lam:.4f}: Mv = {lhs}, λv = {rhs}, match: {np.allclose(lhs, rhs)}")
```

## 4. Singular Value Decomposition (SVD)

```python
# Create a matrix with low-rank structure
np.random.seed(42)
m, n = 5, 4
A = np.random.randn(m, n)

# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)
print(f"U shape: {U.shape}")
print(f"Singular values: {s}")
print(f"V^T shape: {Vt.shape}")

# Reconstruct A from SVD
Sigma = np.zeros((m, n))
Sigma[:n, :n] = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Low-rank approximation (keep top k singular values)
k = 2
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]
A_approx = U_k @ np.diag(s_k) @ Vt_k
print(f"Rank-{k} approximation error: {np.linalg.norm(A - A_approx):.4f}")

# PCA via SVD (center data first)
X = np.random.randn(20, 3)
X_centered = X - X.mean(axis=0)
_, s_pca, Vt_pca = np.linalg.svd(X_centered, full_matrices=False)
explained_variance = s_pca**2 / (s_pca**2).sum()
print(f"Explained variance ratio: {explained_variance}")
print(f"Principal components (rows of V^T):\n{Vt_pca}")
```

## 5. Matrix Calculus with NumPy

```python
# Compute gradient numerically for f(x) = x^T A x
def grad_quadratic(x, A):
    """Gradient of f(x) = x^T A x with respect to x"""
    return (A + A.T) @ x

x = np.array([1.0, 2.0])
A_sym = np.array([[3.0, 1.0], [1.0, 4.0]])
grad = grad_quadratic(x, A_sym)
print(f"Gradient of x^T A x at x={x}: {grad}")

# Numerical verification
eps = 1e-6
f0 = x @ A_sym @ x
f1 = (x + np.array([eps, 0])) @ A_sym @ (x + np.array([eps, 0]))
numerical_grad_x0 = (f1 - f0) / eps
print(f"Numerical gradient (x0): {numerical_grad_x0:.4f} (expected {grad[0]:.4f})")
```

## 6. Word Embedding Analogy (Simplified)

```python
# Simplified word embedding analogy: king - man + woman ≈ queen
embeddings = {
    'king': np.array([0.9, 0.8, 0.5]),
    'man': np.array([0.7, 0.6, 0.3]),
    'woman': np.array([0.5, 0.7, 0.8]),
    'queen': np.array([0.8, 0.9, 0.9]),
    'prince': np.array([0.7, 0.7, 0.4]),
}

target = embeddings['king'] - embeddings['man'] + embeddings['woman']

# Find closest by cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = {word: cosine_sim(target, vec) for word, vec in embeddings.items()}
ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print("Analogy: king - man + woman")
for word, sim in ranked:
    print(f"  {word}: {sim:.4f}")
```

All operations above are fundamental building blocks for neural networks, PCA, and many other ML algorithms.
