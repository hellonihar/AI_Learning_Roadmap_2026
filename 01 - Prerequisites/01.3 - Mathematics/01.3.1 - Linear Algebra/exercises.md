# Linear Algebra — Exercises

## Conceptual Questions

1. **Linear Independence**: Determine whether the following set of vectors is linearly independent:
   $\mathbf{v}_1 = [1, 2, 3]$, $\mathbf{v}_2 = [4, 5, 6]$, $\mathbf{v}_3 = [7, 8, 9]$. Explain why or why not.

2. **Eigenvalue Interpretation**: A symmetric matrix $\mathbf{A}$ has eigenvalues $\lambda_1 = 10$, $\lambda_2 = 2$, $\lambda_3 = 0.5$. What does this tell you about the matrix's properties? Is it invertible? What is $\det(\mathbf{A})$?

3. **Matrix Multiplication Dimensions**: You have a weight matrix $\mathbf{W} \in \mathbb{R}^{128 \times 256}$ and an input batch $\mathbf{X} \in \mathbb{R}^{64 \times 256}$. What is the dimension of $\mathbf{X}\mathbf{W}^T$? What does each row represent?

4. **SVD Intuition**: Explain in your own words why keeping only the largest $k$ singular values gives the best rank-$k$ approximation to a matrix. What happens to the approximation error as $k$ increases?

## Coding Exercises

5. **Neural Network Forward Pass**: Implement a single neural network layer $h = \text{ReLU}(\mathbf{W}\mathbf{x} + \mathbf{b})$ using NumPy. Use $\mathbf{W} \in \mathbb{R}^{4 \times 3}$, random $\mathbf{x} \in \mathbb{R}^{3}$, and $\mathbf{b} \in \mathbb{R}^{4}$. Verify the output shape is (4,).

6. **PCA from Scratch**: Write a function `pca(X, k)` that:
   - Centers the data matrix $\mathbf{X}$ (shape $n \times d$)
   - Computes the covariance matrix
   - Finds the top $k$ eigenvectors using `np.linalg.eig`
   - Returns the projected data of shape $n \times k$
   Test on a random dataset with $n=100$, $d=5$, $k=2$.

7. **Image Compression with SVD**: Given a grayscale image matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, write code to compress it by keeping only the top $k$ singular values. Compute the compression ratio and the reconstruction error. Test with a randomly generated $20 \times 30$ matrix.

---

## Answers

**Answer 1**: The vectors are linearly dependent because $\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1$. Equivalently, $\text{rank} = 2 < 3$.

**Answer 2**: The matrix is positive definite (all eigenvalues > 0), invertible ($\det \neq 0$), and $\det(\mathbf{A}) = 10 \times 2 \times 0.5 = 10$.

**Answer 3**: $\mathbf{X}\mathbf{W}^T \in \mathbb{R}^{64 \times 128}$. Each row represents the 128-dimensional output features for one input sample after the linear transformation.

**Answer 4**: SVD provides the optimal low-rank approximation in the Frobenius norm (Eckart-Young theorem). The error equals the sum of squares of the discarded singular values: $\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$. As $k$ increases, fewer singular values are discarded, so error decreases.

**Answer 5**:
```python
import numpy as np

np.random.seed(42)
W = np.random.randn(4, 3)
x = np.random.randn(3)
b = np.random.randn(4)

h = np.maximum(0, W @ x + b)  # ReLU activation
print(f"Output shape: {h.shape}")  # (4,)
print(f"Output: {h}")
```

**Answer 6**:
```python
import numpy as np

def pca(X, k):
    # Center the data
    X_centered = X - X.mean(axis=0)
    # Covariance matrix
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Top k eigenvectors (sort descending)
    idx = np.argsort(eigenvalues)[::-1][:k]
    W = eigenvectors[:, idx]
    # Project
    return X_centered @ W

np.random.seed(42)
X = np.random.randn(100, 5)
X_proj = pca(X, k=2)
print(f"Projected shape: {X_proj.shape}")  # (100, 2)
```

**Answer 7**:
```python
import numpy as np

def svd_compress(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    original_size = A.shape[0] * A.shape[1]
    compressed_size = A.shape[0] * k + k + k * A.shape[1]
    ratio = compressed_size / original_size
    error = np.linalg.norm(A - A_compressed)
    return A_compressed, ratio, error

A = np.random.randn(20, 30)
A_comp, ratio, err = svd_compress(A, k=5)
print(f"Compression ratio: {ratio:.3f}, Error: {err:.4f}")
```
