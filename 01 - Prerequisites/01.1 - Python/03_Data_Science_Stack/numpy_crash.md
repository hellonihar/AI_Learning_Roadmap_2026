# NumPy Crash Course

## Creating Arrays
```python
import numpy as np
np.array([1, 2, 3])
np.zeros((3, 4))
np.ones((2, 3))
np.eye(4)
np.arange(10)
np.linspace(0, 1, 5)
np.random.randn(3, 3)
np.random.randint(0, 10, (2, 5))
```

## Shape Manipulation
```python
a = np.arange(12)
a.reshape(3, 4)
a.reshape(-1, 4)     # auto infer rows
a.T                   # transpose
a.ravel()             # flatten
```

## Broadcasting
```python
a = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
a + b   # broadcast b across rows
```

## Linear Algebra
```python
a @ b                 # matrix multiply
np.dot(a, b)
np.linalg.inv(a)
np.linalg.eig(a)
np.linalg.svd(a)
```

## Aggregations
```python
a.mean(); a.std(); a.sum()
a.min(axis=0); a.max(axis=1)
a.argmin(); a.argmax()
np.percentile(a, 95)
```

## Indexing & Slicing
```python
a[1:3, :2]            # rows 1-2, cols 0-1
a[a > 0.5]            # boolean indexing
a[[0, 2], :]          # fancy indexing
```
