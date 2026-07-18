# Inheritance & Dunder Methods

## Inheritance
```python
class BaseModel:
    def fit(self, X, y): ...
    def predict(self, X): ...

class RegressionModel(BaseModel):
    def predict(self, X):
        return super().predict(X)  # extends parent

class LogisticRegression(BaseModel):
    pass
```

## Abstract Base Class
```python
from abc import ABC, abstractmethod

class Estimator(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...
```

## Dunder Methods
```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):            # print / repr
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):      # +
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):     # *
        return Vector(self.x * scalar, self.y * scalar)

    def __len__(self):             # len()
        return 2

    def __getitem__(self, i):      # v[0]
        return (self.x, self.y)[i]

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)  # Vector(4, 6)
```
