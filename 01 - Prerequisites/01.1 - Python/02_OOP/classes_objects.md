# Classes & Objects

## Basic Class
```python
class Model:
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        print(f"{self.name} trained")

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Not fitted yet")
        return [0] * len(X)

model = Model("regressor")
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Instance / Class / Static Methods
```python
class Config:
    default_lr = 0.01

    def __init__(self, lr=None):
        self.lr = lr or self.default_lr

    @classmethod
    def from_yaml(cls, path):
        import yaml
        data = yaml.safe_load(open(path))
        return cls(**data)

    @staticmethod
    def validate_lr(lr):
        return 0 < lr < 1
```

## Property Decorator
```python
class Layer:
    def __init__(self, units):
        self._units = units

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value < 1:
            raise ValueError("Must be positive")
        self._units = value
```
