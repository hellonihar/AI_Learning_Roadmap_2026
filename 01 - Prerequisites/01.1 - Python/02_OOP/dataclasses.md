# Dataclasses & Attrs

## Basic Dataclass
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda"
    metrics: list = field(default_factory=list)

config = TrainingConfig(lr=1e-4, epochs=20)
print(config)  # TrainingConfig(lr=0.0001, batch_size=32, ...)
```

## Frozen (Immutable) Dataclass
```python
@dataclass(frozen=True)
class HyperParams:
    d_model: int
    n_heads: int
    dropout: float = 0.1
```

## Property within Dataclass
```python
@dataclass
class Dataset:
    n_train: int
    n_val: int

    @property
    def total(self) -> int:
        return self.n_train + self.n_val
```
