# Type Hints & Mypy

## Basic Annotations
```python
def add(a: int, b: int) -> int:
    return a + b

name: str = "AI"
count: int = 42
```

## Complex Types
```python
from typing import List, Dict, Tuple, Optional, Union, Any

def process(items: List[int]) -> Dict[str, float]:
    return {"mean": sum(items) / len(items)}

config: Dict[str, Any] = {"lr": 1e-3, "device": "cuda"}
maybe: Optional[int] = None  # same as Union[int, None]
id_or_name: Union[int, str] = "abc"
batch: Tuple[torch.Tensor, torch.Tensor] = (X, y)
```

## Callable & TypeVar
```python
from typing import Callable, TypeVar

T = TypeVar("T")
def transform(fn: Callable[[T], T], data: List[T]) -> List[T]:
    return [fn(x) for x in data]
```

## Running Mypy
```bash
pip install mypy
mypy script.py --strict
```
