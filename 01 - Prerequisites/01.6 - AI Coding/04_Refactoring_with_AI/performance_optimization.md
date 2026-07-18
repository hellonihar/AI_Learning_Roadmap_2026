# Performance Optimization

AI can identify slow patterns and suggest faster alternatives.

## Loop → Vectorized

```python
# Before (slow):
def compute_distances(points, reference):
    result = []
    for p in points:
        dist = ((p[0] - reference[0])**2 + (p[1] - reference[1])**2)**0.5
        result.append(dist)
    return result
```

**Prompt:** "Vectorize this using NumPy for better performance"

```python
# After (vectorized):
import numpy as np

def compute_distances(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sqrt(((points - reference)**2).sum(axis=1))
```

## Caching

```python
# Before:
def get_embedding(text):
    return model.encode(text)  # called repeatedly with same text
```

**Prompt:** "Add LRU caching to avoid recomputing for the same input"

```python
# After:
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)
```

## Performance Prompts

```
- "Optimize this for speed / memory"
- "Replace this loop with a vectorized operation"
- "Add caching for repeated expensive calls"
- "Make this lazy using generators instead of lists"
- "Use collections.Counter instead of manual counting"
- "Replace pandas apply with vectorized operations"
```
