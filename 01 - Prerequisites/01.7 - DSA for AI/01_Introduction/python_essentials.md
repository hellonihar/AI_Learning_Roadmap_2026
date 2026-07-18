# Python Essentials for Problem Solving

## Common Patterns

```python
# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict / Counter
from collections import Counter
freq = Counter("abracadabra")

# Two-pointer swap
arr[left], arr[right] = arr[right], arr[left]

# Sliding window
for right in range(n):
    window.add(arr[right])
    while condition:
        window.remove(arr[left])
        left += 1

# Sorting with custom key
sorted(items, key=lambda x: x[1], reverse=True)

# Recursion template
def dfs(state):
    if base_case(state): return result
    for choice in options:
        update(state)
        dfs(state)
        undo(state)
```

## Useful Libraries

| Module | Use |
|---|---|
| `collections.defaultdict` | Auto-initialize missing keys |
| `collections.deque` | O(1) pop from both ends |
| `heapq` | Priority queue / min-heap |
| `bisect` | Binary search on sorted lists |
| `itertools` | permutations, combinations, product |

## Input/Output for Coding

```python
# Single line
n = int(input())
arr = list(map(int, input().split()))

# Multiple lines
import sys
data = sys.stdin.read().split()
```
