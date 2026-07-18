# Stack

```python
# List as stack
stack = []
stack.append(1)       # push
stack.append(2)
stack.pop()           # pop → 2
stack[-1]             # peek (if not empty)
len(stack) == 0       # isEmpty
```

# Monotonic Stack

```python
# Next greater element (to the right)
def next_greater(arr):
    result = [-1] * len(arr)
    stack = []
    for i in range(len(arr)):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result
```

# Valid Parentheses

```python
def is_valid(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
```

## Common Problems
- Valid parentheses
- Min stack (O(1) getMin)
- Daily temperatures (next greater)
- Largest rectangle in histogram
- Remove adjacent duplicates
