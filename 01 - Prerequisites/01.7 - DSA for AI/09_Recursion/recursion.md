# Recursion Fundamentals

```python
# Base case + recursive case
def factorial(n):
    if n <= 1: return 1            # base
    return n * factorial(n - 1)    # recursive

# Fibonacci
def fib(n):
    if n <= 1: return n
    return fib(n - 1) + fib(n - 2)
```

# Recursion Tree

```
fib(4)
├── fib(3)
│   ├── fib(2)
│   │   ├── fib(1) → 1
│   │   └── fib(0) → 0
│   └── fib(1) → 1
└── fib(2)
    ├── fib(1) → 1
    └── fib(0) → 0
```

# Common Recursion Problems

```python
# Reverse string
def reverse(s):
    if len(s) <= 1: return s
    return reverse(s[1:]) + s[0]

# Generate all subsets (backtracking)
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# Tower of Hanoi
def hanoi(n, src, dst, aux):
    if n == 1:
        print(f"Move disk 1 from {src} to {dst}")
        return
    hanoi(n-1, src, aux, dst)
    print(f"Move disk {n} from {src} to {dst}")
    hanoi(n-1, aux, dst, src)
```

## For AI Engineers
- Recursion depth matters: Python limits to ~1000 by default
- Iterative solutions are often more memory-efficient
- Backtracking is used in combinatorial search (feature selection, hyperparameter search)
