# Functions, Decorators, & More

## Basic Functions
```python
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# *args, **kwargs
def log_all(*args, **kwargs):
    for a in args: print(a)
    for k, v in kwargs.items(): print(f"{k}={v}")
```

## Lambdas
```python
square = lambda x: x**2
sorted(items, key=lambda x: x[1])
```

## Decorators
```python
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    ...
```

## Partial Functions
```python
from functools import partial
multiply = lambda a, b: a * b
double = partial(multiply, 2)
double(5)  # 10
```
