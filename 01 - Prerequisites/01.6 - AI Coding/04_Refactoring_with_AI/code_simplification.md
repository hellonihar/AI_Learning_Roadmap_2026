# Code Simplification

AI excels at transforming verbose, nested code into clean, idiomatic code.

## Nested → Flat

```python
# Before (verbose):
def get_active_users(users):
    result = []
    for user in users:
        if user.get("active"):
            result.append(user)
    return result
```

**Prompt:** "Simplify this using a list comprehension"

```python
# After (simplified):
def get_active_users(users):
    return [u for u in users if u.get("active")]
```

## Multiple Conditions → Single Expression

```python
# Before:
if x is not None:
    if x > 0:
        result = x * 2
    else:
        result = 0
else:
    result = 0
```

**Prompt:** "Simplify this using a single expression with defaults"

```python
# After:
result = (x * 2) if (x is not None and x > 0) else 0
# Or:
result = max(x, 0) * 2 if x is not None else 0
```

## Common Simplification Prompts

```
- "Rewrite this using a list/dict comprehension"
- "Replace this loop with map/filter"
- "Simplify these nested ifs"
- "Use a dictionary dispatch instead of if-elif chains"
- "Replace this with any()/all()"
```
