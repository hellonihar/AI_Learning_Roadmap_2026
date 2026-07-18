# Identifying Logical Errors

AI code can look correct but have subtle bugs.

## Common Logical Errors AI Makes

### 1. Off-by-One
```python
def split_data(data, split=0.8):
    n = int(len(data) * split)
    return data[:n], data[n:]  # ✅ correct
```

```python
def split_data(data, split=0.8):
    n = int(len(data) * split)
    return data[:n], data[n+1:]  # ❌ loses one element
```

### 2. Wrong Default Mutable
```python
def process(items=[]):   # ❌ shared mutable default
    items.append("x")
    return items
```

### 3. Early Exit / Missing Return
```python
def find_user(users, id):
    for u in users:
        if u.id == id:
            return u     # ✅ correct
```

```python
def find_user(users, id):
    for u in users:
        if u.id == id:
            return u
        else:
            return None  # ❌ returns None on first non-match
```

### 4. Wrong Comparison
```python
if user.id = 42:     # ❌ assignment instead of ==
```

## Check Before Trusting

- Trace through with a simple test case mentally
- Run with small, known inputs and verify outputs
- Ask the AI: "What happens if the input is empty?"
- Look for the patterns above — they're AI's most common mistakes
