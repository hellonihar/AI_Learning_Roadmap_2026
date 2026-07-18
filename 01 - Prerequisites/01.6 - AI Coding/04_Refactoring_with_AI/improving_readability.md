# Improving Readability

AI can rename variables, add types, extract functions, and apply consistent formatting.

## Variable Renaming

```python
# Before:
def proc(d):
    r = []
    for i in d:
        if i[1] > 0:
            r.append((i[0], i[1] * 2))
    return r
```

**Prompt:** "Rename these variables to be descriptive, add type hints"

```python
# After:
def process_items(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    result = []
    for item in items:
        name, value = item
        if value > 0:
            result.append((name, value * 2))
    return result
```

## Extract Magic Numbers

```python
# Before:
def calculate(data):
    return [x * 0.01 for x in data if x > 100]
```

**Prompt:** "Extract magic numbers into named constants"

```python
# After:
TAX_RATE = 0.01
TAX_THRESHOLD = 100

def calculate(data):
    return [x * TAX_RATE for x in data if x > TAX_THRESHOLD]
```

## Readability Prompts

```
- "Add type hints to this function"
- "Rename variables to follow PEP 8"
- "Extract this block into a named function"
- "Add docstrings explaining parameters and return values"
- "Use early returns to reduce nesting"
- "Convert this to use match/case (Python 3.10+)"
```
