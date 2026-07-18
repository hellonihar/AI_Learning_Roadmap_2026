# Fixing Edge Cases

AI often generates code that works for the happy path but fails on edge cases.

## Common Edge Cases AI Misses

| Edge Case | Example |
|---|---|
| Empty input | `[]`, `""`, `None` |
| Single element | `[5]`, `"a"` |
| Duplicates | `[1, 1, 1]` |
| Boundary values | `0`, `float('inf')`, `-1` for positive-only logic |
| Non-standard types | `pandas.NA`, `np.nan`, `Decimal` |
| Special characters | Unicode, emoji, control characters in strings |
| Concurrency | Race conditions, shared state |

## Example

```python
# AI generated:
def first_or_default(items, default=None):
    return items[0] if len(items) > 0 else default
```

**Prompt:** "What edge cases does this function not handle?"

**AI:**
```
1. items is None → AttributeError
2. items is a string → returns first character, not element
3. items is a generator → len() fails
4. items is a dict → returns first key, maybe unexpected
```

**Fixed version:**
```python
from collections.abc import Sized

def first_or_default(items, default=None):
    if items is None:
        return default
    if isinstance(items, Sized) and len(items) == 0:
        return default
    try:
        return next(iter(items))
    except StopIteration:
        return default
```

## Edge Case Prompt

```
"Review this function for edge cases. Specifically check:
empty input, None, type mismatches, boundary values,
concurrent access, and single-element scenarios."
```
