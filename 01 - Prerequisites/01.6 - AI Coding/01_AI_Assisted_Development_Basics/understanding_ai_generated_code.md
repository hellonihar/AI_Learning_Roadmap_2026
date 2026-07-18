# Understanding AI-Generated Code

Never trust AI output blindly. Always read, understand, and verify before using.

## Mental Checklist

```
Does it run?         → Execute in a safe environment first
Does it do what I asked? → Check the logic, not just the output
Can I explain it?    → If you can't, neither can the AI
Are there antipatterns? → globals(), mutable defaults, bare except
Is it idiomatic?     → Does it follow Python conventions?
```

## Common Issues in AI Code

### 1. Hallucinated APIs

```python
# AI generated:
from openai import MagicClient  # ❌ doesn't exist
client = MagicClient(api_key=key)
```

Always check that imported modules exist before running.

### 2. Wrong Assumptions

```python
# AI generated:
def clean_data(df):
    df.dropna(inplace=True)  # ❌ might modify original unexpectedly
    return df
```

Check if the AI's assumptions match your data.

### 3. Missing Edge Cases

```python
def divide(a, b):
    return a / b  # ❌ no ZeroDivisionError handling
```

Always add edge cases the AI might miss.

## When to Trust

- **Boilerplate** (config loaders, file I/O patterns) — usually fine
- **Algorithms from well-known problem domains** — usually correct
- **Code with clear, testable outputs** — verify with tests
- **After you've read and understood every line** — trust but verify
