# AI Coding Quick Reference

## Prompt Patterns

```
Generate → "Write a Python function that..."
Explain  → "Explain this code line by line..."
Refactor → "Refactor this to use async/await..."
Debug    → "This code throws ValueError. Why?"
Test     → "Write pytest tests for this function..."
```

## Provide Context

```python
# Bad prompt:
"Write a function to load data"

# Good prompt:
"Write a Python function that loads a CSV from an S3 path,
returns a pandas DataFrame, handles missing values by dropping
rows where 'target' is NaN, and logs the row count."
```

## Validate AI Output

- [ ] Does the code run without errors?
- [ ] Are the function signatures correct?
- [ ] Do imported modules/packages exist?
- [ ] Are there hardcoded secrets or credentials?
- [ ] Does it handle edge cases (empty input, None, zero)?
- [ ] Are there security issues (eval(), shell injection)?
- [ ] Are the types consistent with the rest of the codebase?

## Common Red Flags

```
AI suggests:
  result = eval(user_input)          ❌ security risk
  pip install nonexistent-package    ❌ hallucinated
  os.system("rm -rf " + path)       ❌ path injection
  import deprecated_lib              ❌ outdated pattern
  model.predict() without .fit()     ❌ logical error
```
