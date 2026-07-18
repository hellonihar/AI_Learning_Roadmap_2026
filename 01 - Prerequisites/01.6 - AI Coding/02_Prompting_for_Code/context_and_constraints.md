# Providing Context and Constraints

The more context you give, the better the AI's output.

## Types of Context

### 1. Code Context
```python
# Current file:
from typing import List
import pandas as pd

# Prompt:
"Add a function that validates all columns in this DataFrame
have no null values. Return a list of column names with nulls."
```

### 2. Project Context
```
"This is a FastAPI project using Pydantic v2, SQLAlchemy 2.0,
and PostgreSQL. Write a new endpoint that..."
```

### 3. Style Constraints
```
"Use Google-style docstrings. Prefer comprehensions over loops.
Use pathlib over os.path. Maximum line length 88."
```

## Constraint Examples

| Constraint | Prompt Addition |
|---|---|
| Performance | "Must handle 10M rows in under 5 seconds" |
| Compatibility | "Must work with Python 3.10+" |
| Dependencies | "Use only stdlib, no external packages" |
| Concurrency | "Use asyncio and aiohttp for concurrent requests" |
| Testing | "Include pytest tests with fixtures and parametrize" |
| Security | "Sanitize all user inputs, no eval(), no shell=True" |
| Deployment | "Must run in a Docker container with Alpine" |
