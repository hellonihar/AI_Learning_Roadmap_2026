# Validating Outputs

Always verify AI output with concrete checks.

## The 5-Question Validation

| Question | How |
|---|---|
| Does it run? | Execute in a sandbox or test environment |
| Does it produce correct output? | Test with known inputs |
| Does it handle edge cases? | Test with empty, None, zero, extreme values |
| Is it efficient enough? | Profile with realistic data sizes |
| Is it idiomatic? | Compare with project conventions |

## Automated Validation

```python
# Ask AI to write tests alongside the code
prompt = "Write a function `parse_date(s: str) -> datetime` + pytest tests"
```

## Example Validation Flow

```python
# AI generated:
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

# Validate:
assert average([1, 2, 3]) == 2.0              # ✅ basic
assert average([5]) == 5.0                      # ✅ single element
assert average([]) == 0                         # ❌ ZeroDivisionError!
# Fix: handle empty case
```

## Tools for Validation

| Tool | Use |
|---|---|
| pytest | Unit test AI-generated code |
| mypy | Type-check AI-generated code |
| ruff / pylint | Lint for code quality issues |
| bandit | Scan for security issues |
| Hypothesis | Property-based testing for edge cases |
