# Testing AI-Generated Code

AI can write tests for your code, and you should test AI-generated code before using it.

## Ask AI to Write Tests

```
Prompt: "Write pytest tests for this function covering:
- Normal cases
- Edge cases (empty input, None, zero)
- Error cases (invalid types, out-of-range values)
Use parametrize and fixtures where appropriate."
```

## Test-Driven Prompting

Write the test first, then ask AI to make it pass:

```
Prompt: "Here's a test. Write the function that passes it:"
```

```python
def test_chunk_text():
    assert chunk_text("hello world", max_tokens=1) == ["hello", "world"]
    assert chunk_text("a b c", max_tokens=2) == ["a b", "c"]
    assert chunk_text("", max_tokens=10) == []
```

This produces more reliable code because the AI has a concrete target to hit.

## What to Test in AI Code

| Concern | Test |
|---|---|
| Correctness | Known input → known output |
| Edge cases | Empty, None, single item, max values |
| Error handling | Raises appropriate exceptions |
| Performance | Completes within time limit |
| Determinism | Same input = same output (for non-random functions) |
