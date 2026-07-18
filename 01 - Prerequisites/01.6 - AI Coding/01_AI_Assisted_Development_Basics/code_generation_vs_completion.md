# Code Generation vs Code Completion

## Code Completion

The AI predicts the next few tokens as you type, like autocomplete on steroids.

```
User types:        def calculate_mean(values):
AI suggests:          return sum(values) / len(values)
User presses Tab → accepts
```

**Best for**: boilerplate, repetitive patterns, continuing a line you started.

## Code Generation

The AI produces entire blocks of code in response to a natural language prompt.

```
User prompt:  "Write a function that downloads a file from a URL
              with retry logic and progress reporting"

AI output:
```python
def download_file(url: str, retries: int = 3) -> Path:
    ...
```

**Best for**: new functions, complete scripts, non-trivial logic.

## When to Use Which

| Scenario | Use | Reason |
|---|---|---|
| Typing a familiar loop | Completion | Faster, low risk |
| Writing a new algorithm | Generation | Need structure, not speed |
| Adding error handling | Completion | Pattern is standard |
| Designing a class hierarchy | Generation | Need architectural thinking |
| Implementing a known API | Completion | You know what you need |
| Exploring an unknown library | Generation | Need examples to learn from |
