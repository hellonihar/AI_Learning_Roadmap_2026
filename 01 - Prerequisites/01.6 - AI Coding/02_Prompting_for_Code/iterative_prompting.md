# Iterative Prompting

Getting perfect code on the first try is rare. Iterative prompting — refining your request based on the AI's output — is the real skill.

## Iteration Loop

```
Round 1: Write the basic function
→ Review output

Round 2: "Add error handling"
→ Review output

Round 3: "Make it async"
→ Review output

Round 4: "Add type hints and docstrings"
→ Final output
```

## Example

**Round 1** — Basic version:
```
Prompt:  "Write a function that fetches a URL and returns JSON"
Output:  Simple requests.get() with no error handling
```

**Round 2** — Add robustness:
```
Prompt:  "Add retry logic with exponential backoff for 5xx errors"
Output:  Retry wrapper added
```

**Round 3** — Add performance:
```
Prompt:  "Make it async and support batch URL fetching"
Output:  Async version with asyncio.gather
```

**Round 4** — Polish:
```
Prompt:  "Add type hints, docstrings, and a timeout parameter"
Output:  Production-ready function
```

## Common Iteration Prompts

```
- "Add input validation"
- "Make this more memory efficient"
- "Refactor into smaller functions"
- "Add logging at each step"
- "Convert to use dataclasses"
- "Add a progress bar using tqdm"
- "Make the config load from environment variables"
- "Add pytest tests for edge cases"
```
