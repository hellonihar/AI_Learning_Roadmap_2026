# Prompting for Code

Effective prompts make the difference between useless output and production-ready code.

## Anatomy of a Good Prompt

```
[Task]     Write a Python function that...
[Context]  ...takes a list of URLs, downloads each...
[Format]   ...returns a list of file paths...
[Edge]     ...handles timeouts and retries failed URLs up to 3 times.
[Style]    Use httpx, async/await, and type hints.
```

## Bad vs Good Examples

**Bad**: "Write a function to load data"
**Good**: "Write a function `load_data(path: str) -> pd.DataFrame` that reads a CSV from a local path or S3 URI. If S3, use boto3. Handle missing values by dropping rows where 'id' is NaN. Log the row count after loading."

**Bad**: "Sort a list"
**Good**: "Write a merge sort implementation in Python that sorts a list of dicts by the 'priority' key descending. Use recursion, add type hints, and include a test case."

## Tips

- **Be specific about inputs and outputs** — the AI needs to know what goes in and what comes out
- **Mention libraries** — "use pandas", "use FastAPI", "use Polars instead of pandas"
- **Specify constraints** — "must run in <100ms", "must handle 10K concurrent requests", "must work offline"
- **Provide examples** — "Input: [1,2,3], Output: [3,2,1]" when asking for a shuffle function
- **Give error context** — if debugging, paste the full stack trace, not just the error message
