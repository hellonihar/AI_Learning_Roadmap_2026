# Workflow Automation

Generate CI/CD pipelines, cron jobs, and task automation.

## Prompt

```
"Write a GitHub Actions workflow that:
- Triggers on push to main and pull requests
- Sets up Python 3.12
- Installs dependencies from requirements.txt
- Runs ruff linting
- Runs pytest with coverage
- Fails if coverage drops below 80%
- Caches pip dependencies"
```

## Example

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install -r requirements.txt
      - run: pip install ruff pytest pytest-cov
      - run: ruff check src/
      - run: pytest tests/ --cov=src --cov-fail-under=80
```

## Automation Prompts

- "Write a cron job script that runs a model retraining pipeline every Sunday"
- "Create a Makefile with targets: install, lint, test, train, deploy"
- "Write a pre-commit hook that runs ruff and mypy on staged files"
- "Create a shell script to set up a fresh development environment"
- "Write a scheduler script that submits batch inference jobs to AWS Batch"
