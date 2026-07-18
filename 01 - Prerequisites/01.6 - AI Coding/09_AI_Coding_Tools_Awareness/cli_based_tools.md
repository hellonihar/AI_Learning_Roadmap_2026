# CLI-Based AI Tools

AI coding assistants that work from the terminal.

## Claude CLI (Anthropic)

Run Claude directly from your terminal for coding tasks.

```bash
claude "Write a FastAPI app that serves a PyTorch model"
claude "Debug this error" < error.log
claude "Explain this codebase" -d ./src
```

**Strengths**: Can read/write files, run commands, interact with git. Good for script generation and file-level operations.

## GitHub CLI + Copilot

```bash
gh copilot suggest "Write a function to merge two sorted lists"
gh copilot explain "This error means the port is already in use"
```

**Strengths**: Tight integration with GitHub workflows.

## Terminal AI Wrappers

Tools like Shell GPT (`sgpt`), `aichat`, and `mods` let you ask questions from the terminal.

```bash
sgpt "Find all Python files that import pandas and have more than 100 lines"
```

## When to Use CLI

- You're already in the terminal (most natural flow)
- You need AI to process command output
- You want to automate AI calls in scripts
- You prefer keyboard-only workflows
