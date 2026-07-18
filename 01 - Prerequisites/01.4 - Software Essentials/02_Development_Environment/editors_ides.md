# Editors & IDEs

## VS Code

### Essential Extensions for AI
- **Python** — IntelliSense, debugging, environments
- **Pylance** — fast type checking, autocompletion
- **Ruff** — ultra-fast linting and formatting
- **GitLens** — inline git blame, history explorer
- **Jupyter** — run notebooks inside VS Code
- **Docker** — manage containers, attach to running containers
- **GitHub Copilot** — AI pair programming
- **YAML** — YAML language support
- **Even Better TOML** — TOML support for pyproject.toml

### Settings (settings.json)
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "ruff",
    "editor.formatOnSave": true,
    "editor.renderWhitespace": "boundary",
    "files.exclude": {
        "**/.git": true,
        "**/.venv": true,
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### Workspace file for AI projects
```json
{
    "folders": [{"path": "."}],
    "settings": {
        "python.testing.pytestEnabled": true,
        "jupyter.notebookFileRoot": "${workspaceFolder}"
    }
}
```
