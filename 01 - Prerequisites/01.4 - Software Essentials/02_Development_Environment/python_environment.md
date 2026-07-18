# Python Environment Management

```bash
# pyenv (manage Python versions)
pyenv install 3.12.3
pyenv global 3.12.3
pyenv local 3.12.3   # creates .python-version

# venv (built-in virtual environments)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows
deactivate

# uv (fast modern pip replacement)
pip install uv
uv venv
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# requirements.txt
"""
openai>=1.0
torch>=2.0
transformers>=4.30
pandas>=2.0
httpx>=0.25
pydantic>=2.0
"""

# pyproject.toml (modern standard)
"""
[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "torch>=2.0",
]
"""
```
