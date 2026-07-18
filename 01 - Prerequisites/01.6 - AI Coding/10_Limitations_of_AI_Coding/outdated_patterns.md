# Outdated Patterns

AI models have knowledge cutoffs and may suggest deprecated or outdated approaches.

## Examples

```python
# ❌ Deprecated (AI may suggest this for older models):
from transformers import OpenAIGPTModel
# Current: GPT2Model, or use AutoModel

# ❌ Deprecated:
pip install sklearn
# Current: pip install scikit-learn

# ❌ Old style:
with open("file.txt", "r") as f:
    data = f.read()
# Current context managers are fine, but pathlib is preferred:
from pathlib import Path
data = Path("file.txt").read_text()
```

## Knowledge Cutoff Dates

| Model | Knowledge Cutoff |
|---|---|
| GPT-4 (older) | Sep 2021 |
| GPT-4o | Oct 2023 |
| Claude 3.5 Sonnet | Apr 2024 |
| Claude 3.5 Haiku | May 2025 |

Anything released or changed after the cutoff date will not be known to the model.

## How to Mitigate

- **Specify version** in your prompt: "Using PyTorch 2.5+ syntax"
- **Check release dates** — AI may not know about libraries released after its cutoff
- **Cross-reference with current docs** for any critical code
- **Use up-to-date models** if available (newer models have more recent knowledge)

## Prompt

```
"Use the latest stable API for [library]. It's currently [year].
If you're unsure about the latest API, provide the version you assume."
```
