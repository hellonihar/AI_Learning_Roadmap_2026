# Hallucinated APIs

AI models sometimes invent functions, methods, or packages that don't exist.

## Common Examples

```python
# ❌ AI hallucinated:
from sklearn.model_selection import train_valid_split  # doesn't exist
# Correct: train_test_split

# ❌ AI hallucinated:
response = openai.chat_with_history(messages, session_id="abc")  # doesn't exist
# Correct: client.chat.completions.create()

# ❌ AI hallucinated:
pip install pytorch-advanced-optim  # doesn't exist
```

## Why It Happens

The AI has seen similar patterns during training and combines them incorrectly. It doesn't "know" what exists — it predicts what's likely based on patterns.

## How to Protect Yourself

1. **Check imports** — Before running, verify every import exists (`pip install`, `import`, docs)
2. **Check function signatures** — Look up the actual API docs, don't trust AI's parameter names
3. **Run in a sandbox first** — Test in a notebook or isolated environment
4. **Ask the AI** — "Are you sure this function exists? Check the documentation for [library] version [x.y]"

## Prompt to Reduce Hallucinations

```
"Generate code using only documented APIs from [library] version [x.y].
Do not invent functions. If you're unsure, say so."
```
