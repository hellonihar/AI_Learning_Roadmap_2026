# Understanding Unfamiliar Code

When you encounter code you don't understand, AI can help decipher it.

## Prompt Templates

```
"Explain what this function does at a high level"
"Walk through this code step by step"
"What design pattern is this using?"
"Why would someone write it this way?"
```

## Example

```python
def transform(features: list[float]) -> list[float]:
    return [max(0, x) for x in features]
```

**Without AI**: "Something with max(0, x)..."

**With AI prompt** "Explain this":
"Applies the ReLU activation function: replaces negative values with 0, keeps positive values unchanged. Often used in neural networks."

## Strategy

1. **Paste the code** into the AI with "Explain this code"
2. **Get a high-level summary** first ("this is a data loader")
3. **Drill into specific parts** ("explain this decorator", "what does zip do here")
4. **Ask for a diagram** (when relevant) — some AIs can generate Mermaid flowcharts
