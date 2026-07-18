# Explaining Errors and Stack Traces

AI can decode cryptic error messages and stack traces into plain English.

## How to Ask

```
"Explain this error and how to fix it:"
[Paste the full stack trace here]
```

## Example

**Stack trace:**
```
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    loss = criterion(outputs, labels)
ValueError: Expected input batch_size (64) to match target batch_size (32).
```

**AI explanation:**
```
The error occurs at line 42 where the loss function receives:
- outputs with batch_size=64 (from the model forward pass)
- labels with batch_size=32 (from the dataloader)

Most likely cause: The last batch has 64 samples from the model
but only 32 labels because the DataLoader's drop_last=False and
the labels tensor was sliced differently.

Fix: Either set drop_last=True in DataLoader, or ensure model
forward and labels are computed from the same batch size.
```

## Best Practices

- Paste the **full** traceback, not just the last line
- Include relevant code context (2-3 lines around the error)
- Mention what you were doing when the error occurred
- If there's a pattern (happens intermittently), mention it
