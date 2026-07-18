# Writing Clear Coding Prompts

## The Vague → Clear Pipeline

```
Vague:      "Write a function to process data"
↓
Better:     "Write a function to clean a CSV of customer data"
↓
Clear:      "Write a function `clean_customers(df: pd.DataFrame) -> pd.DataFrame`
             that: removes rows with missing email, standardizes phone numbers
             to E.164 format, and normalizes name casing."

```

## Prompt Templates

### Template 1: Function Generation
```
Write a {language} function called {name} that:
- Takes: {inputs}
- Returns: {outputs}
- Logic: {description}
- Libraries: {allowed libs}
- Handle: {edge cases}
```

### Template 2: Script Generation
```
Write a {language} script that:
- Reads {input source}
- Processes by {transformation}
- Writes to {output destination}
- Runs via {CLI, cron, etc}
- Include: error handling, logging, --help flag
```

## Examples

**Prompt:**
```
Write a Python function `chunk_text(text: str, max_tokens: int = 512) -> list[str]`
that splits a long string into chunks no longer than max_tokens using a
simple whitespace tokenizer. Preserve complete words. Log each chunk's length.
```

**AI output:**
```python
def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0
    for word in words:
        if current_count + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_count = 1
        else:
            current_chunk.append(word)
            current_count += 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```
