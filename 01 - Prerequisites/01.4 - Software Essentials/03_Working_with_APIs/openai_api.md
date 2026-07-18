# OpenAI API

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

# Chat Completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."},
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)
```

## Streaming
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Token Counting
```python
response.usage.prompt_tokens       # tokens in input
response.usage.completion_tokens    # tokens in output
response.usage.total_tokens         # total
```

## Retry with Exponential Backoff
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def call_openai(messages):
    return client.chat.completions.create(
        model="gpt-4", messages=messages
    )
```

## Key Headers
- Rate limits are per organization, per model tier
- Monitor `x-ratelimit-remaining-requests` and `x-ratelimit-remaining-tokens`
