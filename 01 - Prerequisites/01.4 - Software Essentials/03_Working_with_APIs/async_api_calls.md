# Async API Calls

```python
import asyncio
import httpx

async def fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

async def main():
    result = await fetch("https://api.github.com/users/octocat")
    print(result["login"])

asyncio.run(main())
```

## Batch Concurrent Calls
```python
async def batch_call(prompts: list[str], model: str = "gpt-4"):
    async def call(prompt: str):
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    tasks = [call(p) for p in prompts]
    return await asyncio.gather(*tasks)

results = asyncio.run(batch_call(["Q1?", "Q2?", "Q3?"]))
```

## Semaphore (Rate Limiting)
```python
sem = asyncio.Semaphore(10)  # max 10 concurrent

async def rate_limited_call(prompt):
    async with sem:
        return await call_llm(prompt)

async def batch_with_limit(prompts):
    tasks = [rate_limited_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```
