# Async / Await Basics

## Async Functions
```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

async def main():
    result = await fetch_data("https://api.example.com")
    print(result)

asyncio.run(main())
```

## Concurrent Calls
```python
async def main():
    urls = ["url1", "url2", "url3"]
    tasks = [fetch_data(u) for u in urls]
    results = await asyncio.gather(*tasks)
    return results
```

## Async for AI
```python
# Useful for parallel LLM calls, batch inference, etc.
async def batch_prompts(prompts: List[str]):
    async def call(p):
        return await llm.ainvoke(p)

    tasks = [call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```
