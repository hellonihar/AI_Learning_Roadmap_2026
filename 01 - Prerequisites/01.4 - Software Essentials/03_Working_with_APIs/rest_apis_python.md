# REST APIs in Python

```python
import httpx

# GET
resp = httpx.get("https://api.github.com/users/octocat")
resp.raise_for_status()
data = resp.json()
print(data["login"])

# POST with JSON body
resp = httpx.post(
    "https://api.example.com/items",
    json={"name": "Widget", "price": 9.99},
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Query parameters
resp = httpx.get(
    "https://api.example.com/search",
    params={"q": "AI", "limit": 10, "page": 1}
)

# Error handling with retry
from httpx import HTTPStatusError, RequestError

def safe_request(url, retries=3):
    for attempt in range(retries):
        try:
            resp = httpx.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            raise
        except RequestError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None
```
