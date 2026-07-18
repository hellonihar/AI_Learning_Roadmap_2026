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
```

## Error Handling

```python
from httpx import HTTPStatusError, RequestError, TimeoutException

def safe_api_call(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = httpx.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()

        except HTTPStatusError as e:
            # 4xx / 5xx responses
            status = e.response.status_code
            body = e.response.text
            if status == 429:
                wait = int(e.response.headers.get("Retry-After", 2 ** attempt))
                print(f"Rate limited. Retrying in {wait}s")
                time.sleep(wait)
                continue
            elif 500 <= status < 600:
                print(f"Server error {status}. Retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"Client error {status}: {body}")
                raise  # don't retry 4xx (bad request, auth, etc.)

        except TimeoutException:
            print(f"Timeout on attempt {attempt + 1}")
            time.sleep(2 ** attempt)
            continue

        except RequestError as e:
            print(f"Connection error: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    return None

# Usage
try:
    data = safe_api_call("https://api.example.com/data")
except Exception as e:
    print(f"API call failed after retries: {e}")
```

**Key rules**
- Never swallow exceptions silently — log them
- Distinguish retryable (429, 5xx) from non-retryable (400, 401, 403, 404)
- Set timeouts on every request to avoid hanging indefinitely
- Use tenacity library for production retry logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_retryable(e: Exception) -> bool:
    return isinstance(e, HTTPStatusError) and e.response.status_code in (429, 502, 503)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), retry=retry_if_exception(is_retryable))
def fetch_data(url: str) -> dict:
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()
```
