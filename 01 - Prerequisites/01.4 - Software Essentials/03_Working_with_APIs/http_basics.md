# HTTP & REST Basics

```http
# Common Methods
GET     /users          # retrieve resources
POST    /users          # create resource
PUT     /users/1        # replace resource
PATCH   /users/1        # partial update
DELETE  /users/1        # remove resource
```

## Status Codes
```
2xx Success
  200 OK
  201 Created
  204 No Content

3xx Redirection
  301 Moved Permanently
  304 Not Modified

4xx Client Error
  400 Bad Request
  401 Unauthorized
  403 Forbidden
  404 Not Found
  429 Too Many Requests (rate limited)

5xx Server Error
  500 Internal Server Error
  502 Bad Gateway
  503 Service Unavailable
```

## Headers
```
Authorization: Bearer sk-abc123
Content-Type: application/json
Accept: application/json
RateLimit-Remaining: 42
X-Request-ID: req-001
```

## Rate Limiting

### Headers (common patterns)
```
RateLimit-Limit: 100          # requests per window
RateLimit-Remaining: 42       # remaining in current window
RateLimit-Reset: 1712345678   # window reset timestamp (Unix)
Retry-After: 30               # seconds to wait (on 429)
x-ratelimit-remaining-tokens: 5000   # OpenAI specific
```

### Strategies

**Client-side throttling**
```python
import time
import httpx

class RateLimiter:
    def __init__(self, max_calls: int, window: float = 60.0):
        self.max_calls = max_calls
        self.window = window
        self.calls = []

    def wait(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c < self.window]
        if len(self.calls) >= self.max_calls:
            sleep = self.calls[0] + self.window - now
            if sleep > 0:
                time.sleep(sleep)
        self.calls.append(time.time())

limiter = RateLimiter(max_calls=50)  # 50 calls per 60s
```

**Exponential backoff (on 429 / 503)**
```python
import time

def request_with_backoff(url, max_retries=5):
    for attempt in range(max_retries):
        resp = httpx.get(url)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise Exception("Max retries exceeded")
```

**Best practices**
- Add jitter: `wait = base_wait + random.uniform(0, 1)`
- Batch independent requests instead of calling one-by-one
- Monitor remaining tokens / requests headers to pre-throttle
