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
- Retry-After header tells you when to retry
- Exponential backoff: wait 1s → 2s → 4s → 8s on 429/503
- Batch requests where possible to stay under limits
