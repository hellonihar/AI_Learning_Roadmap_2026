# Authentication Mechanisms

## API Key (most common for AI APIs)

```python
# Header-based
headers = {"Authorization": "Bearer sk-abc123..."}
resp = httpx.get("https://api.openai.com/v1/models", headers=headers)

# Query param-based (less common)
resp = httpx.get("https://api.example.com/data?api_key=abc123")

# Always use environment variables, never hardcode
import os
api_key = os.environ["OPENAI_API_KEY"]
```

## Bearer Token (JWT)

```python
# Obtain token
resp = httpx.post("https://auth.example.com/token", json={
    "client_id": "...",
    "client_secret": "...",
    "grant_type": "client_credentials"
})
token = resp.json()["access_token"]  # JWT string

# Use token
headers = {"Authorization": f"Bearer {token}"}
resp = httpx.get("https://api.example.com/protected", headers=headers)

# JWT decode (inspect without verifying)
import jwt
decoded = jwt.decode(token, options={"verify_signature": False})
print(decoded["exp"], decoded["sub"])
```

## Basic Auth

```python
from httpx import BasicAuth
resp = httpx.get("https://api.example.com/basic",
    auth=BasicAuth("username", "password"))
```

## OAuth2 Flow (for user-context AI apps)

```
1. Redirect user to auth provider
2. User grants permission
3. Receive authorization code
4. Exchange code for access token
5. Use access token for API calls
```

## Managing Credentials Securely

```bash
# .env file (never commit)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# .gitignore
.env
*.key
credentials.json
```

```python
# python-dotenv
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# Or pydantic-settings (recommended)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model: str = "gpt-4"

    class Config:
        env_file = ".env"

settings = Settings()
```
