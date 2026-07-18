# JSON Handling

## Parsing JSON from APIs

```python
import json
import httpx

resp = httpx.get("https://api.example.com/data")
data = resp.json()          # parse JSON response
print(data["key"])
print(data["nested"][0]["subfield"])
```

## Serializing JSON

```python
# Dict to JSON string
payload = {"name": "Alice", "scores": [85, 92, 78]}
json_str = json.dumps(payload, indent=2)

# Dict to JSON bytes (for httpx content)
json_bytes = json.dumps(payload).encode("utf-8")

# Custom serialization
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dumps({"time": datetime.now()}, cls=CustomEncoder)
```

## Working with API Responses

```python
# Safely accessing nested JSON
user = resp.json()
name = user.get("profile", {}).get("name", "unknown")

# Checking response structure
if "error" in resp.json():
    raise Exception(resp.json()["error"]["message"])

# Flatting nested JSON
def flatten_json(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

flat = flatten_json(resp.json())
# {"user_name": "Alice", "user_address_city": "NYC"}
```

## JSON with pandas

```python
import pandas as pd

# JSON array → DataFrame
df = pd.DataFrame(resp.json())   # list of dicts

# Nested JSON → normalize
df = pd.json_normalize(resp.json(), sep="_")
# Flattens nested objects into columns like "address_city"

# Save DataFrame as JSON
df.to_json("data.json", orient="records", indent=2)
```

## JSON Schema validation

```python
from pydantic import BaseModel

class APIResponse(BaseModel):
    id: int
    name: str
    score: float | None = None

data = resp.json()
validated = APIResponse(**data)  # raises on mismatch
```
