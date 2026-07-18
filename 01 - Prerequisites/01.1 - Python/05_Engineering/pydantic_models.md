# Pydantic Models

## Data Validation
```python
from pydantic import BaseModel, Field, field_validator

class InputData(BaseModel):
    age: int = Field(..., ge=0, le=150)
    name: str = Field(..., min_length=1)
    score: float = Field(default=0.0, ge=0, le=100)

    @field_validator("name")
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be blank")
        return v.strip()
```

## Settings Management
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    model_name: str = "gpt-4"
    max_tokens: int = 1024
    temperature: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()
```

## Nested Models
```python
class Metadata(BaseModel):
    source: str
    created_at: str

class Document(BaseModel):
    content: str
    metadata: Metadata
    embedding: list[float] = []
```
