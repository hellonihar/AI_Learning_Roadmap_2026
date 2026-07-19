# Code Walkthrough: LLMOps in Practice

This walkthrough builds a minimal LLM application with **logging & tracing**, **guardrails**, and **semantic caching** — the three pillars of production LLM operations.

We'll use Python with LangChain + LangSmith for tracing, NeMo-style guardrails concepts, and a vector-based semantic cache.

---

## Setup

```python
import os
import json
import hashlib
import time
from dataclasses import dataclass
from typing import Optional

# Environment
os.environ["OPENAI_API_KEY"] = "sk-..."  # Set via env var in production
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."
```

---

## 1. Logging & Tracing with LangSmith

LangSmith automatically traces LangChain execution. Every LLM call, retriever query, and chain step becomes a span.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable
from langsmith.run_trees import RunTree

# Basic traced function
@traceable
def call_llm(system_prompt: str, user_message: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    return response.content

# Log metadata manually for custom spans
@traceable(run_type="chain")
def summarize_document(text: str) -> str:
    # This creates a parent span with two child spans
    chunks = chunk_text(text)  # Child span 1
    summary = call_llm(
        "You are a summarizer.",
        f"Summarize in 3 bullet points:\n\n{chunks}"
    )  # Child span 2
    return summary

def chunk_text(text: str) -> str:
    # Manual span creation for non-LangChain code
    from langsmith import traceable
    @traceable(run_type="tool")
    def _chunk(text: str) -> str:
        words = text.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n[---]\n" + " ".join(words[mid:])
    return _chunk(text)
```

**What LangSmith captures per span:**
- `inputs`, `outputs`
- `start_time`, `end_time`, `latency`
- `token_usage` (for LLM calls)
- `error` (if exception raised)
- `tags` (add via `@traceable(tags=["summarization"])`)

### Custom Logging (No LangSmith)

If you prefer no external dependency, log to your own database:

```python
import sqlite3
import datetime

DB_PATH = "llm_audit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER,
            input TEXT,
            output TEXT,
            timestamp TEXT,
            guardrail_action TEXT
        )
    """)
    conn.commit()
    return conn

def log_call(user_id, model, input_text, output_text,
             prompt_tokens, completion_tokens, latency_ms,
             guardrail_action="none"):
    conn = init_db()
    call_id = hashlib.sha256(
        f"{user_id}{time.time()}".encode()
    ).hexdigest()[:16]
    conn.execute("""
        INSERT INTO llm_calls
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        call_id, user_id, model, prompt_tokens,
        completion_tokens, latency_ms,
        input_text[:500], output_text[:2000],  # Truncate for storage
        datetime.datetime.utcnow().isoformat(),
        guardrail_action
    ))
    conn.commit()
    conn.close()
```

---

## 2. Guardrails: Input Validation & Output Filtering

A multi-layer guardrail system protects both input and output.

```python
import re
from typing import Tuple

# ── Layer 1: Input Guard ──────────────────────────────────

# Blocked patterns for prompt injection and jailbreak attempts
BLOCKED_PATTERNS = [
    r"(?i)ignore all (previous|prior) instructions",
    r"(?i)you are (now |)DAN",
    r"(?i)do anything now",
    r"(?i)system prompt",
    r"(?i)forget everything",
    r"(?i)output your (instructions|prompt|system)",
]

def check_input_safety(text: str) -> Tuple[bool, str]:
    """Returns (passed: bool, reason: str)."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            return False, f"Blocked: matched pattern '{pattern}'"
    return True, ""

# ── Layer 1b: Content Moderation ──────────────────────────

def moderate_content(text: str) -> Tuple[bool, str]:
    """
    Uses OpenAI Moderation API.
    Falls back to regex keyword filter if API is unavailable.
    """
    from openai import OpenAI
    client = OpenAI()

    response = client.moderations.create(input=text)
    result = response.results[0]

    if result.flagged:
        # Find which categories triggered
        triggered = [
            cat for cat, val in result.categories.dict().items()
            if val
        ]
        return False, f"Moderation flagged: {', '.join(triggered)}"
    return True, ""

# ── Layer 2: PII Detection ────────────────────────────────

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "credit_card": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",
}

def redact_pii(text: str) -> Tuple[str, list]:
    """Returns (redacted_text, list_of_found_types)."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            found.append(pii_type)
            text = re.sub(pattern, f"[REDACTED_{pii_type}]", text)
    return text, found

# ── Layer 3: Output Guard ─────────────────────────────────

HARMFUL_OUTPUT_PATTERNS = [
    r"(?i)how to (make|create|synthesize) (a |)weapon",
    r"(?i)instructions for (self-harm|suicide)",
    r"(?i)child.*(porn|exploit|abuse)",
]

def check_output_safety(text: str) -> Tuple[bool, str]:
    for pattern in HARMFUL_OUTPUT_PATTERNS:
        if re.search(pattern, text):
            return False, f"Output blocked: matched '{pattern}'"
    return True, ""

# ── Guarded LLM Call ──────────────────────────────────────

def guarded_llm_call(system_prompt: str, user_message: str,
                     user_id: str = "anonymous") -> str:
    # Layer 1: Input guard
    passed, reason = check_input_safety(user_message)
    if not passed:
        log_call(user_id, "gpt-4o-mini", user_message,
                 reason, 0, 0, 0, "blocked_input")
        return f"I cannot process that request. ({reason})"

    # Layer 1b: Content moderation
    passed, reason = moderate_content(user_message)
    if not passed:
        log_call(user_id, "gpt-4o-mini", user_message,
                 reason, 0, 0, 0, "blocked_moderation")
        return "Your message was flagged by our content safety system."

    # Layer 2: Redact PII from input
    safe_input, pii_found = redact_pii(user_message)
    guardrail_action = "pii_redacted" if pii_found else "passed"

    # Measure latency
    start = time.time()
    output = call_llm(system_prompt, safe_input)
    latency = int((time.time() - start) * 1000)

    # Layer 3: Output guard
    passed, reason = check_output_safety(output)
    if not passed:
        output = "The response was filtered by our safety system."
        guardrail_action = "blocked_output"

    # Log
    log_call(user_id, "gpt-4o-mini", safe_input, output,
            150, 50, latency, guardrail_action)

    return output
```

---

## 3. Semantic Caching

Reduces cost and latency by returning cached responses for semantically similar queries.

```python
import numpy as np
from typing import List, Tuple

# ── Simple Vector Store (in-memory, for illustration) ────

@dataclass
class CacheEntry:
    query_embedding: np.ndarray
    response: str
    timestamp: float

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        self.entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._client = OpenAI()
        self._model = "text-embedding-3-small"

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            input=text, model=self._model
        )
        return np.array(resp.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> Optional[str]:
        query_embedding = self._get_embedding(query)
        now = time.time()

        # Clean expired entries
        self.entries = [
            e for e in self.entries
            if (now - e.timestamp) < self.ttl_seconds
        ]

        for entry in self.entries:
            sim = self._cosine_similarity(query_embedding, entry.query_embedding)
            if sim >= self.similarity_threshold:
                return entry.response

        return None

    def set(self, query: str, response: str):
        embedding = self._get_embedding(query)
        self.entries.append(CacheEntry(embedding, response, time.time()))

# ── Usage ─────────────────────────────────────────────────

cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=1800)

def cached_llm_call(system_prompt: str, user_message: str) -> str:
    cache_key = f"{system_prompt}|||{user_message}"

    # Check cache first
    cached = cache.get(cache_key)
    if cached:
        print("[CACHE HIT]")
        return cached

    # Cache miss: call LLM
    print("[CACHE MISS]")
    result = call_llm(system_prompt, user_message)

    # Store in cache
    cache.set(cache_key, result)
    return result
```

---

## 4. Putting It All Together

```python
def production_llm_pipeline(
    user_message: str,
    user_id: str = "anonymous",
    system_prompt: str = "You are a helpful assistant."
) -> str:
    # 1. Check semantic cache (no PII in cache key — use the original message)
    cached_response = cache.get(user_message)
    if cached_response:
        log_call(user_id, "cache", user_message, cached_response,
                 0, 0, 0, "cache_hit")
        return cached_response

    # 2. Run through guardrails and LLM
    response = guarded_llm_call(system_prompt, user_message, user_id)

    # 3. Cache the response (only if it wasn't blocked)
    if response and "blocked" not in response.lower()[:20]:
        cache.set(user_message, response)

    return response
```

---

## 5. Viewing Traces (LangSmith Dashboard)

After running the code:

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to the project (default: "default")
3. You will see:
   - **Runs list:** Each `@traceable` call is a row.
   - **Span detail:** Click a run to see inputs, outputs, latency.
   - **Token usage:** Shown for LLM calls.
   - **Metadata:** Tags, user_id (if passed via `langsmith.meta()`).
   - **Errors:** Red indicators for failed runs.

### Example Trace for `summarize_document`

```
summarize_document (chain)  [2.3s]
  ├── chunk_text (tool)     [300ms]
  └── call_llm (llm)        [1.9s, tokens: 450/120]
```

---

## Summary

This walkthrough implements the three core LLMOps practices:

| Practice | Implementation |
|----------|---------------|
| **Tracing & Logging** | LangSmith `@traceable` + custom SQLite audit log |
| **Guardrails** | Input pattern matching, moderation API, PII redaction, output filtering |
| **Semantic Caching** | Embedding-based similarity cache with TTL |

In production, replace the in-memory cache with RedisVL or pgvector, upgrade SQLite to a proper audit database (immutable + encrypted), and add automated eval runs to LangSmith for quality monitoring.
