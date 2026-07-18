# Memory in AI Systems

## 1. Introduction to Memory in AI Systems

### Stateless Nature of LLMs

Every LLM call is an independent prediction. The model has no recollection of previous calls, no persistent state, and no built-in mechanism to carry information across interactions.

```
Call 1: "My name is Alice."
  → Model: "Nice to meet you, Alice!"

Call 2: "What is my name?"
  → Model: "I don't have that information."
  → The second call has no knowledge of the first. The model is stateless.
```

**Statelessness is by design.** It simplifies the model architecture (no hidden state to maintain across calls), enables horizontal scaling (any instance can handle any request), and keeps the API stateless (no server-side memory management). The burden of memory shifts entirely to the application layer.

### Memory vs Context

| Concept | What It Is | Characteristics |
|---|---|---|
| **Context** | Everything in the current input token window | Visible to the model right now. Lost when the call ends. |
| **Memory** | Information persisted across calls | Stored externally. Retrieved and injected into context when needed. |

```
Context = what the model can see right now
Memory = what your application remembers between calls
```

**Analogy:** Context is the text on the page you're currently reading. Memory is the bookmark, the notes you took, and the library you can pull books from.

The application's job is to decide: **what should be remembered, and when should it be injected into context?**

### Short-Term vs Long-Term Memory

| Dimension | Short-Term Memory | Long-Term Memory |
|---|---|---|
| **Duration** | Minutes to hours (session) | Days to years (persistent) |
| **Storage** | In-memory, Redis (volatile) | Database, vector store (persistent) |
| **Capacity** | Limited (token budget) | Unlimited (external storage) |
| **Retrieval** | Always loaded (conversation buffer) | On-demand (search or lookup) |
| **Persistence** | Lost when session ends | Survives restarts and logouts |
| **Example** | Current conversation turns | User preferences, historical purchases |

**Real example — customer support bot:**
```
Short-term: Last 5 messages in the current conversation
  → "I already told you my order number is ORD-12345"
  → Prevents the user from repeating themselves

Long-term: User's account tier, past issues, preferences
  → "This user is a premium member with a history of shipping complaints"
  → Enables personalized, informed responses across sessions
```

### Why Applications Need Memory

| Without Memory | With Memory |
|---|---|
| User repeats information every turn | Model recalls previous context |
| No personalization per user | Adapts to user preferences over time |
| Cannot track progress in multi-step tasks | Resumes tasks from last completed step |
| Every session starts from zero | Builds on past interactions |
| Cannot learn from user corrections | Adjusts behavior based on feedback |

**The stateless LLM is a component.** The application provides memory to make it appear stateful to the user.

---

## 2. Types of Memory

### A. Short-Term Memory

Memory that persists within a single session or conversation.

#### Conversation Buffer

The raw log of recent messages, injected into every call.

```python
conversation_buffer = [
    {"role": "user", "content": "Hi, I need help with my order."},
    {"role": "assistant", "content": "Sure, what's your order number?"},
    {"role": "user", "content": "ORD-12345"},
]

# Inject into every call
messages = [
    {"role": "system", "content": "You are a support agent."},
    *conversation_buffer,  # Full history
    {"role": "user", "content": "Can you check the status?"}
]
```

**Pros:** Simple, lossless, the model sees every detail.
**Cons:** Grows unboundedly, consumes token budget, expensive for long conversations.

#### Sliding Window

Keep only the N most recent messages. Drop older ones.

```python
MAX_TURNS = 10  # Keep last 5 user + 5 assistant messages

def get_window(buffer: list, max_turns: int = MAX_TURNS) -> list:
    return buffer[-max_turns:]

# Oldest messages are dropped when window is exceeded
```

**Pros:** Bounded token usage, predictable cost.
**Cons:** Loses old information. User may need to repeat context.

#### Session-Level State

Variables that persist for the duration of a session but are not conversation messages.

```python
session_state = {
    "user_id": "usr_456",
    "authentication_status": "verified",
    "current_page": "order_status",
    "pending_action": None,
    "error_count": 0,
    "started_at": "2026-07-18T14:30:00Z"
}
```

**Used for:** Tracking progress through a workflow, storing intermediate form data, managing multi-step authentication.

### B. Long-Term Memory

Memory that persists across sessions, restarts, and logouts.

#### Persistent User Data

Data stored in a database, keyed by user ID, loaded at the start of each session.

```python
user_profile = db.query("SELECT * FROM user_profiles WHERE user_id = ?", user_id)
# → {"name": "Alice", "tier": "premium", "preferred_language": "en", "timezone": "UTC+1"}
```

**Load once per session.** Injected into system prompt as context.

#### Historical Interactions

Full or summarized history of past sessions, available for search.

```python
past_interactions = vector_db.search(
    embed("user previous issues with shipping"),
    filter={"user_id": user_id}
)
# → "In March, user reported a damaged package. Resolution: full refund."
```

**Best practice:** Store per-session summaries in a vector DB. On new session, retrieve relevant past interactions.

#### User Preferences

Explicit preferences collected from user actions or stated preferences.

```python
user_preferences = {
    "communication_style": "concise",
    "topics_of_interest": ["machine learning", "python", "data engineering"],
    "notification_preference": "email",
    "accessibility_requirements": None
}
```

**Used for:** Personalizing tone, content, and delivery format.

#### Knowledge Accumulation

Information the system learns about the user over time.

```
Session 1: User mentions they have a dog.
Session 2: User asks about dog-friendly hotels.
Session 3: User mentions their dog is a golden retriever named Max.
  → Knowledge: user has a golden retriever named Max
  → Future responses can reference Max by name
```

**Implementation:** Extract facts from conversations using an LLM, store in a structured knowledge base, retrieve relevant facts before each response.

### C. Working / Execution Memory

Memory needed within a single task or agent execution.

#### Intermediate Outputs

Results from each step in a multi-step workflow.

```python
execution_state = {
    "step": "research_complete",
    "intermediate_results": [
        {"step": "classify_intent", "output": "refund_request"},
        {"step": "extract_order_id", "output": "ORD-12345"},
        {"step": "lookup_order", "output": {"status": "delivered", "date": "2026-07-01"}},
    ],
    "final_output": None
}
```

#### Tool Results

Outputs from tool calls that may be needed later in the same workflow.

```python
tool_results = {
    "get_weather": {"temperature": 22, "condition": "sunny"},
    "search_flights": [...]  # Too large, needs summarization
}
```

#### Task Progress Tracking

The current state of a multi-step task.

```python
task = {
    "id": "task_789",
    "type": "vacation_planning",
    "status": "in_progress",
    "completed_steps": ["destination_selected", "flights_booked"],
    "next_step": "hotel_search",
    "user_inputs": {
        "destination": "Paris",
        "budget": "moderate",
        "dates": "2026-09-15 to 2026-09-22"
    }
}
```

#### Agent State Variables

Variables an agent uses to make decisions during execution.

```python
agent_state = {
    "iteration_count": 0,
    "max_iterations": 10,
    "tools_used": [],
    "last_action": "search_knowledge_base",
    "last_observation": "Found 3 relevant documents",
    "confidence": 0.85,
    "should_continue": True
}
```

---

## 3. Memory Storage Mechanisms

| Storage | Persistence | Latency | Query Type | Best For |
|---|---|---|---|---|
| **In-memory** | Volatile (session) | < 1ms | Key lookup | Conversation buffer, session state |
| **File-based** | Persistent (disk) | 1-10ms | Read/write entire file | Simple config, prototyping |
| **Relational DB** | Persistent (disk) | 1-50ms | SQL queries, joins | User profiles, structured preferences |
| **Vector DB** | Persistent (disk) | 10-100ms | Similarity search | Semantic memory retrieval |
| **KV store** | Persistent (RAM/disk) | < 5ms | Key lookup | Caching, session state at scale |
| **Document store** | Persistent (disk) | 10-100ms | JSON queries, full-text | Complex nested memory structures |

### In-Memory Storage

```python
# Simple dict for session-level memory (single process)
session_memory: dict[str, list] = {}

def add_to_buffer(session_id: str, message: dict):
    if session_id not in session_memory:
        session_memory[session_id] = []
    session_memory[session_id].append(message)
```

**Limitation:** Lost on server restart. Not shared across multiple server instances.

### File-Based Storage

```python
import json

def save_memory(user_id: str, data: dict):
    path = f"memory/{user_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)

def load_memory(user_id: str) -> dict:
    path = f"memory/{user_id}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
```

**Best for:** Prototyping, single-user applications, simple preferences.

### Relational Databases

```python
CREATE TABLE conversation_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conversation_user ON conversation_history(user_id, created_at);
```

```python
def get_recent_messages(user_id: str, limit: int = 20):
    return db.query(
        "SELECT role, content FROM conversation_history "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        user_id, limit
    )
```

### Vector Databases

```python
def store_memory_vector(user_id: str, text: str, metadata: dict):
    embedding = embed(text)
    vector_db.upsert([{
        "id": f"mem_{uuid4()}",
        "values": embedding,
        "metadata": {
            "user_id": user_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
    }])

def retrieve_relevant_memories(user_id: str, query: str, k: int = 5):
    query_embedding = embed(query)
    return vector_db.search(
        query_embedding,
        filter={"user_id": user_id},
        top_k=k
    )
```

### Key-Value Stores

```python
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

# Session state with TTL
r.setex(f"session:{session_id}:state", 3600, json.dumps(session_state))

# Long-term preferences
r.set(f"user:{user_id}:preferences", json.dumps(user_preferences))
```

### Document Stores

```python
# MongoDB-like document for complex memory
{
    "user_id": "usr_456",
    "sessions": [
        {
            "session_id": "sess_001",
            "date": "2026-07-18",
            "summary": "User inquired about refund policy",
            "sentiment": "frustrated",
            "resolution": "refund_processed"
        }
    ],
    "extracted_facts": [
        {"fact": "User prefers email communication", "confidence": 0.95, "source": "session_001"},
        {"fact": "User has a golden retriever named Max", "confidence": 0.90, "source": "session_003"}
    ]
}
```

---

## 4. Memory Retrieval Strategies

### Recency-Based Retrieval

The simplest strategy: always retrieve the most recent items.

```python
def get_recent_context(session_id: str, n: int = 5) -> list:
    """Retrieve the last N messages from the conversation buffer."""
    return session_buffer[session_id][-n:]
```

**When to use:** Short-term context where recent information is most relevant.

**Limitation:** Doesn't distinguish relevance — old but important information is lost.

### Similarity-Based Retrieval

Retrieve memories semantically similar to the current context.

```python
def retrieve_similar_memories(query: str, user_id: str, k: int = 3) -> list:
    """Find the most semantically relevant past interactions."""
    query_embedding = embed(query)
    results = memory_vector_db.search(query_embedding, filter={"user_id": user_id}, top_k=k)
    return [r.metadata["text"] for r in results]

# Example:
# Query: "What did I say about my shipping preferences?"
# Retrieved: "User mentioned they prefer UPS over FedEx due to past delivery issues."
```

**When to use:** Long-term memory where relevance matters more than recency.

### Metadata Filtering

Filter memories by structured metadata before applying similarity or recency.

```python
def retrieve_filtered_memories(
    user_id: str,
    query: str,
    memory_type: str = None,
    date_range: tuple = None,
    k: int = 5
) -> list:
    filters = {"user_id": user_id}
    if memory_type:
        filters["type"] = memory_type
    if date_range:
        filters["timestamp"] = {"$gte": date_range[0], "$lte": date_range[1]}

    query_embedding = embed(query)
    return memory_vector_db.search(query_embedding, filter=filters, top_k=k)

# Examples:
retrieve_filtered_memories("usr_456", "past complaints", memory_type="complaint")
retrieve_filtered_memories("usr_456", "preferences", date_range=("2026-01-01", "2026-07-01"))
```

### Importance-Based Recall

Score memories by importance and retrieve the most important ones.

```python
def score_importance(memory: dict) -> float:
    """Score a memory by its importance for future interactions."""
    importance = 0.0
    if memory.get("user_explicitly_confirmed"):
        importance += 10.0  # User explicitly stated this
    if memory.get("repeated_across_sessions"):
        importance += 5.0   # Mentioned multiple times
    if memory.get("affects_personalization"):
        importance += 3.0
    if memory.get("is_temporal", True):
        importance -= age_days * 0.1  # Older = less important
    return importance

def retrieve_important_memories(user_id: str, threshold: float = 5.0) -> list:
    memories = load_all_user_memories(user_id)
    scored = [(m, score_importance(m)) for m in memories]
    important = [m for m, s in scored if s >= threshold]
    return sorted(important, key=lambda m: m["timestamp"], reverse=True)[:5]
```

### Hybrid Retrieval

Combine multiple strategies for optimal results.

```python
def hybrid_memory_retrieval(
    user_id: str,
    query: str,
    conversation_turns: list
) -> list:
    """Combine recency, similarity, and importance for comprehensive memory retrieval."""
    memories = []

    # 1. Always include recent conversation context
    memories.extend(get_recent_context(user_id, n=3))

    # 2. Retrieve semantically relevant long-term memories
    memories.extend(retrieve_similar_memories(query, user_id, k=3))

    # 3. Include high-importance facts
    memories.extend(retrieve_important_memories(user_id, threshold=7.0))

    # 4. Deduplicate and trim to budget
    return deduplicate_and_truncate(memories, max_tokens=2048)
```

---

## 5. Memory Compression & Pruning

### Conversation Summarization

Replace old conversation turns with a summary to preserve information while saving tokens.

```python
def summarize_conversation(conversation_text: str) -> str:
    prompt = f"""
    Summarize the following conversation. Capture:
    - Key information provided by the user (preferences, facts, decisions)
    - Actions taken or pending
    - Current state of any ongoing tasks

    Conversation:
    {conversation_text}

    Summary (3-5 sentences):
    """
    return llm_call(prompt)
```

```
Before (10 turns, 2000 tokens):
  User: Hi, I ordered a laptop on June 15th.
  Assistant: Let me look that up. What's your order number?
  User: ORD-7890
  Assistant: I found it. The laptop is estimated to arrive July 20th.
  User: That's later than expected. Can I upgrade shipping?
  Assistant: Let me check options... Yes, express shipping is available for $15.
  User: Okay, please upgrade it.
  Assistant: Done. Your estimated arrival is now July 16th.

After summary (50 tokens):
  "User ordered a laptop (ORD-7890) on June 15th. Original ETA July 20th. 
   Upgraded to express shipping for $15. New ETA July 16th."
```

### Sliding Window Truncation

Keep the most recent N tokens. Drop everything else.

```python
MAX_MEMORY_TOKENS = 2048

def truncate_to_budget(memory_texts: list[str], max_tokens: int = MAX_MEMORY_TOKENS) -> list[str]:
    """Keep as many recent items as fit within the token budget."""
    selected = []
    token_count = 0
    for text in reversed(memory_texts):  # Start from most recent
        tokens = count_tokens(text)
        if token_count + tokens <= max_tokens:
            selected.insert(0, text)
            token_count += tokens
        else:
            break
    return selected
```

### Importance Scoring

Rank memories by importance and keep only the most important ones.

```python
def prune_by_importance(memories: list[dict], max_items: int = 20) -> list[dict]:
    scored = [(m, importance_score(m)) for m in memories]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored[:max_items]]

def importance_score(memory: dict) -> float:
    score = 0.0
    # Explicit user facts (name, preferences)
    if memory["type"] == "user_fact": score += 10
    # Information repeated across sessions
    if memory.get("frequency", 1) > 1: score += memory["frequency"]
    # Event-based memories decay over time
    if memory.get("is_temporal"):
        days_old = (datetime.now() - memory["timestamp"]).days
        score *= max(0, 1 - days_old / 30)  # Decay over 30 days
    return score
```

### Deduplication

Remove redundant or near-duplicate memories before injecting into context.

```python
def deduplicate_memories(memories: list[dict], similarity_threshold: float = 0.92) -> list[dict]:
    unique = []
    for memory in memories:
        is_duplicate = any(
            cosine_similarity(memory["embedding"], existing["embedding"]) > similarity_threshold
            for existing in unique
        )
        if not is_duplicate:
            unique.append(memory)
    return unique
```

### Aging & Expiry Policies

Automatically expire memories that are no longer relevant.

```python
MEMORY_EXPIRY = {
    "conversation_turn": timedelta(hours=24),    # Short-lived
    "session_summary": timedelta(days=30),         # Medium-lived
    "user_fact": timedelta(days=365),              # Long-lived
    "confirmed_preference": None                   # Never expires
}

def apply_expiry_policy(memory_store: list[dict]) -> list[dict]:
    now = datetime.now()
    active = []
    for memory in memory_store:
        expiry = MEMORY_EXPIRY.get(memory["type"])
        if expiry is None or (now - memory["timestamp"]) < expiry:
            active.append(memory)
    return active
```

---

## 6. Memory in Agentic Systems

### State Persistence

Agents need to persist their state between invocations to handle long-running tasks.

```python
class AgentState:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.iteration: int = 0
        self.tools_used: list[str] = []
        self.results: dict = {}
        self.status: str = "running"

def save_agent_state(state: AgentState):
    db.query(
        "INSERT INTO agent_states (agent_id, state_json, updated_at) "
        "VALUES (?, ?, NOW()) ON CONFLICT (agent_id) DO UPDATE SET state_json = ?, updated_at = NOW()",
        state.agent_id, json.dumps(vars(state)), json.dumps(vars(state))
    )

def load_agent_state(agent_id: str) -> AgentState:
    row = db.query("SELECT state_json FROM agent_states WHERE agent_id = ?", agent_id)
    if row:
        return AgentState.from_dict(json.loads(row.state_json))
    return AgentState(agent_id)
```

### Checkpointing

Save intermediate states so the agent can resume after failure.

```python
def checkpoint_agent(agent_id: str, step: str, state: dict):
    """Save a checkpoint that the agent can resume from."""
    checkpoint = {
        "agent_id": agent_id,
        "step": step,
        "state": state,
        "timestamp": datetime.now().isoformat()
    }
    db.query(
        "INSERT INTO checkpoints (agent_id, step, checkpoint_json) VALUES (?, ?, ?)",
        agent_id, step, json.dumps(checkpoint)
    )

def resume_from_checkpoint(agent_id: str):
    """Find the latest checkpoint and resume."""
    row = db.query(
        "SELECT checkpoint_json FROM checkpoints WHERE agent_id = ? ORDER BY timestamp DESC LIMIT 1",
        agent_id
    )
    if row:
        checkpoint = json.loads(row.checkpoint_json)
        return checkpoint["step"], checkpoint["state"]
    return None, None
```

### Thread-Level Isolation

Each conversation or task gets its own isolated memory scope.

```python
class ThreadMemory:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.buffer: list = []
        self.state: dict = {}
        self.execution_memory: dict = {}

    def add_message(self, message: dict):
        self.buffer.append(message)

    def set_state(self, key: str, value):
        self.state[key] = value

    def get_state(self, key: str, default=None):
        return self.state.get(key, default)

# Usage: Each user session gets its own ThreadMemory
sessions: dict[str, ThreadMemory] = {}
thread = sessions.get(session_id, ThreadMemory(session_id))
```

### Multi-Agent Shared Memory

When multiple agents collaborate, they need a shared memory space.

```python
class SharedMemory:
    """Shared memory accessible by multiple agents."""
    def __init__(self):
        self.global_facts: dict = {}      # Facts agreed upon by all agents
        self.shared_tasks: list = []      # Task queue
        self.communication_log: list = [] # Messages between agents

    def add_fact(self, key: str, value: any, source_agent: str):
        self.global_facts[key] = {
            "value": value,
            "source": source_agent,
            "timestamp": datetime.now().isoformat(),
            "confirmed_by": [source_agent]
        }

    def confirm_fact(self, key: str, agent: str):
        if key in self.global_facts:
            self.global_facts[key]["confirmed_by"].append(agent)

# Example: Research agent + Writer agent
shared = SharedMemory()
# Research agent adds fact
shared.add_fact("economic_trend", "Inflation decreased 2% in Q2", "research_agent")
# Writer agent reads fact
fact = shared.global_facts["economic_trend"]
# Writer confirms
shared.confirm_fact("economic_trend", "writer_agent")
```

### Human-in-the-Loop Memory Updates

Allow humans to explicitly confirm, correct, or reject memories.

```python
def human_verify_memory(user_id: str, proposed_memory: dict) -> bool:
    """Ask the user to verify a proposed memory before storing."""
    prompt = f"""
    I'd like to remember this information about you:
    "{proposed_memory['text']}"

    Is this correct? (yes/no)
    """
    response = send_to_user(user_id, prompt)
    if response.lower() == "yes":
        store_memory(user_id, {
            "text": proposed_memory["text"],
            "type": proposed_memory.get("type", "fact"),
            "verified_by_user": True,
            "timestamp": datetime.now().isoformat()
        })
        return True
    return False
```

---

## 7. Memory Failure Modes

### Context Overflow

Too much memory content exceeds the context window.

```
Memory injected: 7,500 tokens
System prompt:    500 tokens
User query:        100 tokens
─────────────────────
Total:           8,100 tokens (exceeds 8K window)

→ Model doesn't see the last 100 tokens of memory
→ Critical fact may be truncated
→ Answer is based on incomplete information
```

| Fix | How |
|---|---|
| **Budget allocation** | Reserve 70% for memory, 20% for system, 10% for query |
| **Truncation strategy** | Drop oldest/least important memories first |
| **Compression** | Summarize verbose memories before injecting |
| **Larger model** | Use 128K context model if available |

### Stale or Outdated Memory

The model acts on old information that no longer applies.

```
Memory: "User lives in New York." (stored 2024)
Reality: "User moved to London in 2025."

Query: "Recommend restaurants near me."
Model: "Here are restaurants in New York..." → Wrong city!
```

| Fix | How |
|---|---|
| **Last-confirmed timestamp** | Track when each fact was last confirmed |
| **Expiry policy** | Delete or deprecate facts after a configurable TTL |
| **Re-confirmation** | Ask user to confirm old facts before using them |
| **Versioned memories** | Keep new fact, archive old one |
| **Temporal metadata** | Tag all memories with timestamp; prefer newer facts |

### Contradictory Memory

Two stored memories conflict with each other.

```
Memory A: "User prefers email notifications."
Memory B: "User turned off email notifications."

→ Which one should the model trust?
```

| Resolution Strategy | Implementation |
|---|---|
| **Recency wins** | Prefer the newer memory |
| **Explicit confirmation wins** | Prefer memories the user explicitly confirmed |
| **Frequency wins** | Prefer the pattern expressed more often |
| **Surface conflict** | "I see conflicting preferences. Which is correct?" |
| **Delete on update** | When updating a fact, archive the old version |

### Retrieval Drift

Over time, the retrieval system starts returning less relevant memories.

```
Year 1: Memory retrieval always returns relevant facts.
Year 2: Memory retrieval returns generic or irrelevant facts.
  → User's interests changed, but the embedding model still retrieves old patterns.
  → The system doesn't adapt to the user's evolving needs.
```

| Fix | How |
|---|---|
| **Re-embed periodically** | Recompute embeddings for changed understanding |
| **User feedback on retrieval** | Track which retrieved memories were actually useful |
| **Decay scores** | Lower scores for old memories, even if semantically similar |
| **Recency boost** | Apply recency weight to similarity scores |

### Privacy & Data Leakage Risks

Memory persistence introduces privacy risks.

| Risk | Scenario | Prevention |
|---|---|---|
| **Cross-user leakage** | User A's memories appear in User B's context | Strict user isolation (filter by user_id on every query) |
| **Over-retention** | Sensitive data kept longer than necessary | Expiry policies, automated purging |
| **Unintended recall** | Model recalls a sensitive memory inappropriately | Allow users to delete specific memories |
| **Memory injection** | Attacker injects false memories via prompts | Validate user inputs; don't auto-store unverified facts |

```python
def safe_memory_retrieval(user_id: str, query: str) -> list:
    """Retrieve memories with privacy safeguards."""
    # 1. Always filter by user_id — never leak cross-user data
    results = memory_db.search(query, filter={"user_id": user_id})

    # 2. Exclude expired memories
    results = [r for r in results if not is_expired(r["timestamp"])]

    # 3. Exclude sensitive categories unless explicitly authorized
    results = [r for r in results if r.get("privacy_level") != "sensitive" or is_authorized(user_id)]

    return results
```

---

## 8. Tools & Frameworks for Memory Management

### A. Framework-Level Tools

| Tool | Type | Memory Features | Best For |
|---|---|---|---|
| **LangGraph Checkpointer** | State persistence | Checkpoint agent state at each step, resume on failure | Agentic workflows, long-running agents |
| **LlamaIndex Memory** | Conversation buffer | Sliding window, summary memory, vector memory | RAG systems with memory |
| **Semantic Kernel Memory** | Semantic memory | File-based, vector DB storage, named collections | .NET ecosystems, enterprise |
| **Mem0** | Long-term memory | Automatic extraction, importance scoring, user memory profiles | Personalized AI applications |
| **Supermemory** | Long-term memory | Graph-based memory, semantic retrieval, web research memory | Knowledge-heavy applications |

**LangGraph Checkpointer (code example):**
```python
from langgraph.checkpoint import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")

# Agent state is automatically saved after each step
graph = graph.compile(checkpointer=memory)

# Resume from any checkpoint by thread_id
state = graph.get_state({"configurable": {"thread_id": "thread_123"}})
```

**Mem0 (code example):**
```python
from mem0 import Memory

m = Memory.from_config({"api_key": "..."})

# Automatically extract and store memories from conversation
m.add("User mentioned they love Italian food", user_id="usr_456")

# Retrieve relevant memories for a new interaction
memories = m.search("What does this user like to eat?", user_id="usr_456")
# → ["User loves Italian food", "User is vegetarian"]
```

### B. Storage Backends

| Backend | Memory Type | Persistence | Best For |
|---|---|---|---|
| **PostgreSQL** | Structured, relational | Persistent | User profiles, preferences, sessions |
| **Redis** | KV, session state | Volatile (with optional persistence) | Caching, short-term memory, session state at scale |
| **SQLite** | Embedded, file-based | Persistent | Prototyping, single-user, mobile |
| **MongoDB** | Document, flexible schema | Persistent | Complex memory structures, multi-session data |
| **pgvector** | Vector + relational | Persistent | Semantic memory + structured queries together |
| **Pinecone** | Vector only | Persistent (managed) | Semantic memory at scale |
| **Weaviate** | Vector + object | Persistent (self-hosted/managed) | Memory with built-in hybrid search |
| **Milvus** | Vector only | Persistent (distributed) | Billion-scale semantic memory |
| **Chroma** | Vector only | Persistent (embedded) | Prototyping, small-scale memory |

**Storage decision guide:**

| Memory Type | Storage | Reason |
|---|---|---|
| Conversation buffer (short-term) | Redis | Low latency, TTL expiry, session isolation |
| User profiles (long-term) | PostgreSQL | Structured queries, joins with other tables |
| Semantic memories (long-term) | pgvector or Pinecone | Vector search for relevant memory retrieval |
| Agent checkpoints (execution) | SQLite or PostgreSQL | ACID compliance, resume from failure |
| Session state (working) | Redis | Fast key-value, automatic expiry |
| Multi-agent shared memory | PostgreSQL + pgvector | Shared facts with vector retrieval |

### C. Caching & Session Management

| Tool | Use | Benefits |
|---|---|---|
| **Redis Cache** | Short-term memory, session state | Sub-millisecond reads, TTL-based expiry |
| **In-Memory Session Stores** | Per-process session state | Fastest option (no network), lost on restart |
| **Streamlit Session State** | UI-based session state | Built-in for Streamlit apps, auto-cleanup |
| **FastAPI Session Middleware** | Web app session management | JWT-based, middleware integration, configurable backends |

**Redis session state example:**
```python
import redis
from datetime import timedelta

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def save_session(session_id: str, data: dict, ttl: int = 3600):
    r.setex(f"session:{session_id}", timedelta(seconds=ttl), json.dumps(data))

def load_session(session_id: str) -> dict:
    data = r.get(f"session:{session_id}")
    return json.loads(data) if data else {}
```

### D. Observability & Debugging

| Tool | What It Monitors | Best For |
|---|---|---|
| **LangSmith** | LLM calls, chains, agent steps, memory state | End-to-end LLM application debugging |
| **OpenTelemetry** | Distributed traces, metrics, logs | System-level observability |
| **Custom Logging** | Memory reads/writes, retrieval latency | Targeted memory debugging |

**Custom memory logging example:**
```python
import logging

memory_logger = logging.getLogger("memory")

def retrieve_memory_with_logging(user_id: str, query: str) -> list:
    start = time.time()
    results = memory_db.search(query, filter={"user_id": user_id})
    elapsed = time.time() - start

    memory_logger.info({
        "event": "memory_retrieval",
        "user_id": user_id,
        "query_truncated": query[:100],
        "results_count": len(results),
        "latency_ms": round(elapsed * 1000, 2),
        "memory_types_retrieved": list(set(r["type"] for r in results))
    })

    return results
```

**Observability metrics to track:**
```
Memory Retrieval Metrics:
  - P50/P95/P99 latency per retrieval
  - Average number of memories retrieved per query
  - Cache hit rate (if caching is used)
  - Pruning rate (how many memories are compressed/dropped)
  - Memory storage size per user
  - Expired memories deleted per day
```

**LangSmith for memory debugging:**
- Trace every memory retrieval to understand what was injected
- Compare memory content across sessions for the same user
- Detect memory retrieval drift over time
- Identify which memories contributed to a hallucination
