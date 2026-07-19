# 06.3.2 — State & Memory

Agents without memory are reactive — they respond to each request as if it were the first. To build agents that carry context, learn from experience, and resume interrupted workflows, we need structured memory and state management. This lesson covers the three memory tiers and the state management patterns that tie them together.

## The Three Memory Tiers

Memory in agentic systems maps loosely to human memory: sensory (immediate), short-term (working), and long-term (episodic + semantic).

### 1. Short-Term Memory (STM)

Short-term memory holds the immediate context of the current conversation or task.

**Implementation**: The LLM's context window. Every message (system prompt, user messages, assistant responses, tool results) is appended to a list of messages.

**Characteristics**:
- Limited size (8K–200K tokens depending on model)
- Fast access (no retrieval step)
- Automatically managed by the model (attention over all tokens)
- **Volatile**: content is lost when the conversation ends or the window overflows

**Window Management Strategies**:

- **Sliding window**: Keep the last N messages; drop oldest when limit is reached.
- **Token budget**: Reserve a portion of the context window for the system prompt and recent history; truncate middle messages first.
- **Summarization**: When approaching the limit, ask the LLM to summarize older messages into a compressed representation.
- **Dynamic truncation**: Drop function call arguments and tool results (which tend to be verbose) before dropping messages.

```python
# Pseudocode: sliding window with summarization
messages = [system_prompt] + recent_history
if total_tokens(messages) > max_tokens:
    summary = llm.summarize(older_messages)
    messages = [system_prompt, summary] + recent[-10:]
```

### 2. Long-Term Memory (LTM)

Long-term memory persists across sessions. An agent that remembers user preferences, past conversations, or learned facts uses LTM.

**Storage Backends**:

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| Vector Stores (Pinecone, Chroma, Qdrant) | Semantic search, similarity-based recall | No structure, can return irrelevant results |
| Key-Value Stores (Redis, SQLite) | Fast exact lookup | No fuzzy matching |
| Knowledge Graphs (Neo4j, Amazon Neptune) | Relation traversal, structured queries | Complex setup, rigid schema |
| Relational DB (PostgreSQL) | ACID, complex queries | Slower semantic search |

**Retrieval Patterns**:

- **Semantic retrieval**: Embed the current user query, find top-K similar past conversations, inject into context.
- **Entity extraction**: Extract entities (names, dates, IDs) from conversation and store in a knowledge graph for later relation queries.
- **Preference accumulation**: Save user corrections and preferences explicitly (e.g., "user prefers email over Slack").

**Memory Consolidation**:

At the end of each session (or periodically), the agent can:

1. Extract facts from the conversation
2. Update or create entities in the knowledge graph
3. Summarize the session and store the summary as a retrievable document
4. Prune outdated or irrelevant memories

```python
# Memory consolidation
session_facts = llm.extract_facts(conversation_history)
for fact in session_facts:
    memory_store.save(fact)
```

### 3. Episodic Memory

Episodic memory stores past *experiences* — sequences of actions, observations, and outcomes. This allows the agent to recall "the last time I tried to book a flight, the API returned a 503 and I had to use the backup."

**Use Cases**:
- Learning that certain tool combinations work well (or poorly)
- Avoiding repeated mistakes (e.g., "calling `send_email` before `validate_address` caused an error last time")
- Providing the user with a summary of past interactions

**Implementation**:
- Store action sequences as structured episodes: `[state, action, observation, reward/outcome]`
- Index episodes by goal or task type for retrieval
- Retrieve relevant episodes before planning to inform decision-making

**Episodic vs. Semantic Memory**:
- **Semantic**: "The user's timezone is EST" (fact)
- **Episodic**: "Last time the user asked about EST, the conversation went: user asked time, agent provided EST, user confirmed. This led to a positive outcome."

## State Management

State refers to the agent's current progress in a multi-step task. Without state management, an agent cannot recover from interruptions or continue tasks across turns.

### Tracking Progress

**State Machine Approach**:

Each task is modeled as a finite state machine:

```
Task: "Process refund"
States: [START, VALIDATING, APPROVING, PROCESSING, COMPLETED, FAILED]
Transitions:
  START → VALIDATING (on input received)
  VALIDATING → APPROVING (if valid)
  VALIDATING → FAILED (if invalid)
  APPROVING → PROCESSING (if approved)
  APPROVING → FAILED (if denied)
  PROCESSING → COMPLETED (on success)
  PROCESSING → FAILED (on error)
```

The agent checks the current state before choosing the next action.

**Checklist Approach**:

Maintain a list of steps with their status:

```json
{
  "task": "Prepare quarterly report",
  "steps": [
    {"id": 1, "description": "Gather data from Salesforce", "status": "done"},
    {"id": 2, "description": "Calculate Q2 metrics", "status": "done"},
    {"id": 3, "description": "Generate charts", "status": "in_progress"},
    {"id": 4, "description": "Write executive summary", "status": "pending"}
  ]
}
```

### Multi-Turn Execution

Many agent tasks require multiple user interactions. For example, a travel agent might:

1. Ask for destination → user provides
2. Ask for dates → user provides
3. Search flights → present options
4. User selects → book

State management ensures the agent remembers *which* step it is on and *what data* has been collected so far.

**Slot-Filling Pattern**:

Define a schema of required information (slots). The agent fills slots across turns:

```json
{
  "required_slots": ["destination", "departure_date", "return_date", "budget"],
  "filled_slots": {
    "destination": "Tokyo",
    "departure_date": "2026-07-24"
  },
  "unfilled_slots": ["return_date", "budget"]
}
```

Each turn, the agent checks which slots remain, asks for the next one, and validates the input.

### Checkpoint-Based Recovery

For long-running or critical tasks, agents should persist state to durable storage so they can recover from crashes.

**Checkpoint Contents**:
- Current step / state
- Collected data
- Conversation history (or summary)
- Partial results

**Recovery Flow**:

1. Agent starts and loads the latest checkpoint
2. Reconstructs context from checkpoint data (not full history)
3. Determines the next action based on saved state
4. Proceeds without losing progress

```python
class CheckpointManager:
    def save(state: dict, step: str):
        persist_to_db({"state": state, "step": step, "timestamp": now()})

    def load(task_id: str) -> dict | None:
        return load_from_db(task_id)

    def clear(task_id: str):
        delete_from_db(task_id)
```

**When to Checkpoint**:
- After each successful tool call
- Before and after critical operations (payment, data deletion)
- When switching between agents or handing off to a human

## Memory in Practice

Most production agents use a **hybrid memory architecture**:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Context Window | LLM native | Current conversation |
| Sliding Window | Code | Keep recent N messages |
| Summarization Buffer | LLM + storage | Compress older context |
| Vector Store | Embeddings + DB | Semantic retrieval |
| Knowledge Graph | Graph DB | Entity relations |
| Episodic Store | DB + LLM | Past action sequences |

The LangGraph framework (see 06.3.5) provides a built-in `MemorySaver` that checkpoints graph state after each step, enabling automatic recovery.

## Key Takeaways

- Short-term memory is the context window — manage it with sliding windows, token budgets, or summarization.
- Long-term memory uses vector stores for semantic retrieval and knowledge graphs for structured facts.
- Episodic memory stores past action sequences so the agent can learn from experience.
- State management via FSM, checklists, or slot-filling ensures agents can handle multi-turn tasks.
- Checkpointing enables recovery from failures without restarting from scratch.
- Production agents combine all three memory tiers in a layered architecture.

In 06.3.3, we examine how agents interact with the world through tool use.
