# Agentic AI — Cheatsheet

## 1. Agent Reasoning Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **ReAct** | Interleave Thought → Action → Observation | Most general-purpose agents |
| **CoT** | Step-by-step reasoning before answering | Knowledge-intensive questions |
| **ToT** | Explore multiple reasoning branches in parallel | Puzzles, strategy, planning |
| **Plan-and-Solve** | Generate full plan first, then execute step by step | Compliance, auditable workflows |

### ReAct Loop Pseudocode
`
while not done:
    thought = llm.reason(state)
    action = llm.choose_tool(thought)
    observation = execute(action)
    state.update(observation)
`

## 2. Memory Types

| Memory Type | Storage | Persistence | Access |
|-------------|---------|-------------|--------|
| **Short-term** | Context window | Session only | Direct (in prompt) |
| **Long-term (semantic)** | Vector DB, Knowledge Graph | Cross-session | Semantic search |
| **Episodic** | Action logs DB | Cross-session | Retrieval by task/goal |

### Memory Management
- **Sliding window**: Keep last N messages
- **Summarization**: Compress old context into summary
- **Consolidation**: Extract facts after session → store in vector DB
- **Checkpointing**: Persist state after each step for recovery

## 3. Tool Definition Format

### OpenAI
`json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "What this tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {"type": "string", "description": "..."}
      },
      "required": ["param1"]
    }
  }
}
`

### Anthropic
`json
{
  "name": "tool_name",
  "description": "What this tool does",
  "input_schema": {
    "type": "object",
    "properties": {
      "param1": {"type": "string"}
    },
    "required": ["param1"]
  }
}
`

### Common Tool Types
- **Search**: Web, vector, database, code
- **Code Execution**: Python REPL, shell, SQL
- **File I/O**: read, write, list, search files
- **API**: HTTP client, email, CRM
- **Structured**: Calendar, contacts, tickets

### Tool Selection Modes
- uto — model decides when to call tools
- 
one — no tool calling
- orce — require a specific tool

## 4. MCP Overview

**Model Context Protocol** — Open standard for LLM-tool communication.

`
┌──────────────┐     ┌──────────────┐
│  MCP Client   │────▶│  MCP Server   │
│  (App/Host)   │◀───│  (Tool Prov.)  │
└──────────────┘     └──────────────┘
`

**Servers expose**:
- **Resources**: URI-addressable data (files, DB records, API responses)
- **Tools**: Executable functions (LLM-callable)
- **Prompts**: Pre-written templates

**Transport**: stdio (local) or HTTP+SSE (remote)

**Key advantage**: Write once, use with any MCP-compatible client (Claude Desktop, Cursor, VS Code, custom agents).

## 5. Framework Comparison

| Feature | LangGraph | AutoGen | CrewAI | Semantic Kernel | Assistants API |
|---------|-----------|---------|--------|-----------------|----------------|
| **Paradigm** | State graph | Multi-agent conversation | Role-based crews | Plugin orchestration | Managed cloud |
| **Planning** | Custom edges | Conversation | Sequential/hierarchical | Auto-planner | None |
| **Memory** | Checkpointing | Conversation history | Short/long/entity | Semantic memory | Threads |
| **Multi-Agent** | Custom graphs | Native | Native | Limited | No |
| **Checkpointing** | Built-in | No | No | No | No |
| **Learning Curve** | High | Medium | Low-Medium | Medium | Low |
| **Vendor Lock-in** | None | None | None | Moderate (Azure) | High (OpenAI) |

## 6. Error Recovery Patterns

`python
# Retry with exponential backoff
for attempt in range(max_retries):
    try:
        return execute(tool_call)
    except TransientError:
        time.sleep(2 ** attempt)

# Alternative tool on failure
if "no results" in observation:
    action = "try_backup_tool"  # LLM decides next step

# Human handoff
if all_alternatives_exhausted:
    return "I cannot complete this. Here's what I tried: ..."
`

## 7. Quick Checklist

- [ ] Define tools with clear names and descriptions
- [ ] Set max_turns limit (prevents infinite loops)
- [ ] Handle tool errors by returning error string to LLM
- [ ] Use checkpointing for long-running workflows
- [ ] Implement slot-filling for multi-turn interactions
- [ ] Choose memory strategy based on task duration
- [ ] Select framework based on complexity needs
- [ ] Never use eval() in production — sandbox code execution
- [ ] Log all tool calls for debugging
- [ ] Test failure scenarios explicitly
