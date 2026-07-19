# 06.3.3 — Tool Use

An agent without tools is a chatbot. Tools extend an LLM's capabilities beyond text generation — allowing it to search the web, run code, read files, call APIs, and interact with the physical world. This lesson covers how tools are defined, how agents select them, and how to handle failures.

## Tool Definition Schema

Every tool needs a machine-readable description so the LLM knows when and how to call it. The two dominant schemas are OpenAI function calling and Anthropic tool use.

### OpenAI Function Calling Schema

`json
{
  "type": "function",
  "function": {
    "name": "search_flights",
    "description": "Search for flights between two airports on a given date",
    "parameters": {
      "type": "object",
      "properties": {
        "origin": { "type": "string", "description": "IATA code, e.g. JFK" },
        "destination": { "type": "string", "description": "IATA code, e.g. NRT" },
        "date": { "type": "string", "description": "YYYY-MM-DD" }
      },
      "required": ["origin", "destination", "date"]
    }
  }
}
`

Key elements:
- **name**: Unique identifier the model uses to request the tool
- **description**: Natural language explanation of what the tool does — critical for selection
- **parameters**: JSON Schema defining expected arguments
- **required**: Which parameters the model must supply

### Anthropic Tool Use Schema

`json
{
  "name": "search_flights",
  "description": "Search for flights between two airports on a given date",
  "input_schema": {
    "type": "object",
    "properties": {
      "origin": { "type": "string" },
      "destination": { "type": "string" },
      "date": { "type": "string", "format": "date" }
    },
    "required": ["origin", "destination", "date"]
  }
}
`

### Universal Principles

- **Name**: Short, descriptive, unique
- **Description**: The most important field — be explicit about when to use the tool
- **Parameters**: Clearly typed with descriptions
- **Strictness**: OpenAI supports strict: true for guaranteed schema adherence

## Tool Types

### 1. Search Tools
- **Web Search**: Current events, fact-checking
- **Vector Search**: Semantic retrieval from internal docs
- **Database Query**: Structured data lookup (SQL, GraphQL)
- **Code Search**: Find functions, classes in a codebase

### 2. Code Execution Tools
- **Python REPL**: Execute arbitrary Python code, return stdout/stderr
- **Jupyter Notebook**: Interactive execution with cell outputs
- **Shell**: Run terminal commands (with sandboxing)
- **SQL**: Query databases directly

**Warning**: Always sandbox code execution in containers or restricted environments.

### 3. File I/O Tools
- ead_file(path): Read file contents
- write_file(path, content): Write to a file
- list_directory(path): List files in a directory
- search_files(pattern): Grep/glob for files

### 4. API Call Tools
- call_http(method, url, headers, body): Generic HTTP client
- send_email(to, subject, body): Send email
- get_weather(city): Fetch weather data
- github_*: GitHub operations (PRs, issues, commits)

### 5. Structured Data Tools
- create_calendar_event(title, time, attendees)
- dd_contact(name, email, phone)
- update_crm_record(record_id, fields)

## Tool Selection

### Deterministic Selection

The model picks a tool based on:
1. **Description matching**: The tool's description matches the user's intent
2. **Parameter availability**: Required parameters are known or can be inferred
3. **Context**: Previous tool results inform the next choice

### Parallel Tool Calling

Modern LLMs (GPT-4o, Claude 3.5+) can call multiple tools in one response:

`json
{
  "tool_calls": [
    {"name": "search_flights", "args": {"origin": "JFK", "destination": "NRT", "date": "2026-07-24"}},
    {"name": "get_weather", "args": {"city": "Tokyo", "date": "2026-07-24"}}
  ]
}
`

### Tool Calling Loop

`python
def agent_loop(user_input, tools, max_turns=10):
    messages = [{"role": "user", "content": user_input}]

    for turn in range(max_turns):
        response = llm.chat(messages, tools=tools)

        if response.tool_calls is None:
            return response.content  # Final answer

        for tc in response.tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Max turns reached without resolution."
`

### Forced Tool Selection

Some frameworks allow forcing a specific tool or preventing tool use:

- 	ool_choice: "auto" — model decides (default)
- 	ool_choice: "none" — no tools allowed
- 	ool_choice: {"type": "function", "function": {"name": "search_flights"}} — force a specific tool

## Error Handling When Tools Fail

Tools fail. Networks drop, APIs return 500s, file paths don't exist. A robust agent must handle these gracefully.

### Common Failure Modes

| Failure | Cause | Recovery |
|---------|-------|----------|
| HTTP Error | API down, rate limited | Retry with backoff, try alternative |
| Invalid Arguments | LLM generated bad params | Send error to LLM, let it fix |
| Timeout | Tool took too long | Set timeout limit, return timeout error |
| Permission | Tool not authorized | Inform user, request permission |
| Empty Result | No data found | Return "no results", ask user to refine |

### Error Recovery Pattern

`python
def safe_execute_tool(tool_call, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute_tool(tool_call.name, tool_call.args)
            return result
        except RateLimitError:
            if attempt == max_retries - 1:
                return "Error: Rate limited after 3 retries."
            time.sleep(2 ** attempt)
        except HTTPError as e:
            if e.status == 503:
                return "Error: Service temporarily unavailable."
            return f"Error: HTTP {e.status}"
        except Exception as e:
            return f"Error: {str(e)}"
`

### Returning Errors to the LLM

The LLM treats the error string as a normal observation. A good system prompt instructs it to:

- Try alternative tools when one fails
- Ask the user for missing information
- Admit when a task cannot be completed
- Never make up fake results when tools fail

`python
# The LLM sees this as an observation:
{"role": "tool", "tool_call_id": "call_abc", "content": "Error: Service temporarily unavailable."}
# Next LLM response:
# "The search service is down. I'll try the backup search service instead."
`

### Graceful Degradation

When critical tools fail, the agent should:
1. Inform the user of the failure
2. Offer alternatives or partial results
3. Allow the user to decide next steps

## Best Practices

- **Descriptions matter**: Spend time writing clear tool descriptions — this directly impacts selection accuracy
- **Validate parameters**: Some frameworks support parameter validation before calling
- **Set turn limits**: Always cap the number of tool calls to prevent infinite loops
- **Log tool calls**: Record every tool invocation for debugging and auditing
- **Sandbox execution**: Never run arbitrary code tools in production without isolation
- **Return structured errors**: Error messages should explain what went wrong and what the LLM can do next

## Key Takeaways

- Tools are defined via JSON Schema (OpenAI) or input_schema (Anthropic)
- Common tool categories: search, code execution, file I/O, API calls, structured data
- The LLM selects tools based on descriptions and parameter availability
- Parallel tool calling reduces latency for independent operations
- Error handling must be explicit — tools fail, and the agent must recover gracefully
- Always set turn limits to prevent infinite tool-calling loops

In 06.3.4, we explore the Model Context Protocol (MCP) and how it standardizes tool and data-source interactions.
