# Exercises — Agentic AI

## Exercise 1: Design a ReAct Agent Flow

**Problem**: You are building a customer support agent for an e-commerce platform. The agent needs to handle order status inquiries, returns, and refunds.

Design the ReAct flow for the query: "I want to return my order #4421 because it arrived damaged."

**Answer**:

`
Thought: The user wants to return order #4421 due to damage. I need to verify the order exists and check eligibility.
Action: get_order_status(order_id="4421")
Observation: Order #4421 is delivered on 2026-07-10, eligible for return until 2026-08-10.

Thought: The order is eligible. I should check the return policy for damaged items.
Action: get_return_policy(category="electronics")
Observation: Damaged items are eligible for full refund within 30 days. User needs a return label.

Thought: All conditions met. I'll initiate the return and provide instructions.
Action: initiate_return(order_id="4421", reason="damaged")
Observation: Return initiated. Return label RMA-8842 generated.

Final Answer: I've initiated a return for order #4421. Your RMA number is RMA-8842. A return label has been emailed to you. Please ship the item within 7 days.
`

---

## Exercise 2: Memory Strategy Selection

**Problem**: For each scenario, choose the best memory strategy (short-term window, summarization, vector retrieval, knowledge graph, episodic memory).

| Scenario | Best Strategy | Why |
|----------|---------------|-----|
| Chatbot that remembers user name across sessions | Long-term (key-value or vector store) | Simple fact, must persist across sessions |
| Code assistant that references 100+ lines of conversation history | Summarization + sliding window | Context too large for full window; summarize old parts |
| Research agent that writes 10-page reports referencing past findings | Vector retrieval | Past findings are semantic — similarity search works best |
| Customer support agent handling multi-step refunds | Checkpointing + short-term | Task is short but must survive interruptions |
| Agent that learns which tools work best for which tasks over time | Episodic memory | Learning from past action sequences |

---

## Exercise 3: Fix the Tool Definition

**Problem**: The following tool definition has several issues. Identify and fix them.

`json
{
  "type": "function",
  "function": {
    "name": "send_email",
    "parameters": {
      "properties": {
        "recipient": { "type": "string" },
        "message": { "type": "string" }
      }
    }
  }
}
`

**Answer**:

Issues:
1. **Missing description** for the tool itself — the LLM won't know when to use it.
2. **Missing required field** — neither parameter is marked required.
3. **Missing descriptions** on parameters — helps LLM fill them correctly.

Fixed version:

`json
{
  "type": "function",
  "function": {
    "name": "send_email",
    "description": "Send an email to a recipient. Use when the user asks to send a message via email.",
    "parameters": {
      "type": "object",
      "properties": {
        "recipient": {
          "type": "string",
          "description": "Email address of the recipient"
        },
        "message": {
          "type": "string",
          "description": "Body content of the email"
        }
      },
      "required": ["recipient", "message"]
    }
  }
}
`

---

## Exercise 4: Implement Error Recovery

**Problem**: Write the error handling logic for an agent that calls search_flights. The API may return a 429 (rate limited), 503 (unavailable), or 404 (no flights). Show how the agent should respond in each case.

**Answer**:

`python
def handle_flight_search_error(status_code: int, context: dict) -> str:
    if status_code == 429:
        # Rate limited — retry with backoff
        wait_time = context.get("retry_after", 5)
        return f"Rate limited. Retrying in {wait_time} seconds."

    elif status_code == 503:
        # Service unavailable — try backup
        return ("Primary flight search unavailable. "
                "Trying backup provider.")

    elif status_code == 404:
        # No flights found — inform user
        return ("No flights found for the given route and date. "
                "Ask user if they want to try different dates or nearby airports.")

    else:
        return f"Unexpected error (HTTP {status_code}). Inform user and escalate."
`

LLM responses after receiving these errors:

- **429**: "The search service is busy. I'll wait a moment and retry." (agent retries automatically)
- **503**: "The primary search is down. Let me try the backup system."
- **404**: "I couldn't find any flights from JFK to NRT on July 24. Would you like to try a different date or nearby airports like EWR or LGA?"

---

## Exercise 5: Framework Selection

**Problem**: You need to select a framework for these scenarios. Justify your choice.

| Scenario | Best Framework | Justification |
|----------|---------------|---------------|
| A multi-agent system where a manager agent delegates tasks to specialist agents (researcher, writer, reviewer) | CrewAI or AutoGen | Both support role-based or conversation-based multi-agent setups natively. CrewAI is simpler; AutoGen is more flexible. |
| A production document processing pipeline with 50+ state transitions and rollback | LangGraph | LangGraph's graph-based state machine with built-in checkpointing is ideal for complex stateful workflows. |
| A quick prototype of a code-analysis assistant with file I/O tools | OpenAI Assistants API | Fastest to prototype with built-in code interpreter and file search. Zero infrastructure. |
| An enterprise app in .NET that integrates with Azure OpenAI and Cognitive Search | Semantic Kernel | Deep Azure integration, first-class .NET support, and auto-planning with available plugins. |

---

## Exercise 6: Write an MCP Server (Conceptual)

**Problem**: Outline an MCP server that provides weather data. Show the server capabilities, tool definition, and how a client would connect.

**Answer**:

`python
# MCP Weather Server (conceptual outline)

class WeatherMCPServer:
    \"\"\"MCP server providing weather tools and resources.\"\"\"

    capabilities = {
        "tools": [
            {
                "name": "get_forecast",
                "description": "Get weather forecast for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "default": 1}
                    },
                    "required": ["city"]
                }
            }
        ],
        "resources": [
            {
                "uri": "weather://current/{city}",
                "description": "Current weather for a city",
                "mime_type": "application/json"
            }
        ],
        "prompts": [
            {
                "name": "weather_report",
                "description": "Generate a weather report",
                "arguments": [
                    {"name": "city", "required": true}
                ]
            }
        ]
    }

    async def handle_call_tool(self, name: str, args: dict) -> str:
        if name == "get_forecast":
            return f"Weather in {args['city']}: Sunny, 25°C"

    async def handle_read_resource(self, uri: str) -> str:
        city = uri.split("/")[-1]
        return json.dumps({"city": city, "temp": 25, "condition": "Sunny"})

# Client connects via stdio or HTTP+SSE
# client = MCPClient("stdio", ["python", "weather_server.py"])
# tools = await client.list_tools()  # Discovers get_forecast
# result = await client.call_tool("get_forecast", {"city": "Tokyo"})
`

---

## Bonus Challenge

Extend the code_walkthrough.md ReAct agent to support:

1. **Tool result caching**: If the same tool is called with the same arguments in the same session, return cached result.
2. **Confidence-based fallback**: If tool results are incomplete, ask the LLM to rate its confidence; below threshold, call a human handoff tool.
3. **Token-aware context management**: Before each LLM call, check token count; if approaching limit, summarize and prune old messages.

These exercises cover reasoning, memory, tool use, error handling, framework selection, and MCP — the core competencies of Agentic AI engineering.
