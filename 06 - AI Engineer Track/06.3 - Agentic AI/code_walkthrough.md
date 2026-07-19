# Code Walkthrough: Building a ReAct Agent in Python

This walkthrough builds a simple agent using the ReAct (Reasoning + Acting) pattern. We define tools, make LLM calls with function calling, implement the execution loop, and add error recovery. The code uses OpenAI's SDK but the concepts apply to any provider.

---

## Step 1: Define Tool Schemas

Each tool needs a JSON Schema definition and a corresponding Python function.

`python
import json
import time
from typing import Any
from openai import OpenAI

client = OpenAI()

# --- Tool definitions (OpenAI format) ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Tokyo, New York"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. 2 + 2 * 3"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
`

## Step 2: Implement Tool Functions

`python
# --- Tool implementations ---

def get_weather(city: str) -> str:
    weather_data = {
        "Tokyo": "22°C, partly cloudy",
        "New York": "28°C, humid",
        "London": "15°C, rain",
        "Paris": "19°C, sunny"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

def calculator(expression: str) -> str:
    try:
        # SAFETY: eval is dangerous in production. Use a safe math parser.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

def search_knowledge_base(query: str) -> str:
    knowledge = {
        "python version": "Python 3.12 is the latest stable version",
        "agentic ai": "Agentic AI refers to AI systems that can autonomously plan and execute actions",
        "react pattern": "ReAct = Reasoning + Acting, a pattern where LLMs interleave thought and action"
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"No information found for: {query}"

TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base
}
`

## Step 3: The ReAct Execution Loop

`python
def execute_tool_call(tool_call) -> str:
    \"\"\"Execute a single tool call with error handling.\"\"\"
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"  Calling: {name}({json.dumps(args)})")

    if name not in TOOL_MAP:
        return f"Error: Unknown tool '{name}'"

    try:
        result = TOOL_MAP[name](**args)
        return str(result)
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {str(e)}"
    except Exception as e:
        return f"Error: {name} failed with: {str(e)}"


def react_agent(user_input: str, max_turns: int = 10) -> str:
    \"\"\"Main ReAct agent loop.\"\"\"
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with access to tools. "
            "Think step by step about what information you need. "
            "Use tools to gather information, then provide the answer. "
            "If a tool fails, try an alternative approach or ask for clarification."
        )},
        {"role": "user", "content": user_input}
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_msg = response.choices[0].message

        # If no tool calls, the agent is done
        if not assistant_msg.tool_calls:
            final = assistant_msg.content or ""
            print(f"  Final answer: {final[:100]}...")
            return final

        # Add assistant message to history
        messages.append(assistant_msg)

        # Execute all tool calls in parallel
        for tool_call in assistant_msg.tool_calls:
            result = execute_tool_call(tool_call)

            if result.startswith("Error:"):
                print(f"  ⚠ {result}")
            else:
                print(f"  Result: {result[:80]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Agent reached maximum turn limit without completing the task."
`

## Step 4: Error Recovery in Action

`python
def react_agent_with_retry(
    user_input: str,
    max_turns: int = 10,
    max_retries: int = 3
) -> str:
    \"\"\"ReAct loop with automatic retry for transient errors.\"\"\"
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with access to tools. "
            "When a tool returns an error, explain the issue to the user "
            "and try an alternative approach if possible."
        )},
        {"role": "user", "content": user_input}
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""

        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            result = None
            for attempt in range(max_retries):
                result = execute_tool_call(tool_call)
                if not result.startswith("Error:"):
                    break
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max turns reached."
`

## Step 5: Usage Example

`python
# Example 1: Simple question with tool use
result = react_agent("What is the weather in Tokyo and calculate 15 * 4?")
# Output:
# --- Turn 1 ---
#   Calling: get_weather({"city": "Tokyo"})
#   Result: 22°C, partly cloudy
#   Calling: calculator({"expression": "15 * 4"})
#   Result: 60
# --- Turn 2 ---
#   Final answer: The weather in Tokyo is 22°C and partly cloudy. 15 * 4 = 60.

# Example 2: Research question
result = react_agent("What is the ReAct pattern in AI?")
# --- Turn 1 ---
#   Calling: search_knowledge_base({"query": "ReAct pattern"})
#   Result: ReAct = Reasoning + Acting, a pattern where LLMs interleave...
# --- Turn 2 ---
#   Final answer: The ReAct pattern combines Reasoning and Acting...

# Example 3: Error recovery
result = react_agent("What is the weather in Mars?")
# --- Turn 1 ---
#   Calling: get_weather({"city": "Mars"})
#   Result: Weather data not available for Mars
# --- Turn 2 ---
#   Final answer: I'm sorry, I don't have weather data for Mars.
`

## Key Concepts Demonstrated

1. **ReAct Loop**: The LLM reasons (thought is implicit in the response), acts (tool calls), observes (tool results), and repeats.
2. **Tool Definitions**: JSON Schema informs the LLM what tools exist and when to use them.
3. **Error Propagation**: Tool errors are returned as strings into the message loop — the LLM decides what to do next.
4. **Parallel Calls**: Multiple independent tool calls execute in one turn.
5. **Turn Limit**: Prevents infinite loops.
6. **Retry Logic**: Counts transient failures and retries with exponential backoff.

## Production Considerations

- **Sandboxing**: Never use eval() in production. Use st.literal_eval, a math library, or a sandboxed environment.
- **Async**: Use async/await for concurrent tool execution when tools are I/O-bound.
- **Observability**: Log every message (including tool calls and results) for debugging.
- **Timeouts**: Wrap tool execution in syncio.wait_for() or use threading timeouts.
- **Authentication**: Validate that the LLM can only call permitted tools (avoid prompt injection to access restricted tools).
- **Streaming**: Use streaming for the final answer to reduce perceived latency.
