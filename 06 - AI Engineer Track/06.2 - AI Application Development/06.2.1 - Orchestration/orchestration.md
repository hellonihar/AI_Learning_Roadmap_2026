# Orchestration

## 1. Introduction to Orchestration

### Why Single LLM Calls Are Insufficient for Real Applications

A single LLM call — one prompt, one output — works for simple tasks like classification, summarization, or translation. Real applications demand more:

| Use Case | Single Call Fails Because... |
|---|---|
| **Customer support bot** | Needs to classify intent → retrieve docs → generate answer → check safety → format response. One call can't do all of this reliably. |
| **Code generation agent** | Needs to understand requirements → generate code → run tests → fix errors → iterate. One call is never enough. |
| **Research assistant** | Needs to decompose question → search multiple sources → synthesize → verify facts → cite. One call hallucinates without grounding. |
| **Data extraction pipeline** | Needs to process 10K documents reliably. One call per document works, but you need orchestration to manage batching, retries, error handling, and cost. |

**The core problem:** A single LLM call is **stateless, non-deterministic, and single-pass**. It can't iterate, can't recover from errors, can't decompose complex tasks, and can't use external tools. Orchestration wraps the LLM with control flow that provides these capabilities.

### From "One Prompt → One Output" to Multi-Step Workflows

```
Single call:
  [Prompt] ──→ [LLM] ──→ [Output]

Multi-step workflow:
  [System Prompt] ──→ [LLM Call 1: Classify intent]
                               │
                               ▼
                    [Conditional Router]
                     │              │
              [Call 2a: FAQ]    [Call 2b: Escalate]
                     │              │
                     ▼              ▼
              [Call 3: Format]   [Call 3: Draft handoff]
                               │
                               ▼
                         [Final Output]
```

Each step is a deterministic wrapper around a probabilistic LLM call. The wrapper handles: input construction, output parsing, validation, retries, and handoff to the next step.

### Deterministic Control Around Probabilistic Systems

Orchestration is the practice of building **deterministic control flow** around **probabilistic LLM calls**.

```
┌─────────────────────────────────────────────────────┐
│  Deterministic Layer (Your Code)                    │
│  ─────────────────────────────                      │
│  - Control flow (if/else, loops, error handling)    │
│  - Input assembly (context construction)            │
│  - Output validation (schema checking)              │
│  - Tool execution (API calls, DB queries)           │
│  - State management (conversation history)          │
├─────────────────────────────────────────────────────┤
│  Probabilistic Layer (LLM)                          │
│  ────────────────────────────                       │
│  - Language understanding & generation              │
│  - Reasoning & planning                             │
│  - Format adaptation                                │
│  - Creative variation                               │
└─────────────────────────────────────────────────────┘
```

The deterministic layer constrains what the LLM can do and when. It validates outputs before acting on them, retries when outputs are invalid, and enforces structural rules the LLM can't be trusted to follow alone.

### LLM as a Component, Not the System

The most common mistake in AI engineering: treating the LLM as the entire system.

```
Bad architecture:
  [User Input] ──→ [LLM] ──→ [Output]
  
  → Everything depends on the LLM getting it right on the first try.
  → No validation, no fallback, no recovery.

Good architecture:
  [User Input] ──→ [Router] ──→ [Retriever] ──→ [Context Builder] ──→ [LLM] ──→ [Validator]
                                                                                   │
                                                                              [Pass/Fail]
                                                                              │       │
                                                                           [Output]  [Retry]
```

The LLM is one component among many. Retrieval, validation, tool execution, and state management are equally important. The system should work (perhaps poorly) even if the LLM is replaced with a different model.

### Workflow vs Pipeline vs Decision Loop

| Pattern | Structure | Best For |
|---|---|---|
| **Workflow** | Fixed sequence of steps. Each step runs once. | Document processing, data extraction, report generation |
| **Pipeline** | Directed acyclic graph (DAG). Steps can branch and merge. | ETL, multi-stage RAG, parallel processing |
| **Decision loop** | Cyclic. Model can make decisions, take actions, observe results, and continue. | Agents, chatbots, iterative coding |

```
Workflow (linear):
  Step A → Step B → Step C → Done

Pipeline (branched):
        ┌─→ Step B1 ─→ ┐
  Step A ─→ Step B2 ─→ ─→ Step C → Done
        └─→ Step B3 ─→ ┘

Decision loop:
  Step A → LLM decides → Action → Observe → LLM decides again → ...
              ↑_____________________________________|
```

---

## 2. Core Orchestration Patterns

### A. Sequential Workflows

The simplest orchestration pattern: steps run one after another, each using the output of the previous step.

```
User Input
  ↓
Step 1: Classify intent → "refund_request"
  ↓
Step 2: Extract order_id → "ORD-12345"
  ↓
Step 3: Look up order in database → {order details}
  ↓
Step 4: Generate response → "Your refund for order ORD-12345 has been processed."
```

**Chaining prompts:** Each step constructs a new prompt from the previous step's output.

```python
def refund_workflow(user_input: str) -> str:
    # Step 1: Classify intent
    intent = classify_intent(user_input)  # LLM call

    if intent != "refund_request":
        return route_to_other_workflow(intent)

    # Step 2: Extract order ID
    order_id = extract_order_id(user_input)  # LLM call
    if not order_id:
        return "I couldn't find an order ID in your request."

    # Step 3: Look up in database (deterministic — no LLM)
    order = db.query("SELECT * FROM orders WHERE id = ?", order_id)

    # Step 4: Generate response
    response = generate_refund_response(order)  # LLM call
    return response
```

**When to use:** Tasks where each step depends on the previous one and the sequence is fixed.

**Limitation:** No branching, no error recovery between steps (except at the application level).

### B. Conditional Routing

Direct the workflow to different paths based on LLM output or programmatic conditions.

#### If-Else Branching

```python
def route_request(user_input: str) -> str:
    intent = classify_intent(user_input)

    if intent == "refund":
        return refund_workflow(user_input)
    elif intent == "cancel":
        return cancel_workflow(user_input)
    elif intent == "support":
        return support_workflow(user_input)
    else:
        return fallback_workflow(user_input)
```

**Real example — customer support routing:**

```
User: "I want my money back"
  ↓
Classifier: determine_intent(user_input)
  ↓
  ├── "refund" ──────────→ refund_workflow
  ├── "cancel_subscription" → cancel_workflow
  ├── "technical_issue" ──→ technical_workflow
  ├── "sales" ───────────→ sales_workflow
  └── "unknown" ─────────→ human_handoff
```

#### Classifier-Based Routing

Use a dedicated (usually small, cheap) model for classification, then route to more expensive models only when needed.

```
All requests → Classifier (GPT-4o mini, or fine-tuned BERT)
                  │
           ┌──────┼────────┬────────┬────────┐
           ▼      ▼        ▼        ▼        ▼
        refund  cancel  support  sales   unknown
           │      │        │        │        │
           ▼      ▼        ▼        ▼        ▼
        GPT-4o  GPT-4o  GPT-4o  GPT-4o  Human agent
        (high   (high   (high   (high   (expensive
         cost)   cost)   cost)   cost)    but needed)
```

**Why this matters for cost:** 80% of requests might be simple (FAQs, order status) and can be handled by a cheap model. Only 20% need expensive model capability. A classifier router saves 4× in LLM costs.

#### Tool Selection via Model Output

Let the LLM decide which tool to call based on the user's request.

```
User: "What's the weather in Paris?"
  ↓
LLM: think → "User wants weather data"
  ↓
LLM output: {"tool": "get_weather", "arguments": {"location": "Paris"}}
  ↓
System executes get_weather(location="Paris")
  ↓
Returns result to LLM → LLM generates response
```

**Best practice:** Constrain tool selection to a predefined set. Let the LLM choose the tool and fill the arguments, but validate the tool name against an allowed list before executing.

### C. Parallel Execution

Run multiple LLM calls simultaneously for independent tasks, then aggregate results.

```
User query: "Compare the performance of GPT-4o, Claude 3.5, and Gemini 1.5"
  ↓
Parallel calls:
  ┌──────────────────────────────────────────────────┐
  │ Call 1: "Research GPT-4o specifications"          │
  │ Call 2: "Research Claude 3.5 specifications"      │
  │ Call 3: "Research Gemini 1.5 specifications"      │
  └──────────────────────────────────────────────────┘
  ↓
Aggregator: "Synthesize these into a comparison table"
  ↓
Final output
```

```python
import asyncio

async def compare_models(query: str) -> str:
    # Parallel research calls
    results = await asyncio.gather(
        research_model("GPT-4o"),
        research_model("Claude 3.5"),
        research_model("Gemini 1.5")
    )

    # Aggregate into comparison
    comparison = await synthesize_comparison(results)
    return comparison
```

**Parallel execution patterns:**

| Pattern | How It Works | Latency | Cost |
|---|---|---|---|
| **Fan-out** | Run N independent LLM calls in parallel | Max(N latencies) — best | N× cost |
| **Voting** | Run same prompt N times, pick majority | Same as one call (parallel) | N× cost |
| **Debate** | Multiple models argue positions, then synthesize | Same as one call (parallel) | N× cost |
| **Map-reduce** | Split task into N sub-tasks, run in parallel, reduce | Max(N latencies) | N× cost |

**When to use:** Tasks where subtasks are independent (research, multi-source analysis, ensemble classification).

**Cost consideration:** Parallel execution gives linear cost increase for latency that equals the slowest call. Use it when latency is more important than cost.

### D. Looping & Retry Patterns

Cyclic patterns where the LLM can iterate on its own output.

#### Reflection Loops

The model critiques its own output and improves it.

```python
def generate_with_reflection(prompt: str, max_iterations: int = 3) -> str:
    output = llm_call(prompt)

    for i in range(max_iterations - 1):
        critique_prompt = f"""
        Original task: {prompt}
        Generated output: {output}
        Review the output for errors, omissions, and quality issues.
        Provide a revised and improved version.
        """
        improved = llm_call(critique_prompt)
        if improved == output:  # No change → converged
            break
        output = improved

    return output
```

**Real example — code generation:**
```
Step 1: "Write a Python function to sort a list of numbers" → [initial code]
Step 2: "Review this code for edge cases (empty list, duplicates, already sorted)" → [fixed code]
Step 3: "Now optimize for performance" → [optimized code]
```

**Trade-off:** Each iteration doubles or triples token usage. Most gains come from the first 1–2 iterations. After that, diminishing returns.

#### Self-Critique Cycles

The model explicitly evaluates its own output against criteria before accepting it.

```python
def generate_with_validation(prompt: str, criteria: list[str], max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        output = llm_call(prompt)

        # Ask the model to evaluate its own output
        eval_prompt = f"""
        Does the following output meet these criteria?
        Criteria: {criteria}
        Output: {output}
        Respond with PASS or list what failed.
        """
        evaluation = llm_call(eval_prompt)

        if evaluation.startswith("PASS"):
            return output

        # Include feedback in the next attempt
        prompt += f"\n\nPrevious attempt failed: {evaluation}\nFix the issues."

    return output  # Best effort after max_retries
```

#### Retry on Failure

When output validation fails (invalid JSON, wrong format, empty response), retry with error feedback.

```python
def generate_json_with_retry(prompt: str, schema: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        raw = llm_call(prompt + "\nReturn ONLY valid JSON.")

        try:
            parsed = json.loads(raw)
            validate_schema(parsed, schema)
            return parsed
        except (json.JSONDecodeError, ValidationError) as e:
            prompt += f"\n\nError: {e}. Fix the JSON and try again."

    raise MaxRetriesExceeded("Failed to generate valid JSON after 3 retries")
```

| Retry Trigger | Recovery Action | Max Retries |
|---|---|---|
| Invalid JSON | Re-prompt with parse error message | 3 |
| Empty output | Re-prompt with "Output was empty. Generate a complete response." | 2 |
| Schema violation | Re-prompt with specific validation errors | 3 |
| Refusal ("I can't do that") | Modify prompt to clarify scope (if appropriate) | 1 |
| Timeout | Reduce context and retry | 2 |

#### Max Iteration Safeguards

Always bound loops. An unbound loop can run forever (and cost unlimited money).

```python
MAX_ITERATIONS = 10  # Hard upper bound
iteration = 0

while not task_complete and iteration < MAX_ITERATIONS:
    output = llm_call(prompt)
    task_complete = check_completion(output)
    iteration += 1

if iteration >= MAX_ITERATIONS:
    fallback_action()  # Escalate to human, return partial result, etc.
```

**Why this matters:** An agent with a retry loop and no iteration limit can run for hours and cost thousands of dollars on a single request. Always set a budget and a cap.

---

## 3. Structured Output in Orchestration

Structured output is how you pass data between orchestration steps. Each LLM call must produce output that your code can parse, validate, and route.

### JSON Output Prompting

Every orchestration step should produce structured data, not free text.

```python
step1_prompt = """
You are an intent classifier. Analyze the user's message and return JSON:
{
  "intent": "refund" | "cancel" | "support" | "sales" | "unknown",
  "confidence": 0.0-1.0,
  "entities": {
    "order_id": string or null,
    "product": string or null
  }
}
Message: {user_input}
"""
```

**Why JSON matters for orchestration:** Your code can parse JSON deterministically. It can route based on `result.intent`, extract `result.entities.order_id`, and validate against a schema. With free text, you'd need another LLM call just to parse the output.

### Schema-Constrained Outputs

Define schemas for each step. Include them directly in prompts.

```python
class ExtractionSchema:
    """Define the expected output schema as a prompt string."""
    PROMPT = """
    {
      "amount": number,
      "currency": "USD" | "EUR" | "GBP",
      "date": "YYYY-MM-DD" or null,
      "description": string
    }
    """
```

In orchestration, schemas serve two purposes:
1. Tell the LLM what format to produce
2. Tell your validation code what format to expect

```python
def extract_financial_data(text: str) -> dict:
    prompt = f"""
    Extract financial data from:
    {text}

    Return JSON:
    {ExtractionSchema.PROMPT}
    """
    raw = llm_call(prompt)
    return validate_and_parse(raw, ExtractionSchema.PROMPT)
```

### Function/Tool Calling in Orchestration

API-level function calling is the most reliable way to get structured output for orchestration.

```python
functions = [
    {
        "name": "extract_order_info",
        "description": "Extract order details from user message",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "action": {"type": "string", "enum": ["refund", "cancel", "status_check"]}
            },
            "required": ["action"]
        }
    }
]

# LLM output: {"name": "extract_order_info", "arguments": '{"order_id": "ORD-123", "action": "refund"}'}
```

In orchestration, every step can use function calling:

| Step | Function Name | Purpose |
|---|---|---|
| Classify | `classify_intent` | Determine what to do next |
| Extract | `extract_entities` | Pull structured data from input |
| Retrieve | `search_knowledge_base` | Get relevant documents |
| Generate | `generate_response` | Produce final output |

### Deterministic Format Enforcement

| Technique | How It Works | Reliability |
|---|---|---|
| **Temperature 0** | Removes randomness in generation | High for structured output |
| **JSON mode (API)** | API guarantees JSON output | Very high |
| **Function calling** | API enforces function schema | Very high |
| **Structured generation libs** | Instructor, Outlines, Guidance | Highest |
| **Post-parse + retry** | Validate output, retry on failure | High (costs latency) |

**Best practice in orchestration:** Use function calling or JSON mode as the primary mechanism. Use post-parse validation + retry as the safety net.

### Parsing Strategies in Orchestration

Every orchestration step that receives LLM output must parse and validate before proceeding.

```python
def safe_llm_step(prompt: str, schema: dict, max_retries: int = 2) -> dict:
    """Orchestration step: call LLM, parse output, validate, retry on failure."""
    for attempt in range(max_retries + 1):
        raw = llm_call(prompt, response_format={"type": "json_object"})
        try:
            parsed = json.loads(raw)
            validated = validate_schema(parsed, schema)
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                prompt += f"\nPrevious output was invalid: {e}\nFix the error."
            else:
                raise OrchestrationStepFailed(step=prompt, error=e)
```

---

## 4. Tool Integration & Function Calling

### Tool Calling Basics

Tools extend what the LLM can do beyond generating text. They give the model access to external systems.

```
User asks: "What's the weather in Tokyo?"
  ↓
LLM decides: I need weather data for Tokyo
  ↓
LLM outputs a function call: get_weather(location="Tokyo")
  ↓
Your code executes the function → {"temperature": 22, "condition": "sunny"}
  ↓
Result is returned to LLM as a new message
  ↓
LLM generates response: "The weather in Tokyo is 22°C and sunny."
```

**The LLM doesn't execute tools — it requests tool execution.** Your code is the runtime that executes tools and returns results.

### Structured Function Schemas

Define tools clearly with name, description, and parameter schema.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g., 'AAPL'"}
                },
                "required": ["ticker"]
            }
        }
    }
]
```

**Schema design principles:**
- **Descriptive names:** `get_weather` not `func_a`. The model uses names to decide which tool to call.
- **Clear descriptions:** The description is the most important field — it tells the model when to use this tool.
- **Enums for limited options:** Use `enum` instead of open-ended strings when possible.
- **Required vs optional:** Mark truly required fields only. Optional fields with defaults give the model flexibility.

### Passing Arguments

When the LLM requests a tool call, it provides arguments as a JSON string. Your code must parse and pass them.

```python
def execute_tool_call(tool_call: dict) -> str:
    """Execute a tool call requested by the LLM and return the result."""
    tool_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    if tool_name == "get_weather":
        result = get_weather(**arguments)
    elif tool_name == "get_stock_price":
        result = get_stock_price(**arguments)
    else:
        raise UnknownToolError(f"Unknown tool: {tool_name}")

    return json.dumps(result)  # Return as string for LLM context
```

**Important:** Always validate argument types before calling the tool. The LLM can pass strings where numbers are expected, or omit required fields despite the schema.

### Handling Tool Errors

Tools fail. The LLM must know when a tool failed so it can respond appropriately.

```python
def execute_tool_call(tool_call: dict) -> str:
    try:
        result = actual_tool_execution(tool_call)
        return json.dumps({"status": "success", "data": result})
    except ToolExecutionError as e:
        return json.dumps({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error_type": "unexpected_error",
            "message": "An unexpected error occurred."
        })
```

**Error result injected back into context:**

```
User: "What's the weather in Tokyo?"
Assistant (tool call): get_weather(location="Tokyo")
Tool result: {"status": "error", "error_type": "RateLimitError", "message": "API rate limit exceeded"}
Assistant response: "I'm sorry, I couldn't fetch the weather data right now due to a service limitation. Please try again in a moment."
```

**Never let the model see raw stack traces.** Errors returned to the model should be sanitized.

### Returning Tool Outputs to Model

Tool outputs are injected back into the context as a new message with the `tool` role.

```python
# After tool execution, add the result to messages
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})

# The LLM uses this to generate its final response
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)
```

**Context structure with tools:**

```
System: "You are a helpful assistant with access to weather data."
User: "What's the weather in Tokyo?"
Assistant (tool call): get_weather(location="Tokyo")
Tool: {"temperature": 22, "condition": "sunny"}
Assistant (final): "The weather in Tokyo is 22°C and sunny."
```

### When to Let Model Choose Tool vs Deterministic Tool Routing

| Approach | How It Works | When to Use |
|---|---|---|
| **Model chooses** | LLM sees all tools, decides which to call | Open-ended queries where the right tool depends on user intent |
| **Deterministic routing** | Your code chooses the tool based on classification | Known query types where intent → tool mapping is fixed |
| **Hybrid** | Classifier routes to a subset of tools, model chooses within subset | Large tool sets (20+) where showing all tools degrades quality |

```
Deterministic routing:
  Intent: "weather" → tool = get_weather (no choice)
  Intent: "stock" → tool = get_stock_price (no choice)
  → Cheaper, faster, more reliable. Use when possible.

Model chooses:
  Intent: "I need help planning my trip"
  → LLM might call: get_weather, get_flights, get_hotels, get_restaurants
  → Flexible but less predictable. Use for open-ended tasks.
```

**Best practice:** Start deterministic. Move to model-choice only when the conversation is too open-ended to hardcode routing. Deterministic routing is cheaper, faster, and never calls the wrong tool.

### Tool Integration Anti-Patterns

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| **Giving too many tools** | Model chooses wrong tool or gets confused | Limit to 5–10 tools per conversation. Group related tools. |
| **Poor tool descriptions** | Model calls wrong tool or passes wrong args | Write descriptions as if the model has never seen the tool before |
| **Not handling errors** | Model hallucinates results when tool fails | Always return error status; tell the model it failed |
| **Exposing side effects** | Model calls delete/update tools unpredictably | Separate read tools (model can call) from write tools (require confirmation) |
| **No timeout** | Tool hangs, LLM waits forever | Set 5–10s timeouts on all tool calls |

### Multi-Turn Tool Use

Complex tasks may require multiple tool calls in sequence, where the result of one tool informs the next.

```
User: "Find me a cheap flight to Paris next week and check the weather."

Turn 1:
  Assistant: search_flights(destination="Paris", date="next week", price_limit="cheap")
  Tool: [list of flights]

Turn 2 (model sees flight results, now checks weather):
  Assistant: get_weather(location="Paris", date=best_flight_date)
  Tool: {"temperature": 18, "condition": "rainy"}

Turn 3 (model has all information):
  Assistant: "Here are some affordable flights to Paris next week...
  Forecast shows rain, so pack an umbrella!"
```

**Engineering consideration:** Each tool call adds a round-trip. Multi-turn tool use is powerful but slow and expensive. Set a maximum number of tool calls per request (typically 3–5).

### Orchestration Framework Comparison

For production systems, use an orchestration framework rather than building from scratch (unless you have a dedicated team).

| Framework | Style | Best For | Complexity |
|---|---|---|---|
| **LangChain** | Chain/LCEL | Quick prototyping, broad ecosystem | Medium |
| **Haystack** | Pipeline/DAG | RAG, search, document processing | Medium |
| **LlamaIndex** | Agent/query engine | RAG, data-intensive applications | Medium |
| **CrewAI** | Multi-agent | Agent teams, role-based workflows | Medium |
| **DSPy** | Compiler-based | Optimizing prompt chains programmatically | High (powerful) |
| **OpenAI Assistants API** | Managed agent | Simple tool-use, no infrastructure | Low |
| **Vercel AI SDK** | Streaming/react | Frontend-heavy AI applications | Low |
| **Custom (no framework)** | Full control | Production systems, specific requirements | High |

**When NOT to use a framework:**
- Your workflow is simple (2–3 linear steps)
- You need minimal dependencies
- You want full control over token counting and cost
- You're already experienced and the framework adds more complexity than it saves

**Best practice:** Start with a framework (LangChain or LlamaIndex) for prototyping. If the framework's abstractions leak or constrain you, port to custom orchestration. Most production systems at scale are custom.
