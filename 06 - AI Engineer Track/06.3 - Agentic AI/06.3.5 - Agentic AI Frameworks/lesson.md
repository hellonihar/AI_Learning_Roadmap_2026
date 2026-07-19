# 06.3.5 — Agentic AI Frameworks

Building agents from scratch is educational but impractical for production. Frameworks provide scaffolding for planning, memory, tool use, and multi-agent coordination. This lesson compares the five major frameworks: LangGraph, AutoGen, CrewAI, Semantic Kernel, and OpenAI Assistants API.

## LangGraph

**Creator**: LangChain  
**Approach**: Graph-based state machines  
**Language**: Python (primary), JavaScript

LangGraph models agent workflows as directed graphs where **nodes** are functions and **edges** define state transitions.

### Architecture

`python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("agent", call_llm)     # LLM decides next action
graph.add_node("tools", execute_tools) # Execute tool calls

graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {
    "continue": "agent",
    "end": END
})

graph.set_entry_point("agent")
app = graph.compile(checkpointer=MemorySaver())
`

**Planning**: Nodes can implement any reasoning pattern (ReAct, CoT). Conditional edges enable dynamic replanning.

**Memory**: Built-in MemorySaver checkpoints state after each step. Supports short-term (within-graph) and long-term (persistent store) memory.

**Tool Use**: Tools are standard LangChain tools. The agent node calls the LLM with tool definitions; tool results flow back into the graph.

**Pros**:
- Extremely flexible — you can model any control flow
- Built-in checkpointing for fault tolerance
- Large ecosystem (LangChain integrations)
- Great debugging with LangSmith

**Cons**:
- Steep learning curve (graphs are abstract)
- Verbose for simple agents
- LangChain dependency is heavy

**Best For**: Complex, stateful workflows; production systems needing checkpointing and observability.

## AutoGen

**Creator**: Microsoft Research  
**Approach**: Multi-agent conversation  
**Language**: Python

AutoGen frames agent interaction as **conversations** between agents. Each agent has a role, and they exchange messages to complete tasks.

### Architecture

`python
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o"},
    system_message="You are a helpful assistant."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

# Two-agent conversation
user_proxy.initiate_chat(assistant, message="Write a Python script to plot Q2 sales data.")
`

**Multi-agent patterns**:
- **Two-agent**: Assistant + UserProxy (standard)
- **Group chat**: Multiple agents with a GroupChatManager for routing
- **Nested chats**: An agent spawns sub-conversations

**Planning**: No built-in planning — agents reason via conversation. The assistant agent can use ReAct-style thinking in its responses.

**Memory**: Basic conversation history. No built-in long-term memory — must be implemented externally.

**Tool Use**: Tools are functions registered with agents. Code execution is a first-class feature (UserProxy can run Python).

**Pros**:
- Excellent for multi-agent scenarios
- Code execution built in
- Good for role-playing and debate-style tasks
- Group chat pattern is powerful for complex coordination

**Cons**:
- Single-agent use is overkill
- No built-in state machine or checkpointing
- Conversation-based design can be chatty (high token usage)

**Best For**: Multi-agent systems, research simulations, code generation with execution verification.

## CrewAI

**Creator**: CrewAI (open source)  
**Approach**: Role-based agent crews  
**Language**: Python

CrewAI borrows from the **Orchestra pattern** — you define agents with roles, goals, and backstories, then assemble them into crews that work on tasks.

### Architecture

`python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive information",
    backstory="Expert in data gathering and analysis",
    tools=[search_tool, scrape_tool],
    allow_delegation=True
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear documentation",
    backstory="Expert in explaining complex topics",
    tools=[write_tool]
)

task = Task(
    description="Research and document MCP protocol",
    agent=researcher,
    expected_output="A comprehensive markdown document"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    process=Process.sequential  # or Process.hierarchical
)
`

**Planning**: Agents plan internally via LLM reasoning. Crews support sequential (one agent after another) or hierarchical (manager delegates subtasks) processes.

**Memory**: Supports short-term, long-term, and entity memory via integration with memory stores.

**Tool Use**: Tools are LangChain-compatible tools. Agents can delegate tasks to each other.

**Pros**:
- Intuitive mental model (roles, crews, tasks)
- Great documentation and examples
- Built-in memory support
- Agent delegation (agents can ask other agents for help)

**Cons**:
- Less flexible than LangGraph for custom control flow
- Role-based abstraction can feel too rigid for some use cases
- Dependency on LangChain ecosystem

**Best For**: Content generation pipelines, research workflows, document processing with multiple specialized agents.

## Semantic Kernel

**Creator**: Microsoft  
**Approach**: AI orchestration SDK  
**Language**: C#, Python, Java

Semantic Kernel is Microsoft's official AI orchestration SDK. It connects LLMs to plugins (tools) and orchestrates their execution.

### Architecture

`python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelFunction

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o"))

# Register a plugin
class WeatherPlugin:
    @kernel_function(description="Get weather for a city")
    def get_forecast(self, city: str) -> str:
        return f"Weather in {city}: Sunny, 25°C"

kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

# Run
result = kernel.invoke_prompt(
    function_name="weather_agent",
    prompt="What's the weather in Tokyo?"
)
`

**Planning**: Provides a SequentialPlanner that automatically decomposes a goal into steps using available plugins. Also supports FunctionCallingStepwisePlanner (ReAct-style).

**Memory**: SemanticTextMemory with vector store integration (Azure Cognitive Search, Chroma). Supports memory consolidation and recall.

**Tool Use**: Tools are **plugins** — collections of kernel_function-decorated methods. Plugins can be native code or OpenAI-style prompt-based.

**Pros**:
- Deep Azure/AI integration (Azure OpenAI, Cognitive Search, etc.)
- Enterprise-grade (Microsoft support, .NET ecosystem)
- Auto-planning is powerful
- Good for .NET shops

**Cons**:
- Python SDK lags behind C# in features
- Documentation can be scattered
- Auto-planning can be slow for complex tasks
- Less community adoption than LangChain

**Best For**: Enterprise applications, .NET ecosystems, Azure-heavy stacks, applications needing tight Microsoft integration.

## OpenAI Assistants API

**Creator**: OpenAI  
**Approach**: Managed cloud service  
**Language**: Any (REST API)

The Assistants API is a hosted agent platform. You define an assistant with a model, instructions, and tools — OpenAI manages the execution.

### Architecture

`python
from openai import OpenAI

client = OpenAI()

assistant = client.beta.assistants.create(
    name="Support Agent",
    instructions="You are a customer support agent.",
    model="gpt-4o",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"},
        {"type": "function", "function": {"name": "search_orders", "parameters": {...}}}
    ]
)

thread = client.beta.threads.create()
client.beta.threads.messages.create(thread.id, role="user", content="Where is my order?")

run = client.beta.threads.runs.create(thread.id, assistant_id=assistant.id)

# Poll until complete (runs are async)
while run.status in ("queued", "in_progress", "requires_action"):
    if run.status == "requires_action":
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            output = execute_tool(tool_call)
            client.beta.threads.runs.submit_tool_outputs(thread.id, run.id, ...)
    run = client.beta.threads.runs.retrieve(thread.id, run.id)
`

**Planning**: No explicit planning — the assistant reasons via LLM calls. OpenAI handles the execution loop internally.

**Memory**: **Threads** serve as short-term memory (conversation history). No built-in long-term memory across threads. ile_search provides RAG against uploaded files.

**Tool Use**: Three built-in tool types:
- code_interpreter: Sandboxed Python execution
- ile_search: Vector search over uploaded files
- unction: Custom function calling

**Pros**:
- Zero infrastructure — fully managed by OpenAI
- No framework dependency in your code
- Code interpreter is powerful (generates charts, runs analysis)
- File search is simple RAG out of the box
- Streaming support

**Cons**:
- Vendor lock-in (OpenAI-only)
- Limited control over execution logic
- No multi-agent support
- No built-in checkpointing (runs are ephemeral)
- Thread-based memory is limited (no long-term persistence)
- Polling-based run model adds latency

**Best For**: Rapid prototyping, simple single-agent applications, internal tools where managed infrastructure is preferred.

## Framework Comparison Summary

| Feature | LangGraph | AutoGen | CrewAI | Semantic Kernel | Assistants API |
|---------|-----------|---------|--------|-----------------|----------------|
| **Paradigm** | State graph | Multi-agent conversation | Role-based crews | Plugin orchestration | Managed cloud |
| **Planning** | Custom graph edges | Conversation-driven | Sequential/hierarchical | Auto-planner | None (manual) |
| **Memory** | Checkpointing + store | Conversation history | Short/long/entity | Semantic memory | Thread + file search |
| **Tool Use** | LangChain tools | Registered functions | LangChain tools | Plugins (decorators) | Code_interpreter, function, file_search |
| **Multi-Agent** | Yes (custom graphs) | Yes (native) | Yes (native) | Limited | No |
| **Checkpointing** | Built-in | No | No | No | No |
| **Vendor Lock-in** | No | No | No | Moderate (Azure) | High (OpenAI) |
| **Learning Curve** | High | Medium | Low-Medium | Medium | Low |
| **Best For** | Complex stateful workflows | Multi-agent research | Content pipelines | Enterprise .NET | Quick prototypes |

## Choosing a Framework

**Use LangGraph when** you need fine-grained control over agent execution, complex branching logic, or production-grade checkpointing.

**Use AutoGen when** your problem involves multiple agents collaborating, debating, or reviewing each other's work.

**Use CrewAI when** you want a simple, role-based mental model for multi-agent workflows like research and content creation.

**Use Semantic Kernel when** you're in a Microsoft/.NET environment and need deep Azure integration with auto-planning capabilities.

**Use Assistants API when** you want the quickest path to a working agent with minimal code and don't mind vendor lock-in.

## Key Takeaways

- LangGraph is the most flexible but has the steepest learning curve
- AutoGen excels at multi-agent conversation patterns
- CrewAI offers the most intuitive role-based abstraction
- Semantic Kernel is the enterprise choice for Microsoft shops
- Assistants API is best for rapid prototyping with zero infrastructure
- There is no single best framework — choose based on your control needs, team expertise, and infrastructure requirements

This concludes the Agentic AI section. Next steps: complete the exercises in  6.3/exercises.md, review the cheatsheet in  6.3/cheatsheet.md, and build the ReAct agent walkthrough in  6.3/code_walkthrough.md.
