# 06.3.4 — MCP & Other Protocols

As the agent ecosystem grows, so does the fragmentation problem. Every LLM provider has its own tool format. Every framework defines its own interface. The **Model Context Protocol (MCP)** aims to standardize how LLMs interact with tools and data sources. This lesson explains MCP and compares it with other protocols.

## What is MCP?

The **Model Context Protocol** is an open standard (originated by Anthropic) that defines a universal way for LLM applications to discover and invoke tools, access resources, and use prompts from external servers.

Think of MCP as **USB-C for AI agents** — a single protocol that lets any MCP-compatible client connect to any MCP-compatible server, regardless of the LLM backend.

### Core Concepts

MCP uses a **client-server architecture**:

`
┌──────────────────┐         ┌──────────────────┐
│   MCP Client      │ ──────▶│  MCP Server       │
│  (Host App)       │◀────── │  (Tool Provider)   │
│                   │         │                   │
│  - Claude Desktop │         │  - Calculator     │
│  - VS Code ext    │         │  - GitHub API     │
│  - Custom Agent   │         │  - File System    │
│  - Cursor         │         │  - Database       │
└──────────────────┘         └──────────────────┘
`

**Client (Host)**: The application that the user interacts with. It connects to one or more MCP servers and routes tool calls from the LLM to the appropriate server.

**Server**: A lightweight program that exposes capabilities. Each server provides one or more of:

| Capability | Description |
|------------|-------------|
| **Resources** | Read-only data (files, database records, API responses) exposed as URI-addressable content |
| **Tools** | Executable functions the LLM can invoke with parameters |
| **Prompts** | Pre-written prompt templates that the client can use |

### How MCP Works

**Transport Layer**:
MCP supports two transports:
- **stdio**: Server runs as a subprocess, communicates via stdin/stdout. Simple, secure, good for local tools.
- **HTTP+SSE (Server-Sent Events)**: Server runs remotely, communicates via HTTP. Good for cloud-hosted tools.

**Lifecycle**:

1. **Initialization**: Client connects to server, negotiates protocol version, exchanges capabilities.
2. **Listing**: Client requests list of available tools/resources/prompts from the server.
3. **Calling**: The LLM requests a tool call; the client forwards it to the appropriate MCP server.
4. **Notification**: Servers can send notifications (e.g., "file changed") to clients.

**MCP in Practice**:

`python
# Pseudocode: MCP client usage
mcp_client = MCPClient("stdio", server_command=["python", "weather_server.py"])
tools = await mcp_client.list_tools()
# tools = [Tool(name="get_forecast", description="...", parameters={...})]

# LLM decides to call get_forecast
result = await mcp_client.call_tool("get_forecast", {"city": "Tokyo"})
`

### Resources in MCP

Resources are how MCP handles *data* rather than *actions*:

`
resource://weather/tokyo
resource://files/reports/q2-2026.pdf
resource://db/users/4421
`

The client fetches resources and injects them into the LLM context. This is how MCP handles retrieval-augmented generation (RAG) natively.

## Other Protocols

### OpenAI Plugin Protocol

OpenAI's plugin system (introduced with ChatGPT plugins) defines a **manifest file** and an **API**:

1. **Manifest** (i-plugin.json): Metadata about the plugin (name, description, logo, API endpoint).
2. **OpenAPI spec**: Standard OpenAPI/Swagger document describing endpoints.
3. **Runtime**: ChatGPT calls the plugin's REST API endpoints.

**Comparison with MCP**:

| Aspect | MCP | OpenAI Plugin |
|--------|-----|---------------|
| Transport | stdio, HTTP+SSE | HTTP REST only |
| Tool Discovery | Runtime list_tools() | Static manifest + OpenAPI |
| Auth | Embedded in transport | OAuth2, API Key |
| Resources | Native concept (URIs) | Not supported |
| Prompts | Native concept | Not supported |
| Open Standard | Yes | No (OpenAI-specific) |
| Local Tools | First-class (stdio) | Limited (HTTP required) |

### Anthropic Tool Format

Anthropic's native tool format (covered in 06.3.3) is simple but limited:

`json
{
  "name": "tool_name",
  "description": "...",
  "input_schema": { ... }
}
`

**Limitations**:
- No server abstraction — tools are defined inline in every API call
- No resource concept — all data must be passed via context windows or tool results
- No dynamic discovery — the list is fixed at request time
- No standard transport — tools are just JSON schemas

MCP was designed by Anthropic to address these limitations while keeping the same simple tool schema structure.

### LangChain Tool Interface

LangChain defines its own tool abstraction:

`python
class BaseTool(BaseModel):
    name: str
    description: str
    args_schema: Type[BaseModel]

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError
`

LangChain tools are framework-internal. They do not define a wire protocol — they work only within LangChain chains/agents. MCP serves a similar purpose but at the protocol level, making it framework-agnostic.

**Comparison**:

| Aspect | MCP | LangChain Tool |
|--------|-----|----------------|
| Scope | Cross-framework protocol | Framework internal |
| Transport | stdio, HTTP+SSE | Python function call |
| Language | Agnostic | Python only |
| Modularity | Separate server processes | In-process |
| Adoption | Growing ecosystem | Mature, large ecosystem |

## Why MCP Matters

### For Tool Builders

Write one MCP server and it works with Claude Desktop, Cursor, VS Code, VS Codium, and any custom agent that implements the MCP client. No need to build separate integrations for each platform.

### For Agent Developers

MCP provides a **standard interface** — you don't need to write custom glue code for each tool provider. Your agent dynamically discovers available tools at runtime.

### For the Ecosystem

MCP enables a **marketplace of capabilities**:
- Database MCP servers
- File system MCP servers
- GitHub MCP servers
- Figma MCP servers
- Custom business logic servers

Standardization reduces fragmentation and accelerates adoption.

## MCP vs. Function Calling

MCP is **not** a replacement for function calling — it's a layer above it:

`
LLM API (function calling)
      ↕
MCP Client (routes tool calls)
      ↕
MCP Servers (provide tools)
`

The LLM still uses its native function-calling format. The MCP client translates between the LLM's format and the MCP server's interface.

## Key Takeaways

- MCP standardizes how LLM applications interact with tools and data sources
- Uses client-server architecture with stdio or HTTP+SSE transport
- Three capability types: Resources (data), Tools (actions), Prompts (templates)
- More flexible than OpenAI's plugin protocol (supports local tools, resources, dynamic discovery)
- More standardized than Anthropic's inline tool format
- Framework-agnostic — works with any LLM or framework
- Not a replacement for function calling, but a layer above it

In 06.3.5, we compare the major agentic AI frameworks that implement these concepts.
