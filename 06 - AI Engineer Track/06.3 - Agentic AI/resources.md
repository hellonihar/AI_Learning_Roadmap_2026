# Resources — Agentic AI

## Foundational Papers

### Reasoning & Planning

| Paper | Authors | Year | Summary |
|-------|---------|------|---------|
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Yao et al. | 2022 | Introduces the ReAct paradigm — interleaving reasoning traces with action steps for LLMs. |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Wei et al. | 2022 | Foundational CoT paper — shows step-by-step reasoning improves LLM performance. |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Yao et al. | 2023 | Extends CoT to explore multiple reasoning paths via tree search. |
| [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning](https://arxiv.org/abs/2305.04091) | Wang et al. | 2023 | Separates planning and execution phases for more reliable reasoning. |
| [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200) | Wang et al. | 2024 | CoT reasoning emerges without explicit prompting in large models. |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Shinn et al. | 2023 | Agents reflect on failures and improve via verbal reinforcement. |

### Tool Use & Function Calling

| Paper | Authors | Year | Summary |
|-------|---------|------|---------|
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | Schick et al. | 2023 | LLMs learn to use APIs via self-supervised learning. |
| [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) | Patil et al. | 2023 | Fine-tuned LLM for accurate API calls. |
| [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354) | Qin et al. | 2023 | Comprehensive survey of tool-augmented LLMs. |

### Agents & Frameworks

| Paper | Authors | Year | Summary |
|-------|---------|------|---------|
| [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) | Wu et al. | 2023 | Introduces AutoGen's multi-agent conversation framework. |
| [Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) | Qian et al. | 2023 | ChatDev — multi-agent system for software development. |
| [The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling](https://arxiv.org/abs/2404.11584) | Kapoor et al. | 2024 | Survey of agent architectures. |

### MCP

| Resource | Link |
|----------|------|
| MCP Specification | https://spec.modelcontextprotocol.io |
| MCP GitHub (Anthropic) | https://github.com/modelcontextprotocol |
| MCP Introduction | https://modelcontextprotocol.io/introduction |

## Framework Documentation

| Framework | Documentation | GitHub |
|-----------|---------------|--------|
| **LangGraph** | https://langchain-ai.github.io/langgraph/ | https://github.com/langchain-ai/langgraph |
| **AutoGen** | https://microsoft.github.io/autogen/ | https://github.com/microsoft/autogen |
| **CrewAI** | https://docs.crewai.com/ | https://github.com/crewAIInc/crewAI |
| **Semantic Kernel** | https://learn.microsoft.com/en-us/semantic-kernel/ | https://github.com/microsoft/semantic-kernel |
| **OpenAI Assistants API** | https://platform.openai.com/docs/assistants/overview | N/A (API reference) |

## Tutorials & Blog Posts

### Getting Started
- [Building an Agent with OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [AutoGen Getting Started](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/quickstart.html)

### Deep Dives
- [CrewAI: How to Build Multi-Agent Systems](https://docs.crewai.com/core-concepts/Crews/)
- [Building with MCP Servers](https://modelcontextprotocol.io/quickstart/server)
- [ReAct Pattern in Practice](https://www.promptingguide.ai/techniques/react)
- [Memory in LLM Agents](https://blog.langchain.dev/memory-for-agents/)

### Architecture & Design
- [What Are AI Agents? (Andrej Karpathy)](https://www.youtube.com/watch?v=fT3/svEFBsU) — Video talk
- [The Agentic AI Landscape (Microsoft)](https://azure.microsoft.com/en-us/blog/the-future-of-agentic-ai/)
- [MCP Explained (Anthropic)](https://www.anthropic.com/news/model-context-protocol)
- [Building Reliable Agents (LangChain)](https://blog.langchain.dev/building-reliable-agents/)

## Tools & SDKs

| Tool | Purpose |
|------|---------|
| [MCP Servers Directory](https://github.com/modelcontextprotocol/servers) | Curated list of MCP servers |
| [LangSmith](https://smith.langchain.com/) | Observability for LangGraph/LangChain agents |
| [Pinecone](https://www.pinecone.io/) | Vector database for long-term memory |
| [Chroma](https://www.trychroma.com/) | Open-source vector database |
| [Qdrant](https://qdrant.tech/) | Vector search engine |
| [Neo4j](https://neo4j.com/) | Knowledge graph database |

## Books

- *Building Large Language Model Applications* by O'Reilly (2024) — Covers agent patterns with LangChain.
- *AI Agents in Practice* by Manning (2025) — Practical guide to building production agents.
- *Natural Language Processing with Transformers* by Tunstall et al. — Background on transformer architectures used in agents.

## Community & Events

- [LangChain Discord](https://discord.gg/langchain)
- [AutoGen Discord](https://discord.gg/autogen)
- [Anthropic Research Blog](https://www.anthropic.com/research)
- [OpenAI Research Blog](https://openai.com/research/)
- [AI Engineer Conference](https://www.ai.engineer/) — Annual conference on AI engineering practices.

## How to Stay Current

1. **Follow arXiv**: Subscribe to cs.CL and cs.AI categories.
2. **Twitter/X**: Follow @katherinecrawford @AndrewYng @suchenzang @ylaboratory for agent research.
3. **Newsletters**: The Batch (Andrew Ng), Interconnects (Nathan Lambert), AI Breakfast.
4. **GitHub**: Watch the repos above for new releases.
5. **Papers with Code**: Track agent benchmarks (e.g., SWE-bench, GAIA, ToolBench).
