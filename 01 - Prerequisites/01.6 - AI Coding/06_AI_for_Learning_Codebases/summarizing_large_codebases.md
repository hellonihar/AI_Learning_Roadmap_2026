# Summarizing Large Codebases

When you open a large project for the first time, AI can provide a high-level overview.

## Prompt

```
"Summarize this project. What problem does it solve?
What are the main directories and what does each contain?
What's the entry point? What's the architecture?"
```

## Example

```
"You are an AI codebase navigator. I'm looking at:
https://github.com/langchain-ai/langchain

Give me:
1. A 3-sentence summary of what this project does
2. The top-level directory structure and what each folder contains
3. The entry point for building a simple RAG chain
4. Key abstractions I need to understand"
```

**AI response:**
```
1. LangChain is a framework for building applications with LLMs.
   It provides abstractions for chains, agents, retrievers, and
   memory. It supports dozens of LLM providers and vector stores.

2. libs/langchain/   → core library
   libs/core/        → base abstractions (ABCs)
   libs/community/   → integrations (providers, vector stores)
   cookbook/         → example notebooks

3. Entry point for RAG:
   from langchain.chains import create_retrieval_chain

4. Key abstractions: BaseLLM, BaseRetriever, BaseMemory,
   BaseChatMessageHistory, Document, VectorStore
```

## When to Use

- Joining a new team with an established codebase
- Evaluating an open-source project
- Before attempting a cross-cutting change
- Architectural decision-making (understand existing patterns)
